//! Decode-lane bit gate. KL-tolerance coverage alone is not enough for
//! the step (autoregressive) path — tolerance cannot distinguish a
//! bit-identical fusion from lucky rounding. This records FNV hashes of
//! 16 chained decode steps (output after every step + the carried conv
//! and SSM state at the end) on synthetic weights, per tier and graph
//! mode. Record on a good build; compare after any step-path edit.
//!
//!   cargo test --release --features cuda --test decode_digest -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/decode_snapshot.rs"]
mod decode_snapshot;
#[path = "common/digest.rs"]
mod digest;
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/evidence_digest.rs"]
mod evidence_digest;

use decode_snapshot::{DecodeSnapshot, validate_decode_pair};
use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::inference::{GpuMambaInference, GpuMambaInferenceMixed};
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

fn config() -> MambaConfig {
    MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    }
}

fn report_route(lane: &str, arm: &str, ctx: &GpuCtx) {
    let mode = ctx.gemm_mode();
    eprintln!(
        "DECODE-ROUTE suite=decode_digest lane={lane} arm={arm} mode={} route={:?}",
        mode.as_str(),
        ctx.gemm_route()
    );
    assert_eq!(mode, GemmMode::Deterministic, "decode fixture GEMM mode");
}

fn report_f32(graph: bool, snapshot: &DecodeSnapshot) {
    // Sequential absorption avoids XOR cancellation, while the per-step lines
    // identify which step moved when the chain changes.
    let mut chain = digest::Digest::new();
    for (step_i, out) in snapshot.outputs.iter().enumerate() {
        eprintln!(
            "DECODE-DIGEST step={step_i} {:016x}",
            digest::fnv1a_f32(out)
        );
        chain.absorb_bytes(b"step");
        chain.absorb_f32(out);
    }
    let chained = chain.finish();
    let h_conv = digest::fnv1a_f32(&snapshot.states[0].1);
    let h_ssm = digest::fnv1a_f32(&snapshot.states[1].1);
    eprintln!(
        "DECODE-DIGEST graph={graph} steps=16 d384 L24: out_chain={chained:016x} \
         conv={h_conv:016x} ssm={h_ssm:016x}"
    );
    let arm = if graph { "graph" } else { "eager" };
    evidence_digest::record_digest("decode_digest", arm, "out_chain", chained)
        .expect("acceptance evidence");
    evidence_digest::record_digest("decode_digest", arm, "conv", h_conv)
        .expect("acceptance evidence");
    evidence_digest::record_digest("decode_digest", arm, "ssm", h_ssm)
        .expect("acceptance evidence");
}

fn report_mixed(lane: &str, arm: &str, snapshot: &DecodeSnapshot) {
    let mut chain = digest::Digest::new();
    for (step_i, out) in snapshot.outputs.iter().enumerate() {
        eprintln!(
            "DECODE-DIGEST m1 {lane} {arm} step={step_i} {:016x}",
            digest::fnv1a_f32(out)
        );
        chain.absorb_bytes(b"step");
        chain.absorb_f32(out);
    }
    let chained = chain.finish();
    let h_conv = digest::fnv1a_f32(&snapshot.states[0].1);
    let h_ssm = digest::fnv1a_f32(&snapshot.states[1].1);
    eprintln!(
        "DECODE-DIGEST m1 {lane} {arm} steps=16 d384 L24: out_chain={chained:016x} \
         conv={h_conv:016x} ssm={h_ssm:016x}"
    );
    let evidence_arm = format!("{lane}/{arm}");
    evidence_digest::record_digest("decode_digest", &evidence_arm, "out_chain", chained)
        .expect("acceptance evidence");
    evidence_digest::record_digest("decode_digest", &evidence_arm, "conv", h_conv)
        .expect("acceptance evidence");
    evidence_digest::record_digest("decode_digest", &evidence_arm, "ssm", h_ssm)
        .expect("acceptance evidence");
}

#[test]
#[ignore = "bit-gate recorder"]
fn decode_run_digest() {
    let cfg = config();
    let input_dim = 384usize;
    let batch = 1usize;
    let mut w = MambaWeights::init(&cfg, input_dim, 0xDEC0DE);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let device = GpuDevice::new(0).unwrap();

    let mut snapshots = Vec::new();
    for graph in [false, true] {
        let engine = GpuMambaInference::new(&device, &w, cfg, input_dim, batch).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let mut out = vec![f32::NAN; batch * cfg.d_model];
        let mut engine = engine;
        let arm = if graph { "graph" } else { "eager" };
        report_route("f32", arm, engine.ctx());
        if graph {
            // One warm step before capture, per the capture contract.
            out.fill(f32::NAN);
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
        let mut outputs = Vec::with_capacity(16);
        for step_i in 0..16u32 {
            let input = det(batch * input_dim, 1000 + step_i, 0.1);
            out.fill(f32::NAN);
            engine
                .step(&input, &mut out, &mut state, &mut scratch)
                .unwrap();
            outputs.push(out.clone());
        }
        let mut conv_v = vec![0f32; state.conv.len()];
        let mut ssm_v = vec![0f32; state.ssm.len()];
        state
            .conv
            .download(&engine.ctx().stream, &mut conv_v)
            .unwrap();
        state
            .ssm
            .download(&engine.ctx().stream, &mut ssm_v)
            .unwrap();
        engine.ctx().stream.synchronize().unwrap();
        snapshots.push((
            graph,
            DecodeSnapshot {
                outputs,
                states: vec![("conv", conv_v), ("ssm", ssm_v)],
            },
        ));
    }
    validate_decode_pair(&snapshots[0].1, &snapshots[1].1).unwrap();
    for (graph, snapshot) in &snapshots {
        assert_eq!(snapshot.outputs.len(), 16, "decode step count");
        report_f32(*graph, snapshot);
    }
}

fn decode_run_digest_mixed(dtype: WeightDtype) {
    let lane = match dtype {
        WeightDtype::Bf16 => "bf16",
        WeightDtype::F16 => "f16",
        WeightDtype::F32 | WeightDtype::Tf32 => panic!("mixed decode helper rejects f32 storage"),
    };
    let cfg = config();
    let input_dim = 384usize;
    let batch = 1usize;
    let mut w = MambaWeights::init(&cfg, input_dim, 0xDEC0DE);
    for lw in &mut w.layers {
        lw.a_neg = lw.a_log.iter().map(|&value| -value.exp()).collect();
    }
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let device = GpuDevice::new(0).unwrap();

    let mut snapshots = Vec::new();
    for graph in [false, true] {
        let engine =
            GpuMambaInferenceMixed::new(&device, &w, cfg, input_dim, batch, dtype).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_mixed_scratch().unwrap();
        let mut out = vec![f32::NAN; batch * cfg.d_model];
        let mut engine = engine;
        let arm = if graph { "graph" } else { "eager" };
        report_route(lane, arm, engine.ctx());
        if graph {
            out.fill(f32::NAN);
            engine
                .step_mixed_native(
                    &det(batch * input_dim, 900, 0.1),
                    &mut out,
                    &mut state,
                    &mut scratch,
                )
                .unwrap();
            unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }.unwrap();
            state.reset(engine.stream()).unwrap();
        }
        let mut outputs = Vec::with_capacity(16);
        for step_i in 0..16u32 {
            let input = det(batch * input_dim, 1000 + step_i, 0.1);
            out.fill(f32::NAN);
            engine
                .step_mixed_native(&input, &mut out, &mut state, &mut scratch)
                .unwrap();
            outputs.push(out.clone());
        }
        let mut conv_v = vec![0f32; state.conv.len()];
        let mut ssm_v = vec![0f32; state.ssm.len()];
        state.conv.download(engine.stream(), &mut conv_v).unwrap();
        state.ssm.download(engine.stream(), &mut ssm_v).unwrap();
        engine.stream().synchronize().unwrap();
        snapshots.push(DecodeSnapshot {
            outputs,
            states: vec![("conv", conv_v), ("ssm", ssm_v)],
        });
    }
    validate_decode_pair(&snapshots[0], &snapshots[1]).unwrap();
    for (arm, snapshot) in ["eager", "graph"].into_iter().zip(&snapshots) {
        assert_eq!(snapshot.outputs.len(), 16, "decode step count");
        report_mixed(lane, arm, snapshot);
    }
}

#[test]
#[ignore = "bit-gate recorder"]
fn decode_run_digest_bf16() {
    decode_run_digest_mixed(WeightDtype::Bf16);
}

#[test]
#[ignore = "bit-gate recorder"]
fn decode_run_digest_f16() {
    decode_run_digest_mixed(WeightDtype::F16);
}
