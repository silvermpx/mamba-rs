//! Bit gate for the Mamba-3 SISO kernels that give each element of a head
//! its own thread: FNV hashes of what they produce at the head widths the
//! warp kernels do not serve, so a change to how those kernels walk a head
//! can be shown to keep the bits. The roads: prefill and decode steps on
//! the f32, bf16 and f16 engines (the decode step's block tier, and the
//! chunked forward's fallback tiles where the fused ones do not fit), and
//! two training steps of the f32 and bf16 trainers on the sequential scan
//! (its block-tier forward and backward) and on the chunked one at the
//! widest heads. The gated output norm runs on every road. Record on a good
//! build; compare after any edit of those kernels.
//!
//!   cargo test --release --features cuda --test m3_head_width_digest -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/digest.rs"]
mod digest;

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::{Mamba3GpuEngine, Mamba3GpuEngineMixed};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::{GpuOptions, Mamba3GpuTrainer, TrainSessionCfg};

const PROMPT: usize = 70;
const STEPS: u32 = 3;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Heads of 48, 100, 64 and 512 with states of 16, 24, 128 and 64:
/// specialized and generic state widths on the sequential kernels, and at
/// the official 64 by 128 and at 512 heads whose fused chunked forward does
/// not fit shared memory.
fn configs() -> [Mamba3Config; 4] {
    let base = Mamba3Config {
        n_layers: 2,
        ngroups: 1,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    };
    [
        Mamba3Config {
            d_model: 96,
            headdim: 48,
            d_state: 16,
            ..base
        },
        Mamba3Config {
            d_model: 100,
            headdim: 100,
            d_state: 24,
            ..base
        },
        Mamba3Config {
            d_model: 128,
            headdim: 64,
            d_state: 128,
            ..base
        },
        Mamba3Config {
            d_model: 256,
            headdim: 512,
            d_state: 64,
            ..base
        },
    ]
}

fn label(cfg: &Mamba3Config) -> String {
    format!("hd{}_ds{}", cfg.headdim, cfg.d_state)
}

/// Print one road's chained digest, and each part's own hash so a moved
/// chain names the part that moved.
fn report(road: &str, parts: &[(&str, &[f32])]) {
    let mut d = digest::Digest::new();
    for (i, (name, values)) in parts.iter().enumerate() {
        eprintln!(
            "HEAD-WIDTH-PART {road} {i}:{name} {:016x}",
            digest::fnv1a_f32(values)
        );
        d.absorb_bytes(name.as_bytes());
        d.absorb_f32(values);
    }
    eprintln!("HEAD-WIDTH-DIGEST {road} {:016x}", d.finish());
}

fn download(stream: &std::sync::Arc<cudarc::driver::CudaStream>, buf: &GpuBuffer) -> Vec<f32> {
    let mut v = vec![0f32; buf.len()];
    buf.download(stream, &mut v).unwrap();
    stream.synchronize().unwrap();
    v
}

/// Prefill a prompt, then decode a few steps; hash the prefill's last
/// hidden state, every step's output and the states left behind.
#[test]
#[ignore = "bit-gate recorder"]
fn m3_head_width_inference_digest() {
    let device = GpuDevice::new(0).unwrap();
    for cfg in configs() {
        let dm = cfg.d_model;
        let mut w = Mamba3Weights::init(&cfg, dm, 0x4EAD);
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        let prompt = det(PROMPT * dm, 0x0F00, 0.5);

        let engine = Mamba3GpuEngine::new(&device, &w, cfg, dm, 1).unwrap();
        let stream = engine.ctx().stream.clone();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let mut prefill = engine.alloc_prefill(PROMPT).unwrap();
        let gpu_prompt = GpuBuffer::from_cpu(&stream, &prompt).unwrap();
        let mut last = GpuBuffer::zeros(&stream, dm).unwrap();
        engine
            .prefill_sequence(
                &mut prefill,
                &gpu_prompt,
                PROMPT,
                &mut state,
                false,
                &mut last,
            )
            .unwrap();
        let mut outs = vec![download(&stream, &last)];
        for i in 0..STEPS {
            let mut out = vec![f32::NAN; dm];
            engine
                .step(&det(dm, 0x100 + i, 0.5), &mut out, &mut state, &mut scratch)
                .unwrap();
            outs.push(out);
        }
        let ssm = download(&stream, &state.ssm_state);
        let mut parts: Vec<(&str, &[f32])> = outs.iter().map(|o| ("out", o.as_slice())).collect();
        parts.push(("ssm", &ssm));
        report(&format!("infer f32 {}", label(&cfg)), &parts);

        for (lane, dtype) in [("bf16", WeightDtype::Bf16), ("f16", WeightDtype::F16)] {
            let engine = Mamba3GpuEngineMixed::new(&device, &w, cfg, dm, 1, dtype).unwrap();
            let stream = engine.ctx_stream().clone();
            let mut state = engine.alloc_state().unwrap();
            let mut scratch = engine.alloc_mixed_scratch().unwrap();
            let mut prefill = engine.alloc_prefill(PROMPT).unwrap();
            let gpu_prompt = GpuBuffer::from_cpu(&stream, &prompt).unwrap();
            let mut last = GpuBuffer::zeros(&stream, dm).unwrap();
            engine
                .prefill_sequence(
                    &mut prefill,
                    &gpu_prompt,
                    PROMPT,
                    &mut state,
                    false,
                    &mut last,
                )
                .unwrap();
            let mut outs = vec![download(&stream, &last)];
            for i in 0..STEPS {
                let mut out = vec![f32::NAN; dm];
                engine
                    .step_mixed_native(&det(dm, 0x100 + i, 0.5), &mut out, &mut state, &mut scratch)
                    .unwrap();
                outs.push(out);
            }
            let ssm = download(&stream, &state.ssm_state);
            let mut parts: Vec<(&str, &[f32])> =
                outs.iter().map(|o| ("out", o.as_slice())).collect();
            parts.push(("ssm", &ssm));
            report(&format!("infer {lane} {}", label(&cfg)), &parts);
        }
    }
}

/// Two training steps on each scan; hash the master weights after them.
/// The chunked scan runs at the heads of 64 by 128 and of 512, whose
/// forward takes the fallback tiles.
#[test]
#[ignore = "bit-gate recorder"]
fn m3_head_width_training_digest() {
    const BATCH: usize = 2;
    const SEQ: usize = 64;
    for cfg in configs() {
        let scans: &[(&str, ScanMode)] = if cfg.headdim >= 64 {
            &[("seq", ScanMode::Sequential), ("chunked", ScanMode::Auto)]
        } else {
            &[("seq", ScanMode::Sequential)]
        };
        for &(scan, scan_mode) in scans {
            let cfg = Mamba3Config { scan_mode, ..cfg };
            let dm = cfg.d_model;
            let w = Mamba3Weights::init(&cfg, dm, 0x7A11);
            let input = det(BATCH * SEQ * dm, 0x200, 0.5);
            let d_out = det(BATCH * SEQ * dm, 0x300, 0.05);
            for (lane, dtype) in [("f32", WeightDtype::F32), ("bf16", WeightDtype::Bf16)] {
                let mut t = Mamba3GpuTrainer::new(
                    &w,
                    cfg,
                    TrainSessionCfg {
                        input_dim: dm,
                        batch: BATCH,
                        seq_len: SEQ,
                        lr: 1e-3,
                        weight_decay: 0.1,
                    },
                    GpuOptions::default().dtype(dtype),
                )
                .unwrap();
                for _ in 0..2 {
                    t.step(&input, &d_out).unwrap();
                }
                let m = t.snapshot_master().unwrap();
                let mut parts: Vec<(&str, &[f32])> = vec![
                    ("input_proj_w", &m.input_proj_w),
                    ("input_proj_b", &m.input_proj_b),
                ];
                for l in &m.layers {
                    parts.extend([
                        ("norm", l.norm_weight.as_slice()),
                        ("in_proj", &l.in_proj_w),
                        ("dt_bias", &l.dt_bias),
                        ("b_norm", &l.b_norm_weight),
                        ("c_norm", &l.c_norm_weight),
                        ("b_bias", &l.b_bias),
                        ("c_bias", &l.c_bias),
                        ("d", &l.d_param),
                        ("gate_norm", &l.norm_gate_weight),
                        ("out_proj", &l.out_proj_w),
                    ]);
                }
                parts.push(("norm_f", &m.norm_f_weight));
                report(&format!("train {scan} {lane} {}", label(&cfg)), &parts);
            }
        }
    }
}
