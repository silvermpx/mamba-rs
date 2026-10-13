#![cfg(feature = "cuda")]

//! Mamba-1 on the GPU at a released model's mixer shape: prompt windows
//! (the parallel scan against the sequential one), decode steps and
//! training steps, in f32 and bf16, plus the vision-classifier serve page.
//!
//! Run: `cargo bench --bench m1_mixer_gpu_bench --features cuda`; a first
//! argument `prefill`, `serve`, `decode` or `train` picks one part.
//!
//! Prompt windows time the device prefill from embedded rows already on the
//! GPU (stream-synchronized); decode and training time the public call,
//! host transfers included.

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::train::TrainSessionCfg;
use std::time::Instant;

use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba1::gpu::forward::GpuMambaDims;
use mamba_rs::mamba1::gpu::inference::{MambaGpuBackbone, MambaGpuState};
use mamba_rs::mamba1::gpu::prefill::{
    PrefillOutputs, PrefillRawInputs, gpu_forward_inference_prefill_from_raw,
};
use mamba_rs::mamba1::gpu::trainer::MambaGpuTrainer;
use mamba_rs::mamba1::gpu::weights::GpuMambaWeights;
use mamba_rs::{config::ScanMode, mamba1::MambaConfig};

const LAYERS: usize = 4;
const PREFILL: [(usize, usize); 3] = [(1, 1024), (1, 4096), (8, 1024)];
const DECODE_BATCH: [usize; 2] = [1, 16];
const TRAIN: (usize, usize) = (4, 512);

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

/// The median of `iters` timed calls after `warm` untimed ones, in
/// microseconds.
fn median_us(warm: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
    }
    let mut t: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

/// Whether the part the command line picked (none: every part) is `name`.
fn wants(part: &str, name: &str) -> bool {
    part.is_empty() || part == name
}

/// mamba-130m's mixer: d_model 768, d_state 16, expand 2.
fn mixer(scan_mode: ScanMode) -> MambaConfig {
    MambaConfig {
        d_model: 768,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: LAYERS,
        scan_mode,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

/// tokens/s of the device prefill over `b` sequences of `t` rows.
fn prefill_rate(device: &GpuDevice, cfg: MambaConfig, input_dim: usize, b: usize, t: usize) -> f64 {
    let mut w = MambaWeights::init(&cfg, input_dim, 7);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let (di, ds, nl) = (cfg.d_inner(), cfg.d_state, cfg.n_layers);
    let dims = GpuMambaDims {
        batch: b,
        d_model: cfg.d_model,
        d_inner: di,
        d_state: ds,
        d_conv: cfg.d_conv,
        dt_rank: cfg.dt_rank(),
        xdbl_dim: cfg.xdbl_dim(),
        seq_len: t,
        mamba_input_dim: input_dim,
        n_layers: nl,
        scan_mode: cfg.scan_mode,
        rms_norm_eps: cfg.rms_norm_eps,
        bcdt_rms_eps: None,
    };
    let ctx = GpuCtx::new_with_state_cap(device, ds).unwrap();
    let weights = GpuMambaWeights::from_cpu(&ctx.stream, &w, &cfg).unwrap();
    let mut scratch = GpuMambaTargetScratch::new(&ctx.stream, &dims).unwrap();
    let mut state = MambaGpuState::zeros(&ctx.stream, b, &cfg).unwrap();
    let a_neg_flat: Vec<f32> = w.layers.iter().flat_map(|lw| lw.a_neg.clone()).collect();
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut input = GpuBuffer::zeros(&ctx.stream, b * t * input_dim).unwrap();
    input
        .upload(&ctx.stream, &det(b * t * input_dim, 11))
        .unwrap();
    let mut last = GpuBuffer::zeros(&ctx.stream, b * cfg.d_model).unwrap();
    let us = median_us(2, 7, || {
        state.reset(&ctx.stream).unwrap();
        gpu_forward_inference_prefill_from_raw(
            &ctx,
            PrefillOutputs {
                last_temporal: &mut last,
                full_temporal: None,
            },
            PrefillRawInputs {
                input_flat: &input,
                weights: &weights,
                a_neg_all: &a_neg,
            },
            &mut state,
            &mut scratch,
        )
        .unwrap();
        ctx.stream.synchronize().unwrap();
    });
    (b * t) as f64 / us * 1e6
}

fn prefill(device: &GpuDevice) {
    println!("--- Mamba-1, d_model 768, d_state 16, {LAYERS} layers ---");
    println!("prompt windows, f32 (tokens/s):");
    for &(b, t) in &PREFILL {
        let mut line = format!("  B={b} T={t}:");
        for (mode, name) in [
            (ScanMode::Sequential, "seq"),
            (ScanMode::Parallel, "parallel"),
        ] {
            let rate = prefill_rate(device, mixer(mode), 768, b, t);
            line += &format!("  {name} {rate:>10.0}");
        }
        println!("{line}");
    }
}

/// The vision-classifier page: 4621 rows of 1024 features through 24
/// layers of d_model 384.
fn serve(device: &GpuDevice) {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        ..mixer(ScanMode::Auto)
    };
    let rate = prefill_rate(device, cfg, 1024, 1, 4621);
    println!(
        "serve page, f32, T=4621: {:.3} ms/page",
        4621.0 / rate * 1e3
    );
}

fn decode() {
    let cfg = mixer(ScanMode::Auto);
    let w = MambaWeights::init(&cfg, cfg.d_model, 7);
    println!("decode steps (us/step):");
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        for &b in &DECODE_BATCH {
            let mut bb =
                MambaGpuBackbone::new(&w, cfg, cfg.d_model, b, GpuOptions::default().dtype(dtype))
                    .unwrap();
            let input = det(b * cfg.d_model, 12);
            let mut out = vec![0f32; b * cfg.d_model];
            let eager = median_us(20, 200, || bb.step(&input, &mut out).unwrap());
            bb.capture_graph().unwrap();
            let graph = median_us(20, 200, || bb.step(&input, &mut out).unwrap());
            println!("  {dtype:?} B={b}: eager {eager:>8.1}  graph {graph:>8.1}");
        }
    }
}

fn train() {
    let (b, t) = TRAIN;
    println!("training steps, B={b} T={t}, forward + backward (ms/step):");
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let mut line = format!("  {dtype:?}:");
        for (mode, name) in [
            (ScanMode::Sequential, "seq"),
            (ScanMode::Parallel, "parallel"),
        ] {
            let cfg = mixer(mode);
            let w = MambaWeights::init(&cfg, cfg.d_model, 7);
            let mut tr = MambaGpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(cfg.d_model, b, t),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            let input = det(b * t * cfg.d_model, 13);
            let d_out = det(b * t * cfg.d_model, 14);
            let us = median_us(2, 7, || {
                tr.step(&input, &d_out).unwrap();
            });
            line += &format!("  {name} {:>8.2}", us / 1e3);
        }
        println!("{line}");
    }
}

fn main() {
    let part = std::env::args()
        .skip(1)
        .find(|arg| !arg.starts_with('-'))
        .unwrap_or_default();
    let device = GpuDevice::new(0).unwrap();
    if wants(&part, "prefill") {
        prefill(&device);
    }
    if wants(&part, "serve") {
        serve(&device);
    }
    if wants(&part, "decode") {
        decode();
    }
    if wants(&part, "train") {
        train();
    }
}
