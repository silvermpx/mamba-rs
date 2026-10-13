#![cfg(feature = "cuda")]

//! Mamba-2 and Mamba-3 MIMO on the GPU: prompt windows, decode steps and
//! training steps, the sequential scan against the chunked one, in f32
//! and bf16, at the released models' head shapes.
//!
//! Run: `cargo bench --bench m2_mimo_gpu_bench --features cuda`; a first
//! argument `m2` or `mimo` picks a model, a second `prefill`, `decode` or
//! `train` one part of it.
//!
//! Every time is wall clock around the public call, so it includes the
//! host upload of the input and the download of the rows; both scans pay
//! the same transfers.

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::train::TrainSessionCfg;
use std::time::Instant;

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba2::gpu::{Mamba2GpuEngine, Mamba2GpuTrainer};
use mamba_rs::mamba2::{Mamba2Config, Mamba2Weights};
use mamba_rs::mamba3_mimo::gpu::{Mamba3MimoGpuEngine, Mamba3MimoGpuTrainer};
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};

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

fn modes() -> [(ScanMode, &'static str); 2] {
    [
        (ScanMode::Sequential, "seq"),
        (ScanMode::Parallel, "chunked"),
    ]
}

fn dtypes() -> [WeightDtype; 2] {
    [WeightDtype::F32, WeightDtype::Bf16]
}

/// Whether the part the command line picked (none: every part) is `name`.
fn wants(part: &str, name: &str) -> bool {
    part.is_empty() || part == name
}

fn mamba2(device: &GpuDevice, part: &str) {
    // mamba2-130m's mixer: 24 heads of 64 x 128, one group.
    let base = Mamba2Config {
        d_model: 768,
        d_state: 128,
        headdim: 64,
        n_layers: LAYERS,
        ..Mamba2Config::default()
    };
    let input_dim = base.d_model;
    let w = Mamba2Weights::init(&base, input_dim, 7);
    println!("--- Mamba-2, d_model 768, 24 heads 64x128, {LAYERS} layers ---");
    println!("prompt windows (tokens/s):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "prefill")) {
        for &(b, t) in &PREFILL {
            let mut line = format!("  {dtype:?} B={b} T={t}:");
            for (mode, name) in modes() {
                let cfg = Mamba2Config {
                    scan_mode: mode,
                    ..base
                };
                let eng = Mamba2GpuEngine::new(device, &w, cfg, input_dim, b, dtype).unwrap();
                let mut st = eng.alloc_state().unwrap();
                let mut pf = eng.alloc_prefill(t).unwrap();
                let input = det(b * t * input_dim, 11);
                let mut rows = vec![0f32; b * t * cfg.d_model];
                let us = median_us(2, 7, || {
                    st.reset(eng.ctx()).unwrap();
                    eng.prefill(&input, &mut rows, &mut st, &mut pf).unwrap();
                });
                line += &format!("  {name} {:>10.0}", (b * t) as f64 / us * 1e6);
            }
            println!("{line}");
        }
    }
    println!("decode steps (us/step):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "decode")) {
        for &b in &DECODE_BATCH {
            let mut eng = Mamba2GpuEngine::new(device, &w, base, input_dim, b, dtype).unwrap();
            let mut st = eng.alloc_state().unwrap();
            let mut sc = eng.alloc_scratch().unwrap();
            let input = det(b * input_dim, 12);
            let mut out = vec![0f32; b * base.d_model];
            let eager = median_us(20, 200, || {
                eng.step(&input, &mut out, &mut st, &mut sc).unwrap();
            });
            unsafe { eng.capture_graph(&mut st, &mut sc) }.unwrap();
            let graph = median_us(20, 200, || {
                eng.step(&input, &mut out, &mut st, &mut sc).unwrap();
            });
            println!("  {dtype:?} B={b}: eager {eager:>8.1}  graph {graph:>8.1}");
        }
    }
    let (b, t) = TRAIN;
    println!("training steps, B={b} T={t}, forward + backward (ms/step):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "train")) {
        let mut line = format!("  {dtype:?}:");
        for (mode, name) in modes() {
            let cfg = Mamba2Config {
                scan_mode: mode,
                ..base
            };
            let mut tr = Mamba2GpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, b, t),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            let input = det(b * t * input_dim, 13);
            let d_out = det(b * t * cfg.d_model, 14);
            let mut rows = vec![0f32; b * t * cfg.d_model];
            let us = median_us(2, 7, || {
                tr.reset_state().unwrap();
                tr.forward(&input, &mut rows).unwrap();
                tr.backward(&d_out, None).unwrap();
            });
            line += &format!("  {name} {:>8.2}", us / 1e3);
        }
        println!("{line}");
    }
    println!();
}

fn mimo(device: &GpuDevice, part: &str) {
    // The released MIMO mixer: heads of 64 x 128 at rank 4.
    let base = Mamba3MimoConfig {
        d_model: 768,
        d_state: 128,
        headdim: 64,
        mimo_rank: 4,
        n_layers: LAYERS,
        ..Mamba3MimoConfig::default()
    };
    let input_dim = base.d_model;
    let w = Mamba3MimoWeights::init(&base, input_dim, 7);
    println!("--- Mamba-3 MIMO, d_model 768, 24 heads 64x128, rank 4, {LAYERS} layers ---");
    println!("prompt windows (tokens/s):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "prefill")) {
        for &(b, t) in &PREFILL {
            let mut line = format!("  {dtype:?} B={b} T={t}:");
            for (mode, name) in modes() {
                let cfg = Mamba3MimoConfig {
                    scan_mode: mode,
                    ..base
                };
                let eng = Mamba3MimoGpuEngine::new(device, &w, cfg, input_dim, b, dtype).unwrap();
                let mut st = eng.alloc_state().unwrap();
                let mut win = eng.alloc_window(t).unwrap();
                let input = det(b * t * input_dim, 11);
                let mut rows = vec![0f32; b * t * cfg.d_model];
                let us = median_us(2, 7, || {
                    st = eng.alloc_state().unwrap();
                    eng.run(&input, &mut rows, &mut st, &mut win).unwrap();
                });
                line += &format!("  {name} {:>10.0}", (b * t) as f64 / us * 1e6);
            }
            println!("{line}");
        }
    }
    println!("decode steps (us/step):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "decode")) {
        for &b in &DECODE_BATCH {
            let eng = Mamba3MimoGpuEngine::new(device, &w, base, input_dim, b, dtype).unwrap();
            let mut st = eng.alloc_state().unwrap();
            let mut win = eng.alloc_window(1).unwrap();
            let input = det(b * input_dim, 12);
            let mut out = vec![0f32; b * base.d_model];
            let eager = median_us(20, 200, || {
                eng.run(&input, &mut out, &mut st, &mut win).unwrap();
            });
            unsafe { eng.capture_graph(&mut st, &mut win) }.unwrap();
            let graph = median_us(20, 200, || {
                eng.run(&input, &mut out, &mut st, &mut win).unwrap();
            });
            println!("  {dtype:?} B={b}: eager {eager:>8.1}  graph {graph:>8.1}");
        }
    }
    let (b, t) = TRAIN;
    println!("training steps, B={b} T={t}, forward + backward (ms/step):");
    for dtype in dtypes().into_iter().filter(|_| wants(part, "train")) {
        let mut line = format!("  {dtype:?}:");
        for (mode, name) in modes() {
            let cfg = Mamba3MimoConfig {
                scan_mode: mode,
                ..base
            };
            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, b, t),
                GpuOptions::default().dtype(dtype),
                16,
            )
            .unwrap();
            let input = det(b * t * input_dim, 13);
            let d_out = det(b * t * cfg.d_model, 14);
            let mut rows = vec![0f32; b * t * cfg.d_model];
            let us = median_us(2, 7, || {
                tr.reset_state().unwrap();
                tr.forward(&input, &mut rows).unwrap();
                tr.backward(&d_out, None).unwrap();
            });
            line += &format!("  {name} {:>8.2}", us / 1e3);
        }
        println!("{line}");
    }
    println!();
}

fn main() {
    let device = GpuDevice::new(0).unwrap();
    // cargo bench passes `--bench`; the first other argument picks a model,
    // the second a part.
    let picks: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let only = picks.first().map_or("", String::as_str);
    let part = picks.get(1).map_or("", String::as_str);
    if wants(only, "m2") {
        mamba2(&device, part);
    }
    if wants(only, "mimo") {
        mimo(&device, part);
    }
}
