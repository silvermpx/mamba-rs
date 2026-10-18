#![cfg(feature = "cuda")]

//! The four families side by side at one size, through the API they share:
//! 4 layers of d_model 768, Mamba-2, Mamba-3 SISO and MIMO with heads of
//! 64 x 128 (MIMO rank 4), Mamba-1 with d_state 16 as the released
//! mamba-130m. Each family runs through its `GpuBackbone` and `GpuTrainer`:
//! decode steps at batch 1 and 16 through a captured graph, a prompt of 4096
//! tokens in one pass from host rows, and a full training step (forward,
//! backward and the AdamW update) at batch 4 by 512 tokens.
//!
//! Run: `cargo bench --bench family_compare_bench --features cuda`

use std::time::Instant;

use mamba_rs::{
    GpuBackbone, GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuBackbone, Mamba2GpuTrainer,
    Mamba2Weights, Mamba3Config, Mamba3GpuBackbone, Mamba3GpuTrainer, Mamba3MimoConfig,
    Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights, MambaConfig,
    MambaGpuBackbone, MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
};

const LAYERS: usize = 4;
const D_MODEL: usize = 768;
const TRAIN: (usize, usize) = (4, 512);
const PROMPT: usize = 4096;

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

/// One row of the table: decode B1, decode B16 (us/step, graph), the
/// prompt window (tokens/s) and the training step (ms), or why a part did
/// not run.
struct Row {
    decode1: Result<f64, String>,
    decode16: Result<f64, String>,
    prompt: Result<f64, String>,
    train_ms: Result<f64, String>,
}

fn show(v: &Result<f64, String>, digits: usize) -> String {
    match v {
        Ok(v) => format!("{v:.digits$}"),
        Err(_) => "-".into(),
    }
}

fn print(name: &str, dtype: WeightDtype, r: &Row) {
    println!(
        "  {name:<6} {:<5} decode B1 {:>7}  B16 {:>7} us   prompt {:>8} tok/s   train {:>7} ms",
        format!("{dtype:?}"),
        show(&r.decode1, 1),
        show(&r.decode16, 1),
        show(&r.prompt, 0),
        show(&r.train_ms, 2),
    );
    for (part, v) in [
        ("decode B1", &r.decode1),
        ("decode B16", &r.decode16),
        ("prompt", &r.prompt),
        ("train", &r.train_ms),
    ] {
        if let Err(e) = v {
            println!("    {name} {dtype:?} {part}: {e}");
        }
    }
}

/// Decode at `batch` through a captured graph, in microseconds a step.
fn decode<B: GpuBackbone>(mut bb: B) -> Result<f64, String> {
    let batch = bb.batch();
    let input = det(batch * D_MODEL, 12);
    let mut out = vec![0f32; batch * D_MODEL];
    bb.step(&input, &mut out)?;
    bb.capture_graph()?;
    Ok(median_us(20, 200, || bb.step(&input, &mut out).unwrap()))
}

/// A prompt of `PROMPT` tokens from host rows, in tokens a second.
fn prompt<B: GpuBackbone>(mut bb: B) -> Result<f64, String> {
    let input = det(PROMPT * D_MODEL, 11);
    bb.prefill(&input, PROMPT)?;
    let us = median_us(2, 7, || {
        bb.reset().unwrap();
        bb.prefill(&input, PROMPT).unwrap();
        bb.stream().synchronize().unwrap();
    });
    Ok(PROMPT as f64 / us * 1e6)
}

/// One full training step, in milliseconds.
fn train<T: GpuTrainer>(mut tr: T) -> Result<f64, String> {
    let (b, t) = TRAIN;
    let input = det(b * t * D_MODEL, 13);
    let d_out = det(b * t * D_MODEL, 14);
    tr.step(&input, &d_out)?;
    Ok(median_us(2, 7, || {
        tr.step(&input, &d_out).unwrap();
    }) / 1e3)
}

fn session() -> TrainSessionCfg {
    TrainSessionCfg::new(D_MODEL, TRAIN.0, TRAIN.1)
}

fn mamba1(dtype: WeightDtype) -> Row {
    let cfg = MambaConfig {
        d_model: D_MODEL,
        n_layers: LAYERS,
        ..MambaConfig::default()
    };
    let w = MambaWeights::init(&cfg, D_MODEL, 7);
    let opts = GpuOptions::default().dtype(dtype);
    let bb = |b| MambaGpuBackbone::new(&w, cfg, D_MODEL, b, opts);
    Row {
        decode1: bb(1).and_then(decode),
        decode16: bb(16).and_then(decode),
        prompt: bb(1).and_then(prompt),
        train_ms: MambaGpuTrainer::new(&w, cfg, session(), opts).and_then(train),
    }
}

fn mamba2(dtype: WeightDtype) -> Row {
    let cfg = Mamba2Config {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        n_layers: LAYERS,
        ..Mamba2Config::default()
    };
    let w = Mamba2Weights::init(&cfg, D_MODEL, 7);
    let opts = GpuOptions::default().dtype(dtype);
    let bb = |b| Mamba2GpuBackbone::new(&w, cfg, D_MODEL, b, opts);
    Row {
        decode1: bb(1).and_then(decode),
        decode16: bb(16).and_then(decode),
        prompt: bb(1).and_then(prompt),
        train_ms: Mamba2GpuTrainer::new(&w, cfg, session(), opts).and_then(train),
    }
}

fn siso(dtype: WeightDtype) -> Row {
    let cfg = Mamba3Config {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        n_layers: LAYERS,
        ..Mamba3Config::default()
    };
    let w = Mamba3Weights::init(&cfg, D_MODEL, 7);
    let opts = GpuOptions::default().dtype(dtype);
    let bb = |b| Mamba3GpuBackbone::new(&w, cfg, D_MODEL, b, opts);
    Row {
        decode1: bb(1).and_then(decode),
        decode16: bb(16).and_then(decode),
        prompt: bb(1).and_then(prompt),
        train_ms: Mamba3GpuTrainer::new(&w, cfg, session(), opts).and_then(train),
    }
}

fn mimo(dtype: WeightDtype) -> Row {
    let cfg = Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        mimo_rank: 4,
        n_layers: LAYERS,
        ..Mamba3MimoConfig::default()
    };
    let w = Mamba3MimoWeights::init(&cfg, D_MODEL, 7);
    let opts = GpuOptions::default().dtype(dtype);
    let bb = |b| Mamba3MimoGpuBackbone::new(&w, cfg, D_MODEL, b, opts);
    Row {
        decode1: bb(1).and_then(decode),
        decode16: bb(16).and_then(decode),
        prompt: bb(1).and_then(prompt),
        train_ms: Mamba3MimoGpuTrainer::new(&w, cfg, session(), opts).and_then(train),
    }
}

fn main() {
    println!(
        "--- four families, {LAYERS} layers of d_model {D_MODEL}; train B={} T={}; prompt T={PROMPT} ---",
        TRAIN.0, TRAIN.1
    );
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        print("M1", dtype, &mamba1(dtype));
        print("M2", dtype, &mamba2(dtype));
        print("SISO", dtype, &siso(dtype));
        print("MIMO", dtype, &mimo(dtype));
    }
}
