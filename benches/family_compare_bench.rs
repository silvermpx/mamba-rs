#![cfg(feature = "cuda")]

//! The four families side by side at one size: 4 layers of d_model 768,
//! Mamba-2, Mamba-3 SISO and MIMO with heads of 64 x 128 (MIMO rank 4),
//! Mamba-1 with d_state 16 as the released mamba-130m. Decode steps at batch
//! 1 and 16 through a captured graph, a full training step (forward,
//! backward and the AdamW update) at batch 4 by 512 tokens, and the chunked
//! prompt window where the engine takes a host window.
//!
//! Run: `cargo bench --bench family_compare_bench --features cuda`

use std::time::Instant;

use mamba_rs::config::MambaConfig;
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::gpu::trainer::MambaTrainer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba2::gpu::{Mamba2GpuInferenceEngine, Mamba2GpuTrainer};
use mamba_rs::mamba2::{Mamba2Config, Mamba2Weights};
use mamba_rs::mamba3_mimo::gpu::{Mamba3MimoGpuEngine, Mamba3MimoGpuTrainer};
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::weights::MambaWeights;

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
/// training step (ms) and the prompt window (tokens/s, when measured).
struct Row {
    decode1: f64,
    decode16: f64,
    train_ms: Option<f64>,
    prompt: Option<f64>,
}

fn print(name: &str, dtype: WeightDtype, r: &Row) {
    let prompt = r
        .prompt
        .map_or_else(|| "-".to_string(), |v| format!("{v:.0}"));
    let train = r
        .train_ms
        .map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    println!(
        "  {name:<6} {:<5} decode B1 {:>7.1}  B16 {:>7.1} us   train {train:>7} ms   prompt {prompt:>8} tok/s",
        format!("{dtype:?}"),
        r.decode1,
        r.decode16,
    );
}

fn mamba1(device_ordinal: usize, dtype: WeightDtype) -> Row {
    let cfg = MambaConfig {
        d_model: D_MODEL,
        n_layers: LAYERS,
        ..MambaConfig::default()
    };
    let w = MambaWeights::init(&cfg, D_MODEL, 7);
    let mut decode = [0f64; 2];
    for (slot, b) in [1usize, 16].into_iter().enumerate() {
        let mut bb =
            GpuMambaBackbone::new_with_dtype(device_ordinal, &w, cfg, D_MODEL, b, dtype).unwrap();
        let input = det(b * D_MODEL, 12);
        let mut out = vec![0f32; b * D_MODEL];
        bb.step(&input, &mut out).unwrap();
        bb.capture_graph().unwrap();
        decode[slot] = median_us(20, 200, || bb.step(&input, &mut out).unwrap());
    }
    let (b, t) = TRAIN;
    let mut tr =
        MambaTrainer::new_with_dtype(device_ordinal, &w, cfg, D_MODEL, b, t, dtype).unwrap();
    let input = det(b * t * D_MODEL, 13);
    let d_out = det(b * t * D_MODEL, 14);
    let train = median_us(2, 7, || {
        tr.step(&input, &d_out).unwrap();
    });
    Row {
        decode1: decode[0],
        decode16: decode[1],
        train_ms: Some(train / 1e3),
        prompt: None,
    }
}

fn siso(device_ordinal: usize, dtype: WeightDtype) -> Row {
    let cfg = Mamba3Config {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        n_layers: LAYERS,
        ..Mamba3Config::default()
    };
    let w = Mamba3Weights::init(&cfg, D_MODEL, 7);
    let mut decode = [0f64; 2];
    for (slot, b) in [1usize, 16].into_iter().enumerate() {
        let mut bb =
            GpuMamba3Backbone::new_with_dtype(device_ordinal, &w, cfg, D_MODEL, b, dtype).unwrap();
        let input = det(b * D_MODEL, 12);
        let mut out = vec![0f32; b * D_MODEL];
        bb.step(&input, &mut out).unwrap();
        bb.capture_graph().unwrap();
        decode[slot] = median_us(20, 200, || bb.step(&input, &mut out).unwrap());
    }
    let prompt = {
        let mut bb =
            GpuMamba3Backbone::new_with_dtype(device_ordinal, &w, cfg, D_MODEL, 1, dtype).unwrap();
        bb.supports_prefill().then(|| {
            let mut pf = bb.alloc_prefill(PROMPT).unwrap();
            let host = det(PROMPT * D_MODEL, 11);
            let input = mamba_rs::gpu::buffers::GpuBuffer::from_cpu(bb.stream(), &host).unwrap();
            let us = median_us(2, 7, || {
                bb.prefill_sequence(&mut pf, &input, PROMPT, false).unwrap();
                bb.stream().synchronize().unwrap();
            });
            PROMPT as f64 / us * 1e6
        })
    };
    // The GPU trainer serves warp-tier heads only; at this head width it
    // refuses, and the row says so instead of a time.
    let (b, t) = TRAIN;
    let train_ms =
        match Mamba3Trainer::new_with_dtype(device_ordinal, &w, cfg, D_MODEL, b, t, dtype) {
            Ok(mut tr) => {
                let input = det(b * t * D_MODEL, 13);
                let d_out = det(b * t * D_MODEL, 14);
                Some(
                    median_us(2, 7, || {
                        tr.step(&input, &d_out).unwrap();
                    }) / 1e3,
                )
            }
            Err(e) => {
                println!("  SISO   {dtype:?} training refused: {e}");
                None
            }
        };
    Row {
        decode1: decode[0],
        decode16: decode[1],
        train_ms,
        prompt,
    }
}

fn mamba2(device: &GpuDevice, dtype: WeightDtype) -> Row {
    let cfg = Mamba2Config {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        n_layers: LAYERS,
        ..Mamba2Config::default()
    };
    let w = Mamba2Weights::init(&cfg, D_MODEL, 7);
    let mut decode = [0f64; 2];
    for (slot, b) in [1usize, 16].into_iter().enumerate() {
        let mut eng =
            Mamba2GpuInferenceEngine::new_with_dtype(device, &w, (cfg, D_MODEL, b), dtype).unwrap();
        let (mut st, mut sc) = (eng.alloc_state().unwrap(), eng.alloc_scratch().unwrap());
        let input = det(b * D_MODEL, 12);
        let mut out = vec![0f32; b * D_MODEL];
        eng.step(&input, &mut out, &mut st, &mut sc).unwrap();
        unsafe { eng.capture_graph(&mut st, &mut sc) }.unwrap();
        decode[slot] = median_us(20, 200, || {
            eng.step(&input, &mut out, &mut st, &mut sc).unwrap()
        });
    }
    let prompt = {
        let eng =
            Mamba2GpuInferenceEngine::new_with_dtype(device, &w, (cfg, D_MODEL, 1), dtype).unwrap();
        let mut st = eng.alloc_state().unwrap();
        let mut pf = eng.alloc_prefill(PROMPT).unwrap();
        let input = det(PROMPT * D_MODEL, 11);
        let mut rows = vec![0f32; PROMPT * D_MODEL];
        let us = median_us(2, 7, || {
            st.reset(eng.ctx()).unwrap();
            eng.prefill(&input, &mut rows, &mut st, &mut pf).unwrap();
        });
        PROMPT as f64 / us * 1e6
    };
    let (b, t) = TRAIN;
    let mut tr = Mamba2GpuTrainer::new_with_dtype(device, &w, (cfg, D_MODEL, b, t), dtype).unwrap();
    let input = det(b * t * D_MODEL, 13);
    let d_out = det(b * t * D_MODEL, 14);
    let mut rows = vec![0f32; b * t * D_MODEL];
    let train = median_us(2, 7, || {
        tr.reset_state().unwrap();
        tr.forward(&input, &mut rows).unwrap();
        tr.backward(&d_out, None).unwrap();
        tr.apply_adamw().unwrap();
    });
    Row {
        decode1: decode[0],
        decode16: decode[1],
        train_ms: Some(train / 1e3),
        prompt: Some(prompt),
    }
}

fn mimo(device: &GpuDevice, dtype: WeightDtype) -> Row {
    let cfg = Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 128,
        headdim: 64,
        mimo_rank: 4,
        n_layers: LAYERS,
        ..Mamba3MimoConfig::default()
    };
    let w = Mamba3MimoWeights::init(&cfg, D_MODEL, 7);
    let mut decode = [0f64; 2];
    for (slot, b) in [1usize, 16].into_iter().enumerate() {
        let mut eng = Mamba3MimoGpuEngine::new(device, &w, (cfg, D_MODEL, b), dtype).unwrap();
        let mut st = eng.alloc_state().unwrap();
        let mut win = eng.alloc_window(1).unwrap();
        let input = det(b * D_MODEL, 12);
        let mut out = vec![0f32; b * D_MODEL];
        eng.run(&input, &mut out, &mut st, &mut win).unwrap();
        unsafe { eng.capture_graph(&mut st, &mut win) }.unwrap();
        decode[slot] = median_us(20, 200, || {
            eng.run(&input, &mut out, &mut st, &mut win).unwrap()
        });
    }
    let prompt = {
        let eng = Mamba3MimoGpuEngine::new(device, &w, (cfg, D_MODEL, 1), dtype).unwrap();
        let mut win = eng.alloc_window(PROMPT).unwrap();
        let input = det(PROMPT * D_MODEL, 11);
        let mut rows = vec![0f32; PROMPT * D_MODEL];
        let us = median_us(2, 7, || {
            let mut st = eng.alloc_state().unwrap();
            eng.run(&input, &mut rows, &mut st, &mut win).unwrap();
        });
        PROMPT as f64 / us * 1e6
    };
    let (b, t) = TRAIN;
    let mut tr = Mamba3MimoGpuTrainer::new(device, &w, (cfg, D_MODEL, b, t), (dtype, 16)).unwrap();
    let input = det(b * t * D_MODEL, 13);
    let d_out = det(b * t * D_MODEL, 14);
    let mut rows = vec![0f32; b * t * D_MODEL];
    let train = median_us(2, 7, || {
        tr.reset_state().unwrap();
        tr.forward(&input, &mut rows).unwrap();
        tr.backward(&d_out, None).unwrap();
        tr.apply_adamw().unwrap();
    });
    Row {
        decode1: decode[0],
        decode16: decode[1],
        train_ms: Some(train / 1e3),
        prompt: Some(prompt),
    }
}

fn main() {
    let device = GpuDevice::new(0).unwrap();
    println!(
        "--- four families, {LAYERS} layers of d_model {D_MODEL}; train B={} T={}; prompt T={PROMPT} ---",
        TRAIN.0, TRAIN.1
    );
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        print("M1", dtype, &mamba1(0, dtype));
        print("M2", dtype, &mamba2(&device, dtype));
        print("SISO", dtype, &siso(0, dtype));
        print("MIMO", dtype, &mimo(&device, dtype));
    }
}
