#![cfg(feature = "cuda")]

//! Bit digests of every Mamba-2 and MIMO GPU output, printed for an A/B
//! between two builds: a kernel rewrite that claims to keep the bits must
//! print the same lines before and after.
//!
//! Each line covers one shape and dtype, both scans: the chunked or
//! sequential prompt window, decode steps continuing it, a second window
//! continuing those, then a training step's forward rows, input gradient
//! and weight gradients (before the optimizer, whose first step is about
//! the gradient's sign and would hide a one-ulp change), and the rows after
//! the AdamW step.
//!
//! Run: `cargo test --release --features cuda --test m2_mimo_bits_digest -- --ignored --nocapture`

#[path = "common/digest.rs"]
mod digest;

use digest::{Digest, fnv1a_f32};
use mamba_rs::config::ScanMode;
use mamba_rs::mamba_ssm::gpu::GpuOptions;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::TrainSessionCfg;
use mamba_rs::mamba2::gpu::{Mamba2GpuEngine, Mamba2GpuTrainer};
use mamba_rs::mamba2::{Mamba2Config, Mamba2Weights};
use mamba_rs::mamba3_mimo::gpu::{Mamba3MimoGpuEngine, Mamba3MimoGpuTrainer};
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};

const INPUT_DIM: usize = 48;
const BATCH: usize = 2;
const SEQ: usize = 150;

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

fn dtypes() -> [WeightDtype; 4] {
    [
        WeightDtype::F32,
        WeightDtype::Tf32,
        WeightDtype::Bf16,
        WeightDtype::F16,
    ]
}

fn modes() -> [(ScanMode, &'static str); 2] {
    [
        (ScanMode::Sequential, "seq"),
        (ScanMode::Parallel, "chunked"),
    ]
}

/// The weight gradients' bits: a weight struct's Debug form prints every
/// f32 as the shortest decimal that reads back to the same bits.
fn absorb_debug(d: &mut Digest, value: &impl std::fmt::Debug) {
    d.absorb_bytes(format!("{value:?}").as_bytes());
}

#[test]
#[ignore = "digest printer for an A/B between builds"]
fn print_m2_bits_digest() {
    let device = GpuDevice::new(0).unwrap();
    for (hd, ds) in [(16usize, 16usize), (64, 128), (24, 20), (8, 64)] {
        for dtype in dtypes() {
            let base = Mamba2Config {
                d_model: 96,
                d_state: ds,
                headdim: hd,
                n_layers: 2,
                ..Mamba2Config::default()
            };
            let w = Mamba2Weights::init(&base, INPUT_DIM, 5 + hd as u64);
            let mut line = format!("m2 hd {hd:>2} ds {ds:>3} {dtype:?}:");
            for (mode, name) in modes() {
                let cfg = Mamba2Config {
                    scan_mode: mode,
                    ..base
                };
                let eng = Mamba2GpuEngine::new(&device, &w, cfg, INPUT_DIM, BATCH, dtype).unwrap();
                let mut st = eng.alloc_state().unwrap();
                let mut sc = eng.alloc_scratch().unwrap();
                let mut pf = eng.alloc_prefill(SEQ).unwrap();
                let mut d = Digest::new();
                let mut rows = vec![0f32; BATCH * SEQ * cfg.d_model];
                eng.prefill(
                    &det(BATCH * SEQ * INPUT_DIM, 1),
                    &mut rows,
                    &mut st,
                    &mut pf,
                )
                .unwrap();
                d.absorb_f32(&rows);
                let mut out = vec![0f32; BATCH * cfg.d_model];
                for s in 0..6 {
                    eng.step(&det(BATCH * INPUT_DIM, 10 + s), &mut out, &mut st, &mut sc)
                        .unwrap();
                    d.absorb_bytes(&fnv1a_f32(&out).to_le_bytes());
                }
                eng.prefill(
                    &det(BATCH * SEQ * INPUT_DIM, 2),
                    &mut rows,
                    &mut st,
                    &mut pf,
                )
                .unwrap();
                d.absorb_f32(&rows);
                let mut tr = Mamba2GpuTrainer::new(
                    &w,
                    cfg,
                    TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ),
                    GpuOptions::default().dtype(dtype),
                )
                .unwrap();
                let mut dx = vec![0f32; BATCH * SEQ * INPUT_DIM];
                tr.forward(&det(BATCH * SEQ * INPUT_DIM, 3), &mut rows)
                    .unwrap();
                d.absorb_f32(&rows);
                tr.backward(&det(BATCH * SEQ * cfg.d_model, 4), Some(&mut dx))
                    .unwrap();
                d.absorb_f32(&dx);
                absorb_debug(&mut d, &tr.grads().unwrap());
                tr.apply_adamw().unwrap();
                tr.reset_state().unwrap();
                tr.forward(&det(BATCH * SEQ * INPUT_DIM, 3), &mut rows)
                    .unwrap();
                d.absorb_f32(&rows);
                line += &format!("  {name} {:016x}", d.finish());
            }
            println!("{line}");
        }
    }
}

#[test]
#[ignore = "digest printer for an A/B between builds"]
fn print_mimo_bits_digest() {
    let device = GpuDevice::new(0).unwrap();
    for rank in [1usize, 2, 3, 4, 8, 16] {
        for dtype in dtypes() {
            let base = Mamba3MimoConfig {
                d_model: 64,
                d_state: 32,
                headdim: 16,
                mimo_rank: rank,
                n_layers: 2,
                ..Mamba3MimoConfig::default()
            };
            let w = Mamba3MimoWeights::init(&base, INPUT_DIM, 5 + rank as u64);
            let mut line = format!("mimo rank {rank:>2} {dtype:?}:");
            for (mode, name) in modes() {
                let cfg = Mamba3MimoConfig {
                    scan_mode: mode,
                    ..base
                };
                let eng =
                    Mamba3MimoGpuEngine::new(&device, &w, cfg, INPUT_DIM, BATCH, dtype).unwrap();
                let mut st = eng.alloc_state().unwrap();
                let mut win = eng.alloc_window(SEQ).unwrap();
                let mut one = eng.alloc_window(1).unwrap();
                let mut d = Digest::new();
                let mut rows = vec![0f32; BATCH * SEQ * cfg.d_model];
                eng.run(
                    &det(BATCH * SEQ * INPUT_DIM, 1),
                    &mut rows,
                    &mut st,
                    &mut win,
                )
                .unwrap();
                d.absorb_f32(&rows);
                let mut out = vec![0f32; BATCH * cfg.d_model];
                for s in 0..6 {
                    eng.run(&det(BATCH * INPUT_DIM, 10 + s), &mut out, &mut st, &mut one)
                        .unwrap();
                    d.absorb_bytes(&fnv1a_f32(&out).to_le_bytes());
                }
                eng.run(
                    &det(BATCH * SEQ * INPUT_DIM, 2),
                    &mut rows,
                    &mut st,
                    &mut win,
                )
                .unwrap();
                d.absorb_f32(&rows);
                let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                    &w,
                    cfg,
                    TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ),
                    GpuOptions::default().dtype(dtype),
                    16,
                )
                .unwrap();
                let mut dx = vec![0f32; BATCH * SEQ * INPUT_DIM];
                tr.forward(&det(BATCH * SEQ * INPUT_DIM, 3), &mut rows)
                    .unwrap();
                d.absorb_f32(&rows);
                tr.backward(&det(BATCH * SEQ * cfg.d_model, 4), Some(&mut dx))
                    .unwrap();
                d.absorb_f32(&dx);
                absorb_debug(&mut d, &tr.grads().unwrap());
                tr.apply_adamw().unwrap();
                tr.reset_state().unwrap();
                tr.forward(&det(BATCH * SEQ * INPUT_DIM, 3), &mut rows)
                    .unwrap();
                d.absorb_f32(&rows);
                line += &format!("  {name} {:016x}", d.finish());
            }
            println!("{line}");
        }
    }
}
