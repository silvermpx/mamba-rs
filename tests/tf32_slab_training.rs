//! The trainers that keep their weights in one flat slab, Mamba-2 and
//! Mamba-3 MIMO, train in TF32 at a d_model of 768. There their projection
//! matrices start inside a row of the slab allocation's grid, and the
//! tensor-map GEMM routes (SM90, SM100, SM120) describe such a matrix from
//! its own first element. The TF32 weight and input gradients agree with
//! the f32 trainer's to the rounding of TF32 products and repeat their bits
//! from a fresh trainer.
//!
//! Run on a GPU box:
//!   cargo test --release --features cuda --test tf32_slab_training -- --test-threads=1
#![cfg(feature = "cuda")]

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::WeightDtype;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba2::gpu::Mamba2GpuTrainer;
use mamba_rs::mamba2::{Mamba2Config, Mamba2Weights};
use mamba_rs::mamba3_mimo::gpu::Mamba3MimoGpuTrainer;
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};

const INPUT_DIM: usize = 768;
const BATCH: usize = 2;
const SEQ: usize = 128;

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

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every layer's projection gradients, then the input gradient.
type Run = Vec<(String, Vec<f32>)>;

fn mamba2(dtype: WeightDtype) -> Run {
    let cfg = Mamba2Config {
        d_model: 768,
        d_state: 128,
        headdim: 64,
        n_layers: 2,
        ..Mamba2Config::default()
    };
    cfg.validate().unwrap();
    let w = Mamba2Weights::init(&cfg, INPUT_DIM, 5);
    let mut tr = Mamba2GpuTrainer::new(
        &w,
        cfg,
        TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ),
        GpuOptions::default().dtype(dtype),
    )
    .unwrap();
    let mut rows = vec![0f32; BATCH * SEQ * cfg.d_model];
    tr.forward(&det(BATCH * SEQ * INPUT_DIM, 7), &mut rows)
        .unwrap();
    let mut d_in = vec![0f32; BATCH * SEQ * INPUT_DIM];
    tr.backward(&det(BATCH * SEQ * cfg.d_model, 8), Some(&mut d_in))
        .unwrap();
    let g = tr.grads().unwrap();
    let mut out = Run::new();
    for (l, lw) in g.layers.into_iter().enumerate() {
        out.push((format!("layer{l}.in_proj_w"), lw.in_proj_w));
        out.push((format!("layer{l}.out_proj_w"), lw.out_proj_w));
    }
    out.push(("input".to_string(), d_in));
    out
}

fn mimo(dtype: WeightDtype) -> Run {
    let cfg = Mamba3MimoConfig {
        d_model: 768,
        d_state: 128,
        headdim: 64,
        mimo_rank: 4,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    cfg.validate().unwrap();
    let w = Mamba3MimoWeights::init(&cfg, INPUT_DIM, 5);
    let mut tr = Mamba3MimoGpuTrainer::new(
        &w,
        cfg,
        TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ),
        GpuOptions::default().dtype(dtype),
    )
    .unwrap();
    let mut rows = vec![0f32; BATCH * SEQ * cfg.d_model];
    tr.forward(&det(BATCH * SEQ * INPUT_DIM, 7), &mut rows)
        .unwrap();
    let mut d_in = vec![0f32; BATCH * SEQ * INPUT_DIM];
    tr.backward(&det(BATCH * SEQ * cfg.d_model, 8), Some(&mut d_in))
        .unwrap();
    let g = tr.grads().unwrap();
    let mut out = Run::new();
    for (l, lw) in g.layers.into_iter().enumerate() {
        out.push((format!("layer{l}.in_proj_w"), lw.in_proj_w));
        out.push((format!("layer{l}.out_proj_w"), lw.out_proj_w));
    }
    out.push(("input".to_string(), d_in));
    out
}

// TF32 keeps 11 significant bits of each product's operands where f32 keeps
// 24; these gradients measure about 5e-4 against f32, the bound holds about
// ten times that, and a matrix read from a wrong origin fails by orders of
// magnitude.
#[test]
fn tf32_trains_the_slab_trainers_at_d_model_768() {
    for (family, run) in [
        ("Mamba-2", mamba2 as fn(WeightDtype) -> Run),
        ("Mamba-3 MIMO", mimo),
    ] {
        let exact = run(WeightDtype::F32);
        let tf32 = run(WeightDtype::Tf32);
        let again = run(WeightDtype::Tf32);
        for (((name, t), (_, e)), (_, a)) in tf32.iter().zip(&exact).zip(&again) {
            assert!(
                bits(t) == bits(a),
                "{family} {name}: TF32 moved from a fresh trainer"
            );
            let err = rel_l2(t, e);
            eprintln!("{family} {name}: TF32 against f32 {err:.2e}");
            assert!(err < 6e-3, "{family} {name}: TF32 against f32 {err:.3e}");
        }
    }
}
