#![cfg(feature = "cuda")]

//! A half-precision decode of a model with its own input projection (an
//! input wider or narrower than d_model, the RL shape) runs, stays within
//! the storage precision of the f32 decode, repeats its bits, and replays
//! them from a captured graph.

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuBackbone;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::{config::ScanMode, mamba1::MambaConfig};

const INPUT_DIM: usize = 40;
const BATCH: usize = 2;
const STEPS: usize = 12;

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

/// The outputs of `STEPS` decode steps, one input per step.
fn decode(
    d_model: usize,
    mut step: impl FnMut(&[f32], &mut [f32]) -> Result<(), String>,
) -> Vec<f32> {
    let mut all = Vec::with_capacity(STEPS * BATCH * d_model);
    let mut out = vec![0f32; BATCH * d_model];
    for s in 0..STEPS {
        step(&det(BATCH * INPUT_DIM, 100 + s as u64), &mut out).unwrap();
        all.extend_from_slice(&out);
    }
    all
}

/// The largest difference relative to the reference's largest magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The bf16 and f16 storage tolerances against f32 over the steps.
fn tolerance(dtype: WeightDtype) -> f32 {
    match dtype {
        WeightDtype::Bf16 => 0.06,
        _ => 0.01,
    }
}

fn m1_config() -> MambaConfig {
    MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

fn siso_config() -> Mamba3Config {
    Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        ..Mamba3Config::default()
    }
}

#[test]
fn half_inproj_m1_decode() {
    let cfg = m1_config();
    let w = MambaWeights::init(&cfg, INPUT_DIM, 0x4D31);
    let run = |dtype| {
        let mut m = MambaGpuBackbone::new(
            &w,
            cfg,
            INPUT_DIM,
            BATCH,
            GpuOptions::default().dtype(dtype),
        )
        .unwrap();
        decode(cfg.d_model, |x, y| m.step(x, y))
    };
    let reference = run(WeightDtype::F32);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let first = run(dtype);
        assert_eq!(bits(&first), bits(&run(dtype)), "M1 {dtype:?} repeat");
        let err = rel_err(&first, &reference);
        assert!(err < tolerance(dtype), "M1 {dtype:?} against f32: {err}");
    }
}

fn m3_siso_case(input_dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let cfg = siso_config();
    let mut w = Mamba3Weights::init(&cfg, input_dim, seed);
    if input_dim == cfg.d_model {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let run = |dtype| {
        let mut m = Mamba3GpuBackbone::new(
            &w,
            cfg,
            input_dim,
            BATCH,
            GpuOptions::default().dtype(dtype),
        )
        .unwrap();
        let mut all = Vec::with_capacity(STEPS * BATCH * cfg.d_model);
        let mut out = vec![0f32; BATCH * cfg.d_model];
        for s in 0..STEPS {
            m.step(&det(BATCH * input_dim, 100 + s as u64), &mut out)
                .unwrap();
            all.extend_from_slice(&out);
        }
        all
    };
    (
        run(WeightDtype::F32),
        run(WeightDtype::Bf16),
        run(WeightDtype::Bf16),
        run(WeightDtype::F16),
    )
}

#[test]
fn half_inproj_m3_siso_decode() {
    for (input_dim, what) in [(64, "identity"), (INPUT_DIM, "projected")] {
        let (reference, bf16, bf16_again, f16) = m3_siso_case(input_dim, 0x4D33);
        assert_eq!(bits(&bf16), bits(&bf16_again), "SISO {what} Bf16 repeat");
        let (eb, eh) = (rel_err(&bf16, &reference), rel_err(&f16, &reference));
        assert!(
            eb < tolerance(WeightDtype::Bf16) && eh < tolerance(WeightDtype::F16),
            "SISO {what} against f32: bf16 {eb}, f16 {eh}"
        );
    }
}

/// A decode graph replays the eager steps bit for bit when the model has its
/// own input projection. That projection reads its downcast input from the
/// context's staging buffer, so the capture holds only if the buffer is at
/// its graph size before the eager step whose GEMM launches the capture must
/// reproduce.
#[test]
fn half_inproj_decode_graph_replays_the_eager_steps() {
    let cfg = m1_config();
    let w = MambaWeights::init(&cfg, INPUT_DIM, 0x4D31);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let new = || {
            MambaGpuBackbone::new(
                &w,
                cfg,
                INPUT_DIM,
                BATCH,
                GpuOptions::default().dtype(dtype),
            )
            .unwrap()
        };
        let mut eager = new();
        let want = decode(cfg.d_model, |x, y| eager.step(x, y));
        let mut graphed = new();
        graphed.capture_graph().unwrap();
        let got = decode(cfg.d_model, |x, y| graphed.step(x, y));
        assert_eq!(bits(&got), bits(&want), "M1 {dtype:?} graph replay");
    }

    let cfg = siso_config();
    let w = Mamba3Weights::init(&cfg, INPUT_DIM, 0x4D33);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let new = || {
            Mamba3GpuBackbone::new(
                &w,
                cfg,
                INPUT_DIM,
                BATCH,
                GpuOptions::default().dtype(dtype),
            )
            .unwrap()
        };
        let mut eager = new();
        let want = decode(cfg.d_model, |x, y| eager.step(x, y));
        let mut graphed = new();
        graphed.capture_graph().unwrap();
        let got = decode(cfg.d_model, |x, y| graphed.step(x, y));
        assert_eq!(bits(&got), bits(&want), "SISO {dtype:?} graph replay");
    }
}
