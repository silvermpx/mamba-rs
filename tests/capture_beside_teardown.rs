//! A CUDA graph capture in one thread while another thread builds, runs and
//! drops models and trainers. A synchronize of the device or of its context
//! is invalid while any stream of it captures, and it ends that capture
//! whichever thread runs it. So building or dropping a model may wait on its
//! own stream only: two models in one process, one capturing its decode step
//! while the other is built or dropped, is a plain serving setup.
#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyGpuBackbone, GemmMode, GpuBackbone, GpuOptions, Mamba2Config, Mamba2GpuBackbone,
    Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3GpuBackbone, Mamba3GpuTrainer,
    Mamba3MimoConfig, Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer, Mamba3MimoWeights,
    Mamba3Weights, MambaConfig, MambaGpuBackbone, MambaGpuTrainer, MambaWeights, TrainSessionCfg,
    WeightDtype,
};

const BATCH: usize = 2;
const D_MODEL: usize = 32;
/// Wider than the model, so every family runs its input projection.
const INPUT_DIM: usize = 24;
const PROMPT: usize = 6;

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    }
}

fn siso_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    }
}

/// One backbone of every family, storing its weights in `dtype`, in GEMM
/// `mode`.
fn every_backbone(dtype: WeightDtype, mode: GemmMode) -> Vec<AnyGpuBackbone> {
    let opts = GpuOptions::default().dtype(dtype).mode(mode);
    let m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 11);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 12);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 13);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 14);
    vec![
        MambaGpuBackbone::new(&m1, m1_cfg(), INPUT_DIM, BATCH, opts)
            .unwrap()
            .into(),
        Mamba2GpuBackbone::new(&m2, m2_cfg(), INPUT_DIM, BATCH, opts)
            .unwrap()
            .into(),
        Mamba3GpuBackbone::new(&siso, siso_cfg(), INPUT_DIM, BATCH, opts)
            .unwrap()
            .into(),
        Mamba3MimoGpuBackbone::new(&mimo, mimo_cfg(), INPUT_DIM, BATCH, opts)
            .unwrap()
            .into(),
    ]
}

/// Build one trainer of every family in `dtype` and drop it.
fn every_trainer(dtype: WeightDtype) {
    let opts = GpuOptions::default().dtype(dtype);
    let session = TrainSessionCfg::new(INPUT_DIM, BATCH, PROMPT);
    let m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 21);
    drop(MambaGpuTrainer::new(&m1, m1_cfg(), session, opts).unwrap());
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 22);
    drop(Mamba2GpuTrainer::new(&m2, m2_cfg(), session, opts).unwrap());
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 23);
    drop(Mamba3GpuTrainer::new(&siso, siso_cfg(), session, opts).unwrap());
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 24);
    drop(Mamba3MimoGpuTrainer::new(&mimo, mimo_cfg(), session, opts).unwrap());
}

/// Models of every family capture their decode step again and again, in
/// the deterministic and a cuBLAS GEMM mode, while another thread builds,
/// runs and drops the backbones and trainers of every family, in f32 and
/// bf16 and both GEMM modes. Every capture holds.
#[test]
fn a_capture_holds_while_models_are_built_and_dropped_beside_it() {
    let stop = Arc::new(AtomicBool::new(false));
    let rounds = Arc::new(AtomicUsize::new(0));
    let churn = {
        let (stop, rounds) = (stop.clone(), rounds.clone());
        std::thread::spawn(move || {
            let prompt = vec![0.25f32; BATCH * PROMPT * INPUT_DIM];
            while !stop.load(Ordering::Relaxed) {
                for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
                    for mode in [GemmMode::Deterministic, GemmMode::CublasPedantic] {
                        for mut model in every_backbone(dtype, mode) {
                            model.prefill(&prompt, PROMPT).unwrap();
                        }
                    }
                    every_trainer(dtype);
                }
                rounds.fetch_add(1, Ordering::Relaxed);
            }
        })
    };

    let mut models = every_backbone(WeightDtype::F32, GemmMode::Deterministic);
    models.extend(every_backbone(WeightDtype::Bf16, GemmMode::CublasPedantic));
    let (mut captures, mut failure) = (0usize, None);
    while failure.is_none() && rounds.load(Ordering::Relaxed) < 2 && !churn.is_finished() {
        for model in &mut models {
            let captured = model.capture_graph().and_then(|()| model.clear_graph());
            if let Err(error) = captured {
                failure = Some(format!(
                    "{:?} {:?} capture {captures}: {error}",
                    model.family(),
                    model.dtype()
                ));
                break;
            }
            captures += 1;
        }
    }
    stop.store(true, Ordering::Relaxed);
    churn
        .join()
        .expect("the thread that builds and drops models");
    assert_eq!(failure, None, "a capture beside models built and dropped");
    assert!(captures > 0, "no capture ran");
}
