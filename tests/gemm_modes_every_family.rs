//! The cuBLAS GEMM modes in every family.
//!
//! Each cuBLAS mode serves a prompt and the steps after it within the
//! tolerance its arithmetic allows of the deterministic mode, repeats its
//! own bits from run to run on one card, replays a captured graph with the
//! eager bits, and trains every family. The deterministic mode is pinned
//! by its own recorders; this file checks what the cuBLAS modes promise.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyGpuBackbone, BackwardOpts, GemmMode, GpuBackbone, GpuOptions, GpuTrainer, Mamba2Config,
    Mamba2GpuBackbone, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3GpuBackbone,
    Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuBackbone, MambaGpuTrainer, MambaWeights,
    TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const D_MODEL: usize = 32;
/// Wider than the model, so every family runs its input projection.
const INPUT_DIM: usize = 24;
const PROMPT: usize = 100;
const SEQ_LEN: usize = 16;

const CUBLAS: [GemmMode; 2] = [GemmMode::CublasPedantic, GemmMode::CublasFast];
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        block_norm: mamba_rs::BlockNorm::Rms,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
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

/// One backbone of every family, built from the same weights whatever
/// the options say.
fn every_family(opts: GpuOptions) -> Vec<AnyGpuBackbone> {
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

/// One trainer of every family, built from the same weights whatever the
/// options say.
fn every_trainer(opts: GpuOptions) -> Vec<Box<dyn GpuTrainer>> {
    let s = TrainSessionCfg {
        lr: 3e-3,
        ..TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN)
    };
    vec![
        Box::new(
            MambaGpuTrainer::new(
                &MambaWeights::init(&m1_cfg(), INPUT_DIM, 51),
                m1_cfg(),
                s,
                opts,
            )
            .unwrap(),
        ),
        Box::new(
            Mamba2GpuTrainer::new(
                &Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 52),
                m2_cfg(),
                s,
                opts,
            )
            .unwrap(),
        ),
        Box::new(
            Mamba3GpuTrainer::new(
                &Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 53),
                siso_cfg(),
                s,
                opts,
            )
            .unwrap(),
        ),
        Box::new(
            Mamba3MimoGpuTrainer::new(
                &Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 54),
                mimo_cfg(),
                s,
                opts,
            )
            .unwrap(),
        ),
    ]
}

const NAMES: [&str; 4] = ["Mamba-1", "Mamba-2", "Mamba-3 SISO", "Mamba-3 MIMO"];

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn max_rel_diff(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// How far a cuBLAS mode may land from the deterministic mode. Pedantic
/// f32 compute on f32 storage differs only in the order of its sums; TF32
/// products, or half storage rounded after a different order of sums,
/// move the outputs by a few units in the third digit.
fn tolerance(mode: GemmMode, dtype: WeightDtype) -> f32 {
    match (mode, dtype) {
        (GemmMode::CublasPedantic, WeightDtype::F32) => 1e-3,
        _ => 5e-2,
    }
}

/// The rows after a prompt and after each of the steps that follow it,
/// one after another.
fn serve(bb: &mut AnyGpuBackbone, prompt: &[f32], steps: &[Vec<f32>]) -> Vec<f32> {
    let rows = BATCH * bb.d_model();
    let mut out = vec![0.0f32; rows * (1 + steps.len())];
    bb.reset().unwrap();
    bb.prefill(prompt, PROMPT).unwrap();
    bb.download_temporal(&mut out[..rows]).unwrap();
    for (i, input) in steps.iter().enumerate() {
        bb.step(input, &mut out[(i + 1) * rows..(i + 2) * rows])
            .unwrap();
    }
    out
}

fn prompt_and_steps() -> (Vec<f32>, Vec<Vec<f32>>) {
    let prompt = values(BATCH * PROMPT * INPUT_DIM, 4);
    let steps = (0..4).map(|i| values(BATCH * INPUT_DIM, 20 + i)).collect();
    (prompt, steps)
}

#[test]
fn cublas_modes_serve_like_the_deterministic_mode_in_every_family() {
    let (prompt, steps) = prompt_and_steps();
    for dtype in DTYPES {
        let det = GpuOptions::default()
            .dtype(dtype)
            .mode(GemmMode::Deterministic);
        let want: Vec<Vec<f32>> = every_family(det)
            .iter_mut()
            .map(|bb| serve(bb, &prompt, &steps))
            .collect();
        for mode in CUBLAS {
            let opts = GpuOptions::default().dtype(dtype).mode(mode);
            for ((bb, want), name) in every_family(opts).iter_mut().zip(&want).zip(NAMES) {
                let got = serve(bb, &prompt, &steps);
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{name} {dtype:?} {mode:?}"
                );
                let rel = max_rel_diff(&got, want);
                eprintln!("{name} {dtype:?} {mode:?}: max rel {rel:.2e}");
                assert!(
                    rel <= tolerance(mode, dtype),
                    "{name} {dtype:?} {mode:?}: max rel {rel:.2e} from the deterministic mode"
                );
            }
        }
    }
}

#[test]
fn cublas_modes_repeat_their_bits_run_to_run_in_every_family() {
    let (prompt, steps) = prompt_and_steps();
    for dtype in DTYPES {
        for mode in CUBLAS {
            let opts = GpuOptions::default().dtype(dtype).mode(mode);
            let first: Vec<Vec<u32>> = every_family(opts)
                .iter_mut()
                .map(|bb| bits(&serve(bb, &prompt, &steps)))
                .collect();
            let second: Vec<Vec<u32>> = every_family(opts)
                .iter_mut()
                .map(|bb| bits(&serve(bb, &prompt, &steps)))
                .collect();
            for ((a, b), name) in first.iter().zip(&second).zip(NAMES) {
                assert_eq!(a, b, "{name} {dtype:?} {mode:?}: a second run moved bits");
            }
        }
    }
}

#[test]
fn cublas_modes_replay_a_captured_graph_bit_for_bit_in_every_family() {
    let (prompt, steps) = prompt_and_steps();
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        for mode in CUBLAS {
            let opts = GpuOptions::default().dtype(dtype).mode(mode);
            for (mut bb, name) in every_family(opts).into_iter().zip(NAMES) {
                let eager = bits(&serve(&mut bb, &prompt, &steps));
                bb.capture_graph().unwrap();
                assert!(bb.has_graph(), "{name} {dtype:?} {mode:?}: no graph");
                let replayed = bits(&serve(&mut bb, &prompt, &steps));
                assert_eq!(replayed, eager, "{name} {dtype:?} {mode:?}: replay");
            }
        }
    }
}

#[test]
fn cublas_modes_train_every_family() {
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 7);
    let target = values(BATCH * SEQ_LEN * D_MODEL, 8);
    let n = (BATCH * SEQ_LEN * D_MODEL) as f32;
    // Six steps of a mean squared error; the losses and the first output.
    let train = |t: &mut dyn GpuTrainer| -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        let mut first = Vec::new();
        let mut losses = Vec::new();
        for i in 0..6 {
            t.reset_state().unwrap();
            t.forward(&input, &mut out).unwrap();
            if i == 0 {
                first = out.clone();
            }
            losses.push(
                out.iter()
                    .zip(&target)
                    .map(|(o, y)| (o - y) * (o - y))
                    .sum::<f32>()
                    / n,
            );
            let d: Vec<f32> = out
                .iter()
                .zip(&target)
                .map(|(o, y)| 2.0 * (o - y) / n)
                .collect();
            t.backward_step(&d, BackwardOpts::default()).unwrap();
        }
        (losses, first)
    };
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let det = GpuOptions::default()
            .dtype(dtype)
            .mode(GemmMode::Deterministic);
        let want: Vec<Vec<f32>> = every_trainer(det)
            .iter_mut()
            .map(|t| train(t.as_mut()).1)
            .collect();
        for mode in CUBLAS {
            let opts = GpuOptions::default().dtype(dtype).mode(mode);
            for ((mut t, want), name) in every_trainer(opts).into_iter().zip(&want).zip(NAMES) {
                let (losses, first) = train(t.as_mut());
                let rel = max_rel_diff(&first, want);
                eprintln!(
                    "{name} {dtype:?} {mode:?}: loss {:.5} -> {:.5}, first forward max rel {rel:.2e}",
                    losses[0], losses[5]
                );
                assert!(
                    losses.iter().all(|l| l.is_finite()) && losses[5] < losses[0],
                    "{name} {dtype:?} {mode:?}: {losses:?}"
                );
                assert!(
                    rel <= tolerance(mode, dtype),
                    "{name} {dtype:?} {mode:?}: first forward max rel {rel:.2e}"
                );
            }
        }
    }
}
