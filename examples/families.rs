//! The four families through one API: a GPU backbone of each family
//! serves a prompt and a few decode steps, and a GPU trainer of each family
//! fits a fixed target for a few steps. Requires the `cuda` feature and an
//! NVIDIA GPU.
//!
//! ```bash
//! cargo run --release --example families --features cuda
//! ```
//!
//! Every family is built from its own configuration and weights, then
//! driven only through `GpuBackbone` and `GpuTrainer`, so the same code
//! runs Mamba, Mamba-2, Mamba-3 SISO and Mamba-3 MIMO.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs the cuda feature: --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), String> {
    use mamba_rs::{
        AnyGpuBackbone, BackwardOpts, GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuBackbone,
        Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3GpuBackbone, Mamba3GpuTrainer,
        Mamba3MimoConfig, Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer, Mamba3MimoWeights,
        Mamba3Weights, MambaConfig, MambaGpuBackbone, MambaGpuTrainer, MambaWeights,
        TrainSessionCfg,
    };
    use std::time::Instant;

    const D_MODEL: usize = 256;
    const INPUT_DIM: usize = 64;
    const BATCH: usize = 4;
    const PROMPT: usize = 512;
    const SEQ_LEN: usize = 128;

    let m1 = MambaConfig {
        d_model: D_MODEL,
        n_layers: 4,
        ..MambaConfig::default()
    };
    let m2 = Mamba2Config {
        d_model: D_MODEL,
        headdim: 64,
        d_state: 64,
        n_layers: 4,
        ..Mamba2Config::default()
    };
    let siso = Mamba3Config {
        d_model: D_MODEL,
        headdim: 64,
        d_state: 64,
        n_layers: 4,
        ..Mamba3Config::default()
    };
    let mimo = Mamba3MimoConfig {
        d_model: D_MODEL,
        headdim: 64,
        d_state: 64,
        n_layers: 4,
        ..Mamba3MimoConfig::default()
    };
    let opts = GpuOptions::default();

    // Serving: one backbone of each family, behind one type.
    let backbones: Vec<(&str, AnyGpuBackbone)> = vec![
        (
            "Mamba",
            MambaGpuBackbone::new(
                &MambaWeights::init(&m1, INPUT_DIM, 1),
                m1,
                INPUT_DIM,
                BATCH,
                opts,
            )?
            .into(),
        ),
        (
            "Mamba-2",
            Mamba2GpuBackbone::new(
                &Mamba2Weights::init(&m2, INPUT_DIM, 2),
                m2,
                INPUT_DIM,
                BATCH,
                opts,
            )?
            .into(),
        ),
        (
            "Mamba-3 SISO",
            Mamba3GpuBackbone::new(
                &Mamba3Weights::init(&siso, INPUT_DIM, 3),
                siso,
                INPUT_DIM,
                BATCH,
                opts,
            )?
            .into(),
        ),
        (
            "Mamba-3 MIMO",
            Mamba3MimoGpuBackbone::new(
                &Mamba3MimoWeights::init(&mimo, INPUT_DIM, 4),
                mimo,
                INPUT_DIM,
                BATCH,
                opts,
            )?
            .into(),
        ),
    ];
    let prompt = vec![0.1f32; BATCH * PROMPT * INPUT_DIM];
    let token = vec![0.1f32; BATCH * INPUT_DIM];
    println!("serving, batch {BATCH}, a prompt of {PROMPT} tokens then 64 steps:");
    for (name, mut bb) in backbones {
        serve(name, &mut bb, &prompt, &token)?;
    }

    // Training: one trainer of each family, behind one trait.
    let session = TrainSessionCfg {
        lr: 1e-3,
        ..TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN)
    };
    let mut trainers: Vec<(&str, Box<dyn GpuTrainer>)> = vec![
        (
            "Mamba",
            Box::new(MambaGpuTrainer::new(
                &MambaWeights::init(&m1, INPUT_DIM, 1),
                m1,
                session,
                opts,
            )?),
        ),
        (
            "Mamba-2",
            Box::new(Mamba2GpuTrainer::new(
                &Mamba2Weights::init(&m2, INPUT_DIM, 2),
                m2,
                session,
                opts,
            )?),
        ),
        (
            "Mamba-3 SISO",
            Box::new(Mamba3GpuTrainer::new(
                &Mamba3Weights::init(&siso, INPUT_DIM, 3),
                siso,
                session,
                opts,
            )?),
        ),
        (
            "Mamba-3 MIMO",
            Box::new(Mamba3MimoGpuTrainer::new(
                &Mamba3MimoWeights::init(&mimo, INPUT_DIM, 4),
                mimo,
                session,
                opts,
            )?),
        ),
    ];
    let input: Vec<f32> = (0..BATCH * SEQ_LEN * INPUT_DIM)
        .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
        .collect();
    let target: Vec<f32> = (0..BATCH * SEQ_LEN * D_MODEL)
        .map(|i| ((i % 89) as f32 / 89.0) - 0.5)
        .collect();
    println!("\ntraining, batch {BATCH} by {SEQ_LEN} tokens, mean squared error:");
    for (name, t) in &mut trainers {
        let mut out = vec![0.0f32; target.len()];
        let n = out.len() as f32;
        let mut first = 0.0;
        let start = Instant::now();
        for step in 0..10 {
            t.reset_state()?;
            t.forward(&input, &mut out)?;
            let loss = out
                .iter()
                .zip(&target)
                .map(|(o, y)| (o - y) * (o - y))
                .sum::<f32>()
                / n;
            let d: Vec<f32> = out
                .iter()
                .zip(&target)
                .map(|(o, y)| 2.0 * (o - y) / n)
                .collect();
            t.backward_step(&d, BackwardOpts::default().with_clip_max_norm(1.0))?;
            if step == 0 {
                first = loss;
            } else if step == 9 {
                println!(
                    "  {name:<13} loss {first:.4} -> {loss:.4}, {:.1} ms a step",
                    start.elapsed().as_secs_f64() * 1e3 / 10.0
                );
            }
        }
    }
    Ok(())
}

/// A prompt in one pass, then decode steps replayed from a captured graph.
#[cfg(feature = "cuda")]
fn serve(
    name: &str,
    bb: &mut mamba_rs::AnyGpuBackbone,
    prompt: &[f32],
    token: &[f32],
) -> Result<(), String> {
    use mamba_rs::GpuBackbone;
    use std::time::Instant;

    let seq_len = prompt.len() / (bb.batch() * bb.input_dim());
    // The longest prompt runs before the capture, so the GEMM staging the
    // graph fixes is large enough for it.
    bb.prefill(prompt, seq_len)?;
    bb.capture_graph()?;

    let start = Instant::now();
    bb.reset()?;
    bb.prefill(prompt, seq_len)?;
    bb.stream()
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut out = vec![0.0f32; bb.batch() * bb.d_model()];
    let start = Instant::now();
    for _ in 0..64 {
        bb.step(token, &mut out)?;
    }
    let step_us = start.elapsed().as_secs_f64() * 1e6 / 64.0;
    println!("  {name:<13} prompt {prefill_ms:6.2} ms, step {step_us:6.1} us");
    Ok(())
}
