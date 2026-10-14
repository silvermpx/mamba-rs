//! A training window cut in two carries the recurrence across the cut: for
//! every family, on the CPU and on the GPU in each scan mode, two forwards
//! of half a window give the second half of one whole window's output and
//! its final state. Truncated backpropagation through time rests on it: a
//! trainer that dropped the state at a window's start would train on
//! sequences that restart every window.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyCpuTrainer, AnyGpuTrainer, CpuTrainer, GpuOptions, GpuTrainer, Mamba2Config,
    Mamba2CpuTrainer, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3CpuTrainer,
    Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaCpuTrainer, MambaGpuTrainer, MambaWeights,
    TrainSessionCfg, TrainerState, WeightDtype,
};

const BATCH: usize = 2;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;

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

/// The largest elementwise difference; a NaN on either side is infinite.
fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    })
}

/// Rows `range` of every sequence of a `[BATCH, seq_len, width]` tensor.
fn rows(t: &[f32], seq_len: usize, width: usize, range: std::ops::Range<usize>) -> Vec<f32> {
    t.chunks_exact(seq_len * width)
        .flat_map(|seq| seq[range.start * width..range.end * width].iter().copied())
        .collect()
}

/// One trainer of a family on one device, for a window of `seq_len`.
enum Trainer {
    Cpu(AnyCpuTrainer),
    Gpu(AnyGpuTrainer),
}

impl Trainer {
    fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        match self {
            Trainer::Cpu(t) => t.forward(input, out).unwrap(),
            Trainer::Gpu(t) => t.forward(input, out).unwrap(),
        }
    }

    fn state(&self) -> TrainerState {
        match self {
            Trainer::Cpu(t) => t.trainer_state().unwrap(),
            Trainer::Gpu(t) => t.trainer_state().unwrap(),
        }
    }
}

/// The device a trainer runs on: the CPU, or the GPU in a scan mode.
#[derive(Clone, Copy, Debug)]
enum Device {
    Cpu,
    Gpu(ScanMode),
}

/// Every family's trainer on `device` for windows of `seq_len`.
fn trainers(device: Device, seq_len: usize) -> Vec<(&'static str, Trainer)> {
    let s = TrainSessionCfg::new(INPUT_DIM, BATCH, seq_len);
    let m1 = MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        block_norm: mamba_rs::BlockNorm::Rms,
        final_norm: true,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let m2 = Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    };
    let siso = Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    };
    let mimo = Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let w1 = MambaWeights::init(&m1, INPUT_DIM, 1);
    let w2 = Mamba2Weights::init(&m2, INPUT_DIM, 2);
    let w3 = Mamba3Weights::init(&siso, INPUT_DIM, 3);
    let w4 = Mamba3MimoWeights::init(&mimo, INPUT_DIM, 4);
    match device {
        Device::Cpu => vec![
            (
                "Mamba-1",
                Trainer::Cpu(MambaCpuTrainer::new(&w1, m1, s).unwrap().into()),
            ),
            (
                "Mamba-2",
                Trainer::Cpu(Mamba2CpuTrainer::new(&w2, m2, s).unwrap().into()),
            ),
            (
                "Mamba-3 SISO",
                Trainer::Cpu(Mamba3CpuTrainer::new(&w3, siso, s).unwrap().into()),
            ),
            (
                "Mamba-3 MIMO",
                Trainer::Cpu(Mamba3MimoCpuTrainer::new(&w4, mimo, s).unwrap().into()),
            ),
        ],
        Device::Gpu(mode) => {
            let o = GpuOptions::default()
                .dtype(WeightDtype::F32)
                .scan_mode(mode);
            vec![
                (
                    "Mamba-1",
                    Trainer::Gpu(MambaGpuTrainer::new(&w1, m1, s, o).unwrap().into()),
                ),
                (
                    "Mamba-2",
                    Trainer::Gpu(Mamba2GpuTrainer::new(&w2, m2, s, o).unwrap().into()),
                ),
                (
                    "Mamba-3 SISO",
                    Trainer::Gpu(Mamba3GpuTrainer::new(&w3, siso, s, o).unwrap().into()),
                ),
                (
                    "Mamba-3 MIMO",
                    Trainer::Gpu(Mamba3MimoGpuTrainer::new(&w4, mimo, s, o).unwrap().into()),
                ),
            ]
        }
    }
}

#[test]
fn a_window_cut_in_two_carries_the_state_on_every_device() {
    let mut failures = Vec::new();
    for device in [
        Device::Cpu,
        Device::Gpu(ScanMode::Sequential),
        Device::Gpu(ScanMode::Parallel),
    ] {
        // One chunk of the chunked scan and several, each cut mid-chunk.
        for seq_len in [24usize, 160] {
            let half = seq_len / 2;
            let input = values(BATCH * seq_len * INPUT_DIM, seq_len as u32);
            let (first, second) = (
                rows(&input, seq_len, INPUT_DIM, 0..half),
                rows(&input, seq_len, INPUT_DIM, half..seq_len),
            );
            let wholes = trainers(device, seq_len);
            let halves = trainers(device, half);
            for ((family, mut whole), (_, mut cut)) in wholes.into_iter().zip(halves) {
                let mut out = vec![0.0f32; BATCH * seq_len * D_MODEL];
                whole.forward(&input, &mut out);
                let mut out_cut = vec![0.0f32; BATCH * half * D_MODEL];
                cut.forward(&first, &mut out_cut);
                cut.forward(&second, &mut out_cut);
                let tail = rows(&out, seq_len, D_MODEL, half..seq_len);
                let out_diff = worst(&tail, &out_cut);
                let mut line = format!(
                    "{family} on {device:?}, window {seq_len}: the second half differs by \
                     {out_diff:e}"
                );
                let (sw, sc) = (whole.state(), cut.state());
                let mut state_diff = 0.0f32;
                for ((name, vw), (_, vc)) in sw.recurrent.iter().zip(&sc.recurrent) {
                    let scale = vw.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                    let d = worst(vw, vc) / scale;
                    line.push_str(&format!(", {name} {d:e}"));
                    state_diff = state_diff.max(d);
                }
                eprintln!("{line}");
                if out_diff > 1e-4 || state_diff > 1e-4 {
                    failures.push(line);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the state did not cross the cut:\n{}",
        failures.join("\n")
    );
}
