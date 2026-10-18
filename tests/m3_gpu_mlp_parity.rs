//! The gated MLP on the GPU, on a model that has one: the f32 engine and
//! the bf16 engine, decode and prefill, at a warp-tier and a block-tier
//! head width, against the CPU. Each arm also proves it is not vacuous:
//! the GPU error must sit well below what the MLP adds to the output, so a
//! skipped or misplaced MLP branch fails it.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::cpu::prefill::{Mamba3PrefillScratch, forward_mamba3_backbone_prefill};
use mamba_rs::mamba3_siso::gpu::inference::{Mamba3GpuEngine, Mamba3GpuEngineMixed};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
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

// f32 GPU against the CPU: the sm_120 floor the chunked prefill parity
// measured, with headroom.
const F32_TOL: f64 = 2e-3;
// bf16 storage and activations against the f32 CPU: bf16 carries 8
// mantissa bits (relative step 2^-8 = 3.9e-3) and the error compounds
// over the layers' GEMMs and residual adds.
const BF16_TOL: f64 = 5e-2;
// Every arm's error must stay under a fraction of the MLP's own
// contribution to the output: a missing or misplaced branch gives an error
// of about the whole contribution. f32 sits five orders below it; bf16's
// own rounding (about 6e-3 here) is a fixed floor, so its fraction is a
// quarter.
const F32_SIGNAL_FRACTION: f64 = 0.01;
const BF16_SIGNAL_FRACTION: f64 = 0.25;

/// The two shapes: headdim 32 (warp tier) and headdim 64 with d_state 128
/// (block tier, the official checkpoints' head). The MLP width 300 rounds
/// up to 384, so the GEMMs see a width that is no power of two.
fn shapes() -> [(Mamba3Config, &'static str); 2] {
    let base = Mamba3Config {
        d_model: 128,
        expand: 2,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: false,
        d_intermediate: 300,
        ..Mamba3Config::default()
    };
    [
        (
            Mamba3Config {
                headdim: 32,
                d_state: 64,
                ..base
            },
            "warp hd32",
        ),
        (
            Mamba3Config {
                headdim: 64,
                d_state: 128,
                ..base
            },
            "block hd64",
        ),
    ]
}

/// The CPU decode outputs of `inputs`, with and without the MLP.
fn cpu_decode(cfg: &Mamba3Config, w: &Mamba3Weights, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut state = Mamba3State::zeros(cfg);
    let mut scratch = Mamba3StepScratch::new(cfg);
    inputs
        .iter()
        .map(|x| {
            let mut out = vec![0f32; cfg.d_model];
            mamba3_step(&mut out, x, &mut scratch, w, &mut state.layers, cfg);
            out
        })
        .collect()
}

fn without_mlp(cfg: &Mamba3Config, w: &Mamba3Weights) -> (Mamba3Config, Mamba3Weights) {
    let cfg0 = Mamba3Config {
        d_intermediate: 0,
        ..*cfg
    };
    let mut w0 = w.clone();
    w0.mlp.clear();
    (cfg0, w0)
}

fn check(label: &str, got: &[f32], want: &[f32], no_mlp: &[f32], tol: f64) {
    let err = rel_l2(got, want);
    let signal = rel_l2(no_mlp, want);
    let fraction = if tol == F32_TOL {
        F32_SIGNAL_FRACTION
    } else {
        BF16_SIGNAL_FRACTION
    };
    eprintln!("{label}: rel_l2={err:.3e}, MLP contribution={signal:.3e}");
    assert!(err < tol, "{label}: rel_l2 {err:.3e} over {tol:.1e}");
    assert!(
        err < fraction * signal,
        "{label}: error {err:.3e} is not small against the MLP's {signal:.3e}"
    );
}

/// Decode steps on the f32 and the bf16 engine against the CPU step chain.
#[test]
fn m3_gpu_decode_with_mlp_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    for (cfg, name) in shapes() {
        cfg.validate().unwrap();
        let d = cfg.d_model;
        let mut w = Mamba3Weights::init(&cfg, d, 0x31F);
        // The identity input projection of a language model: the bf16
        // engine's decode serves only that path.
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        assert_eq!(w.mlp.len(), cfg.n_layers);
        let inputs: Vec<Vec<f32>> = (0..10u64).map(|t| det(d, 500 + t, 0.5)).collect();
        let want = cpu_decode(&cfg, &w, &inputs);
        let (cfg0, w0) = without_mlp(&cfg, &w);
        let bare = cpu_decode(&cfg0, &w0, &inputs);

        let engine = Mamba3GpuEngine::new(&device, &w, cfg, d, 1).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let mut out = vec![0f32; d];
        for (t, x) in inputs.iter().enumerate() {
            engine.step(x, &mut out, &mut state, &mut scratch).unwrap();
            check(
                &format!("{name} f32 step {t}"),
                &out,
                &want[t],
                &bare[t],
                F32_TOL,
            );
        }

        let mixed = Mamba3GpuEngineMixed::new(&device, &w, cfg, d, 1, WeightDtype::Bf16).unwrap();
        let mut state = mixed.alloc_state().unwrap();
        let mut scratch = mixed.alloc_mixed_scratch().unwrap();
        for (t, x) in inputs.iter().enumerate() {
            mixed
                .step_mixed_native(x, &mut out, &mut state, &mut scratch)
                .unwrap();
            check(
                &format!("{name} bf16 step {t}"),
                &out,
                &want[t],
                &bare[t],
                BF16_TOL,
            );
        }
    }
}

/// A prompt window, its length not a whole number of chunks, prefilled on
/// the f32 and the bf16 engine and then decoded, against the CPU prefill
/// and step chain.
#[test]
fn m3_gpu_prefill_with_mlp_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let seq_len = 45usize;
    for (cfg, name) in shapes() {
        cfg.validate().unwrap();
        let d = cfg.d_model;
        let mut w = Mamba3Weights::init(&cfg, d, 0x51F);
        // The identity input projection of a language model: the bf16
        // engine's decode serves only that path.
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        let (cfg0, w0) = without_mlp(&cfg, &w);
        let prompt = det(seq_len * d, 9, 0.5);
        let decode: Vec<Vec<f32>> = (0..4u64).map(|t| det(d, 700 + t, 0.5)).collect();

        // The CPU reference: prefill rows and the decode after them, with
        // and without the MLP.
        let cpu = |cfg: &Mamba3Config, w: &Mamba3Weights| -> (Vec<f32>, Vec<Vec<f32>>) {
            let dims = Mamba3Dims::from_config(cfg, seq_len);
            let mut state = Mamba3State::zeros(cfg);
            let mut scratch = Mamba3PrefillScratch::new(&dims, d);
            let mut rows = vec![0f32; seq_len * d];
            forward_mamba3_backbone_prefill(
                &mut rows,
                &prompt,
                w,
                &mut state.layers,
                &mut scratch,
                &dims,
            );
            let mut step = Mamba3StepScratch::new(cfg);
            let outs = decode
                .iter()
                .map(|x| {
                    let mut out = vec![0f32; d];
                    mamba3_step(&mut out, x, &mut step, w, &mut state.layers, cfg);
                    out
                })
                .collect();
            (rows[(seq_len - 1) * d..].to_vec(), outs)
        };
        let (want_last, want_dec) = cpu(&cfg, &w);
        let (bare_last, bare_dec) = cpu(&cfg0, &w0);

        let engine = Mamba3GpuEngine::new(&device, &w, cfg, d, 1).unwrap();
        let stream = &engine.ctx().stream;
        let mut prefill = engine.alloc_prefill(seq_len).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let gpu_prompt = GpuBuffer::from_cpu(stream, &prompt).unwrap();
        let mut last = GpuBuffer::zeros(stream, d).unwrap();
        engine
            .prefill_sequence(
                &mut prefill,
                &gpu_prompt,
                seq_len,
                &mut state,
                false,
                &mut last,
            )
            .unwrap();
        stream.synchronize().unwrap();
        let mut got = vec![0f32; d];
        last.download(stream, &mut got).unwrap();
        stream.synchronize().unwrap();
        check(
            &format!("{name} f32 prefill last row"),
            &got,
            &want_last,
            &bare_last,
            F32_TOL,
        );
        for (t, x) in decode.iter().enumerate() {
            engine.step(x, &mut got, &mut state, &mut scratch).unwrap();
            check(
                &format!("{name} f32 decode after prefill {t}"),
                &got,
                &want_dec[t],
                &bare_dec[t],
                F32_TOL,
            );
        }

        let mixed = Mamba3GpuEngineMixed::new(&device, &w, cfg, d, 1, WeightDtype::Bf16).unwrap();
        let stream = mixed.ctx_stream();
        let mut prefill = mixed.alloc_prefill(seq_len).unwrap();
        let mut state = mixed.alloc_state().unwrap();
        let mut scratch = mixed.alloc_mixed_scratch().unwrap();
        let gpu_prompt = GpuBuffer::from_cpu(stream, &prompt).unwrap();
        let mut last = GpuBuffer::zeros(stream, d).unwrap();
        mixed
            .prefill_sequence(
                &mut prefill,
                &gpu_prompt,
                seq_len,
                &mut state,
                false,
                &mut last,
            )
            .unwrap();
        stream.synchronize().unwrap();
        last.download(stream, &mut got).unwrap();
        stream.synchronize().unwrap();
        check(
            &format!("{name} bf16 prefill last row"),
            &got,
            &want_last,
            &bare_last,
            BF16_TOL,
        );
        for (t, x) in decode.iter().enumerate() {
            mixed
                .step_mixed_native(x, &mut got, &mut state, &mut scratch)
                .unwrap();
            check(
                &format!("{name} bf16 decode after prefill {t}"),
                &got,
                &want_dec[t],
                &bare_dec[t],
                BF16_TOL,
            );
        }
    }
}
