#![cfg(feature = "cuda")]

//! Bit recorder for the Mamba-1 parallel prefill: the one-pass prompt
//! window through the parallel scan, then a few decode steps that read the
//! state the window left behind. The decode recorder starts every model
//! from zero state and the training recorders run the saving scan, so
//! this is the only recorder of the inference scan.
//!
//! Run before and after a kernel change and compare the printed lines:
//! `cargo test --release --features cuda --test m1_prefill_digest --
//! --ignored --nocapture`.

#[path = "common/digest.rs"]
mod digest;

use digest::{Digest, fnv1a_f32};
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;
use mamba_rs::{config::ScanMode, mamba1::MambaConfig};

const DECODE_STEPS: usize = 3;

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

/// The digests of the window's last row and of the decode outputs after it.
fn run(cfg: MambaConfig, batch: usize, t: usize, dtype: WeightDtype) -> (u64, u64) {
    let mut w = MambaWeights::init(&cfg, cfg.d_model, 0x5CA7);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let mut bb = MambaGpuBackbone::new(
        &w,
        cfg,
        cfg.d_model,
        batch,
        GpuOptions::default().dtype(dtype),
    )
    .unwrap();
    let prompt = det(batch * t * cfg.d_model, 41);
    let mut input = GpuBuffer::zeros(bb.stream(), prompt.len()).unwrap();
    input.upload(bb.stream(), &prompt).unwrap();
    if dtype == WeightDtype::F32 {
        let mut scratch = bb.alloc_prefill_scratch(t).unwrap();
        bb.prefill_sequence(&input, &mut scratch).unwrap();
    } else {
        let mut scratch = bb.alloc_prefill_mixed_scratch(t).unwrap();
        bb.prefill_sequence_mixed(&input, &mut scratch).unwrap();
    }
    bb.stream().synchronize().unwrap();
    let mut row = vec![0f32; batch * cfg.d_model];
    bb.download_temporal(&mut row).unwrap();
    let window = fnv1a_f32(&row);
    let mut steps = Digest::new();
    for s in 0..DECODE_STEPS {
        bb.step(&det(batch * cfg.d_model, 100 + s as u64), &mut row)
            .unwrap();
        steps.absorb_bytes(b"step").absorb_f32(&row);
    }
    (window, steps.finish())
}

#[test]
#[ignore = "bit recorder: prints digests to compare across a kernel change"]
fn print_m1_prefill_digests() {
    // Several chunks with a ragged last one; a window inside one chunk; a
    // d_inner that is a multiple of four but not of eight; a wider state.
    let shapes = [
        (384usize, 3usize, 16usize, 1usize, 4621usize),
        (256, 2, 16, 2, 1500),
        (128, 2, 16, 3, 333),
        (66, 2, 16, 2, 1100),
        (128, 2, 64, 1, 2100),
    ];
    for (d_model, n_layers, d_state, batch, t) in shapes {
        let cfg = MambaConfig {
            d_model,
            n_layers,
            d_state,
            d_conv: 4,
            expand: 2,
            scan_mode: ScanMode::Parallel,
            rms_norm_eps: 1e-5,
            bcdt_rms_eps: None,
            dt_rank: None,
            proj_bias: false,
            conv_bias: true,
        };
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let (window, steps) = run(cfg, batch, t, dtype);
            println!(
                "m1 prefill d{d_model} L{n_layers} N{d_state} B{batch} T{t} {dtype:?}: \
                 window {window:016x} steps {steps:016x}"
            );
        }
    }
}
