#![cfg(feature = "cuda")]

//! A Mamba-1 prompt as the first GEMM work of a process. The Hopper and
//! datacenter-Blackwell Inference rungs prove themselves once per process,
//! on first use, and that proof cannot run while the prompt records its
//! GEMM manifest; a bf16 prompt that came before any decode step failed on
//! those boards. The verdict is process-wide, so this binary holds one test
//! and nothing runs a GEMM before it. Other boards have no such rung and
//! pass trivially.

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaConfig;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;

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

#[test]
fn a_bf16_prompt_can_be_the_first_gemm_work_of_a_process() {
    if let Err(error) = GpuDevice::new(0) {
        eprintln!("skipping the first-use prompt check without a CUDA device: {error}");
        return;
    }
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        ..MambaConfig::default()
    };
    let weights = MambaWeights::init(&cfg, cfg.d_model, 7);
    let options = GpuOptions::default().dtype(WeightDtype::Bf16);
    let mut backbone = MambaGpuBackbone::new(&weights, cfg, cfg.d_model, 1, options).unwrap();
    let seq_len = 64;
    let prompt = det(seq_len * cfg.d_model, 11);
    backbone.prefill(&prompt, seq_len).unwrap();
    let mut rows = vec![0.0f32; cfg.d_model];
    backbone.download_temporal(&mut rows).unwrap();
    assert!(rows.iter().all(|value| value.is_finite()));
    assert!(rows.iter().any(|&value| value != 0.0));
}
