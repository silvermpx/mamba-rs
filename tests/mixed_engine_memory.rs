//! The bf16 inference engines of Mamba-1 and Mamba-3 SISO keep only their
//! own weights on the device. Building one grows the device's memory by
//! less than an f32 copy of the model would take: they used to upload such
//! a copy beside the half one and never read it. The models are deep
//! enough that their weights, not the context's own buffers, decide the
//! growth.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::mamba1::gpu::inference::MambaGpuEngineMixed;
use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuEngineMixed;
use mamba_rs::{Mamba3Config, Mamba3Weights, MambaConfig, MambaWeights, WeightDtype};

/// Bytes of device memory in use, by every process on the device.
fn used(device: &GpuDevice) -> usize {
    let (free, total) = device.context().mem_get_info().unwrap();
    total - free
}

/// Bytes an f32 copy of every tensor of a Mamba-3 SISO model takes.
fn m3_f32_bytes(w: &Mamba3Weights) -> usize {
    let mut n = w.input_proj_w.len() + w.input_proj_b.len() + w.norm_f_weight.len();
    for l in &w.layers {
        n += [
            &l.norm_weight,
            &l.in_proj_w,
            &l.dt_bias,
            &l.b_norm_weight,
            &l.c_norm_weight,
            &l.b_bias,
            &l.c_bias,
            &l.d_param,
            &l.norm_gate_weight,
            &l.out_proj_w,
        ]
        .iter()
        .map(|t| t.len())
        .sum::<usize>();
    }
    4 * n
}

#[test]
fn half_engines_hold_no_f32_copy_of_the_model() {
    let device = GpuDevice::new(0).unwrap();

    let cfg = Mamba3Config {
        d_model: 1024,
        n_layers: 32,
        headdim: 64,
        d_state: 16,
        ..Mamba3Config::default()
    };
    let w = Mamba3Weights::init(&cfg, cfg.d_model, 5);
    let f32_bytes = m3_f32_bytes(&w);
    let before = used(&device);
    let engine =
        Mamba3GpuEngineMixed::new(&device, &w, cfg, cfg.d_model, 1, WeightDtype::Bf16).unwrap();
    engine.ctx_stream().synchronize().unwrap();
    let grown = used(&device).saturating_sub(before);
    drop(engine);
    eprintln!("Mamba-3 SISO bf16 engine: {grown} B on the device, f32 weights {f32_bytes} B");
    assert!(
        grown < f32_bytes,
        "the Mamba-3 SISO bf16 engine took {grown} B, more than the {f32_bytes} B of an f32 \
         copy of its weights"
    );

    let cfg = MambaConfig {
        d_model: 1024,
        n_layers: 32,
        ..MambaConfig::default()
    };
    let w = MambaWeights::init(&cfg, cfg.d_model, 5);
    let f32_bytes = 4 * w.param_count(cfg.d_model, &cfg);
    let before = used(&device);
    let engine =
        MambaGpuEngineMixed::new(&device, &w, cfg, cfg.d_model, 1, WeightDtype::Bf16).unwrap();
    engine.ctx().stream.synchronize().unwrap();
    let grown = used(&device).saturating_sub(before);
    drop(engine);
    eprintln!("Mamba-1 bf16 engine: {grown} B on the device, f32 weights {f32_bytes} B");
    assert!(
        grown < f32_bytes,
        "the Mamba-1 bf16 engine took {grown} B, more than the {f32_bytes} B of an f32 copy \
         of its weights"
    );
}
