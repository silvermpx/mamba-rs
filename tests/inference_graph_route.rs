#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::inference::{GpuMambaInference, GpuMambaInferenceMixed};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::{Mamba3GpuInferenceEngine, Mamba3GpuInferenceMixed};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::weights::MambaWeights;

fn m1_config() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    }
}

fn m3_config() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    }
}

#[test]
fn decode_graphs_reject_complete_route_drift() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let input = vec![0.01; 32];
    let mut output = vec![0.0; 32];

    let cfg = m1_config();
    let mut weights = MambaWeights::init(&cfg, cfg.d_model, 0xDEC0_DE01);
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut f32 = GpuMambaInference::new(&device, &weights, cfg, cfg.d_model, 1).expect("M1 f32");
    let mut state = f32.alloc_state().expect("M1 f32 state");
    let mut scratch = f32.alloc_scratch().expect("M1 f32 scratch");
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 f32 warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { f32.capture_graph(&mut state, &mut scratch) }.expect("M1 f32 capture");
    f32.ctx().set_fast_gemm(true);
    assert!(
        f32.step(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M1 f32 route drift")
            .starts_with("inference graph replay: GEMM route changed")
    );

    let mut mixed =
        GpuMambaInferenceMixed::new(&device, &weights, cfg, cfg.d_model, 1, WeightDtype::Bf16)
            .expect("M1 mixed");
    let mut state = mixed.alloc_state().expect("M1 mixed state");
    let mut scratch = mixed.alloc_mixed_scratch().expect("M1 mixed scratch");
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 mixed warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { mixed.capture_graph_mixed_native(&mut state, &mut scratch) }
        .expect("M1 mixed capture");
    mixed.ctx().set_fast_gemm(true);
    assert!(
        mixed
            .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M1 mixed route drift")
            .starts_with("mixed inference graph replay: GEMM route changed")
    );

    let cfg = m3_config();
    let mut weights = Mamba3Weights::init(&cfg, cfg.d_model, 0xDEC0_DE03);
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut f32 =
        Mamba3GpuInferenceEngine::new(&device, &weights, cfg, cfg.d_model, 1).expect("M3 f32");
    let mut state = f32.alloc_state().expect("M3 f32 state");
    let mut scratch = f32.alloc_scratch().expect("M3 f32 scratch");
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 f32 warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { f32.capture_graph(&mut state, &mut scratch) }.expect("M3 f32 capture");
    f32.ctx().set_fast_gemm(true);
    assert!(
        f32.step(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M3 f32 route drift")
            .starts_with("M3 inference graph replay: GEMM route changed")
    );

    let mut mixed =
        Mamba3GpuInferenceMixed::new(&device, &weights, cfg, cfg.d_model, 1, WeightDtype::Bf16)
            .expect("M3 mixed");
    let mut state = mixed.alloc_state().expect("M3 mixed state");
    let mut scratch = mixed.alloc_mixed_scratch().expect("M3 mixed scratch");
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 mixed warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { mixed.capture_graph_mixed_native(&mut state, &mut scratch) }
        .expect("M3 mixed capture");
    mixed.engine_ref().ctx().set_fast_gemm(true);
    assert!(
        mixed
            .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M3 mixed route drift")
            .starts_with("M3 mixed inference graph replay: GEMM route changed")
    );
}
