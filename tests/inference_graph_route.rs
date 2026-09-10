#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::context::{
    BiGemmFamily, F32TriadPolicy, GemmMode, GpuCtx, HalfTriadPolicy,
};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::inference::{GpuMambaInference, GpuMambaInferenceMixed};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::{Mamba3GpuInferenceEngine, Mamba3GpuInferenceMixed};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::weights::MambaWeights;

const OUTPUT_POISON_BITS: u32 = 0x7fc0_d00d;

fn configure_decode_route(ctx: &GpuCtx, tensor_cores: bool, family: BiGemmFamily) {
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(family);
    ctx.set_bi_tensor_cores(tensor_cores);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    ctx.set_half_triad_policy(HalfTriadPolicy::TiledParityV1);
}

fn assert_decode_route(ctx: &GpuCtx, label: &str, tensor_cores: bool, family: BiGemmFamily) {
    let route = ctx.gemm_route();
    assert!(
        route.policy.batch_invariant,
        "{label} batch-invariant route"
    );
    assert_eq!(
        route.policy.bi_gemm_family, family,
        "{label} selected family"
    );
    assert_eq!(
        route.policy.f32_triad_policy,
        F32TriadPolicy::ExactScalarFmaV1,
        "{label} exact-F32 policy"
    );
    assert_eq!(
        route.policy.bi_tensor_cores, tensor_cores,
        "{label} tensor-core tier"
    );
    assert_eq!(
        route.policy.half_triad_policy,
        HalfTriadPolicy::TiledParityV1,
        "{label} tiled-parity half policy"
    );
    assert!(!route.policy.fast_gemm, "{label} fast GEMM disabled");
}

fn poison_output(output: &mut [f32]) {
    output.fill(f32::from_bits(OUTPUT_POISON_BITS));
}

fn assert_positive_graph_output(output: &[f32], label: &str) {
    assert!(
        output.iter().all(|value| value.is_finite()),
        "{label} graph replay must replace every poisoned output with a finite value"
    );
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() != OUTPUT_POISON_BITS),
        "{label} graph replay retained output poison"
    );
}

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
    decode_graphs_reject_route_drift(BiGemmFamily::Triad);
}

#[test]
fn inference_decode_graphs_reject_complete_route_drift() {
    decode_graphs_reject_route_drift(BiGemmFamily::Inference);
}

fn decode_graphs_reject_route_drift(family: BiGemmFamily) {
    let device = GpuDevice::new(0).expect("CUDA device");
    let input = vec![0.01; 32];
    let mut output = vec![0.0; 32];

    let cfg = m1_config();
    let mut weights = MambaWeights::init(&cfg, cfg.d_model, 0xDEC0_DE01);
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut f32 = GpuMambaInference::new(&device, &weights, cfg, cfg.d_model, 1).expect("M1 f32");
    configure_decode_route(f32.ctx(), false, family);
    assert_decode_route(f32.ctx(), "M1 f32", false, family);
    let captured_route = f32.ctx().gemm_route();
    let mut state = f32.alloc_state().expect("M1 f32 state");
    let mut scratch = f32.alloc_scratch().expect("M1 f32 scratch");
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 f32 warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { f32.capture_graph(&mut state, &mut scratch) }.expect("M1 f32 capture");
    assert!(f32.has_graph(), "M1 f32 graph exists after capture");
    assert_eq!(f32.ctx().gemm_route(), captured_route, "M1 f32 route pin");
    poison_output(&mut output);
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 f32 positive graph replay");
    assert_positive_graph_output(&output, "M1 f32");
    f32.ctx().set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    assert!(
        f32.step(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M1 f32 route drift")
            .starts_with("inference graph replay: GEMM route changed")
    );
    drop(f32);

    let mut mixed =
        GpuMambaInferenceMixed::new(&device, &weights, cfg, cfg.d_model, 1, WeightDtype::Bf16)
            .expect("M1 mixed");
    configure_decode_route(mixed.ctx(), true, family);
    assert_decode_route(mixed.ctx(), "M1 BF16", true, family);
    let captured_route = mixed.ctx().gemm_route();
    let mut state = mixed.alloc_state().expect("M1 mixed state");
    let mut scratch = mixed.alloc_mixed_scratch().expect("M1 mixed scratch");
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 mixed warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { mixed.capture_graph_mixed_native(&mut state, &mut scratch) }
        .expect("M1 mixed capture");
    assert!(mixed.has_graph(), "M1 BF16 graph exists after capture");
    assert_eq!(
        mixed.ctx().gemm_route(),
        captured_route,
        "M1 BF16 route pin"
    );
    poison_output(&mut output);
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M1 BF16 positive graph replay");
    assert_positive_graph_output(&output, "M1 BF16");
    mixed.ctx().set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    assert!(
        mixed
            .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M1 mixed route drift")
            .starts_with("mixed inference graph replay: GEMM route changed")
    );
    drop(mixed);

    let cfg = m3_config();
    let mut weights = Mamba3Weights::init(&cfg, cfg.d_model, 0xDEC0_DE03);
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut f32 =
        Mamba3GpuInferenceEngine::new(&device, &weights, cfg, cfg.d_model, 1).expect("M3 f32");
    configure_decode_route(f32.ctx(), false, family);
    assert_decode_route(f32.ctx(), "M3 f32", false, family);
    let captured_route = f32.ctx().gemm_route();
    let mut state = f32.alloc_state().expect("M3 f32 state");
    let mut scratch = f32.alloc_scratch().expect("M3 f32 scratch");
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 f32 warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { f32.capture_graph(&mut state, &mut scratch) }.expect("M3 f32 capture");
    assert!(f32.has_graph(), "M3 f32 graph exists after capture");
    assert_eq!(f32.ctx().gemm_route(), captured_route, "M3 f32 route pin");
    poison_output(&mut output);
    f32.step(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 f32 positive graph replay");
    assert_positive_graph_output(&output, "M3 f32");
    f32.ctx().set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    assert!(
        f32.step(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M3 f32 route drift")
            .starts_with("M3 inference graph replay: GEMM route changed")
    );
    drop(f32);

    let mut mixed =
        Mamba3GpuInferenceMixed::new(&device, &weights, cfg, cfg.d_model, 1, WeightDtype::Bf16)
            .expect("M3 mixed");
    configure_decode_route(mixed.engine_ref().ctx(), true, family);
    assert_decode_route(mixed.engine_ref().ctx(), "M3 BF16", true, family);
    let captured_route = mixed.engine_ref().ctx().gemm_route();
    let mut state = mixed.alloc_state().expect("M3 mixed state");
    let mut scratch = mixed.alloc_mixed_scratch().expect("M3 mixed scratch");
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 mixed warmup");
    // State and scratch stay alive until the engine graph is cleared.
    unsafe { mixed.capture_graph_mixed_native(&mut state, &mut scratch) }
        .expect("M3 mixed capture");
    assert!(mixed.has_graph(), "M3 BF16 graph exists after capture");
    assert_eq!(
        mixed.engine_ref().ctx().gemm_route(),
        captured_route,
        "M3 BF16 route pin"
    );
    poison_output(&mut output);
    mixed
        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
        .expect("M3 BF16 positive graph replay");
    assert_positive_graph_output(&output, "M3 BF16");
    mixed
        .engine_ref()
        .ctx()
        .set_gemm_mode(GemmMode::CublasPedantic)
        .unwrap();
    assert!(
        mixed
            .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
            .expect_err("M3 mixed route drift")
            .starts_with("M3 mixed inference graph replay: GEMM route changed")
    );
    drop(mixed);
}
