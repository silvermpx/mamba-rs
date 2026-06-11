//! Step 20 — verify f16 inference works end-to-end through the public
//! `GpuMambaBackbone` / `GpuMamba3Backbone` wrappers.
//!
//! The audit flagged f16 inference as "not in wrapper API" — false alarm:
//! both wrappers have always dispatched `Bf16 | F16` into the `Mixed`
//! engine path that's parameterized on dtype. This test confirms a clean
//! end-to-end run: construct → step → finite output.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

#[test]
fn m1_inference_f16_wrapper_smoke() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::inference::GpuMambaBackbone;
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let batch = 1;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xF16_C0FF);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut bb = GpuMambaBackbone::new_with_dtype(0, &cpu, cfg, input_dim, batch, WeightDtype::F16)
        .expect("construct f16 backbone");
    assert_eq!(bb.dtype(), WeightDtype::F16);

    let input = vec![0.05f32; batch * input_dim];
    let mut out = vec![0.0f32; batch * 64];
    bb.step(&input, &mut out).expect("f16 step");
    assert!(out.iter().all(|v| v.is_finite()), "f16 output non-finite");
    eprintln!("M1 f16 inference: out[0..4]={:?}", &out[..4]);
}

#[test]
fn m3_inference_f16_wrapper_smoke() {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 64,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0xF16_DECA);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    let mut bb =
        GpuMamba3Backbone::new_with_dtype(0, &cpu, cfg, input_dim, batch, WeightDtype::F16)
            .expect("construct M3 f16 backbone");
    assert_eq!(bb.dtype(), WeightDtype::F16);

    let input = vec![0.05f32; batch * input_dim];
    let mut out = vec![0.0f32; batch * 64];
    bb.step(&input, &mut out).expect("M3 f16 step");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "M3 f16 output non-finite"
    );
    eprintln!("M3 f16 inference: out[0..4]={:?}", &out[..4]);
}
