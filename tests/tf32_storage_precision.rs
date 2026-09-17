//! `WeightDtype::Tf32` is f32 storage whose deterministic GEMMs may take the
//! TF32 kernels. The constructors must carry it to the context's f32 numeric
//! contract, report it back, and produce the bits the route produced when it
//! was still spelled as a policy on an f32 context.

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GemmMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::inference::GpuMambaBackbone;
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::weights::MambaWeights;

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    }
}

fn m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    }
}

fn session(input_dim: usize) -> TrainSessionCfg {
    TrainSessionCfg {
        input_dim,
        batch: 4,
        seq_len: 8,
        lr: 1e-3,
        weight_decay: 0.0,
    }
}

fn ramp(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index * 7 + salt * 13) % 23) as f32 / 23.0 - 0.5)
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn tf32_storage_reaches_the_context_and_reports_itself() {
    let cfg = m1_cfg();
    let weights = MambaWeights::init(&cfg, cfg.d_model, 0x7f32_0001);

    let model =
        GpuMambaBackbone::new_with_dtype(0, &weights, cfg, cfg.d_model, 2, WeightDtype::Tf32)
            .expect("Tf32 backbone");
    assert_eq!(model.dtype(), WeightDtype::Tf32);
    assert_eq!(model.ctx().gemm_mode(), GemmMode::Deterministic);
    let controls = model.ctx().route_controls();
    assert_eq!(controls.family(), BiGemmFamily::Inference);
    assert_eq!(
        controls.f32_policy(),
        F32TriadPolicy::AllowDeterministicTf32
    );

    let exact =
        GpuMambaBackbone::new_with_dtype(0, &weights, cfg, cfg.d_model, 2, WeightDtype::F32)
            .expect("F32 backbone");
    assert_eq!(exact.dtype(), WeightDtype::F32);
    assert_eq!(
        exact.ctx().route_controls().f32_policy(),
        F32TriadPolicy::ExactScalarFma
    );

    let trainer = MambaTrainer::new_full(0, &weights, cfg, session(cfg.d_model), WeightDtype::Tf32)
        .expect("Tf32 trainer");
    assert_eq!(trainer.dtype(), WeightDtype::Tf32);
    let controls = trainer.ctx().route_controls();
    assert_eq!(controls.family(), BiGemmFamily::Triad);
    assert_eq!(
        controls.f32_policy(),
        F32TriadPolicy::AllowDeterministicTf32
    );

    let vendor = MambaTrainer::new_full_with_mode(
        0,
        &weights,
        cfg,
        session(cfg.d_model),
        WeightDtype::Tf32,
        GemmMode::CublasPedantic,
    )
    .expect("Tf32 trainer in a cuBLAS mode");
    assert_eq!(vendor.dtype(), WeightDtype::Tf32);
    assert_eq!(vendor.ctx().gemm_mode(), GemmMode::CublasPedantic);

    let cfg3 = m3_cfg();
    let weights3 = Mamba3Weights::init(&cfg3, cfg3.d_model, 0x7f32_0003);
    let model3 =
        GpuMamba3Backbone::new_with_dtype(0, &weights3, cfg3, cfg3.d_model, 2, WeightDtype::Tf32)
            .expect("Tf32 M3 backbone");
    assert_eq!(model3.dtype(), WeightDtype::Tf32);
    assert_eq!(
        model3.ctx().route_controls().f32_policy(),
        F32TriadPolicy::AllowDeterministicTf32
    );
    let trainer3 =
        Mamba3Trainer::new_full(0, &weights3, cfg3, session(cfg3.d_model), WeightDtype::Tf32)
            .expect("Tf32 M3 trainer");
    assert_eq!(trainer3.dtype(), WeightDtype::Tf32);
    assert_eq!(
        trainer3.ctx().route_controls().f32_policy(),
        F32TriadPolicy::AllowDeterministicTf32
    );
}

/// The storage precision is the same route the f32 context reaches when its
/// numeric contract is switched to TF32 permission by hand: bit for bit,
/// on a model step and on a training step.
#[test]
fn tf32_storage_produces_the_bits_of_the_tf32_permitted_f32_route() {
    let cfg = m1_cfg();
    let weights = MambaWeights::init(&cfg, cfg.d_model, 0x7f32_0011);
    let batch = 2;
    let input = ramp(batch * cfg.d_model, 1);

    let mut by_dtype =
        GpuMambaBackbone::new_with_dtype(0, &weights, cfg, cfg.d_model, batch, WeightDtype::Tf32)
            .expect("Tf32 backbone");
    let mut by_policy =
        GpuMambaBackbone::new_with_dtype(0, &weights, cfg, cfg.d_model, batch, WeightDtype::F32)
            .expect("F32 backbone");
    by_policy
        .ctx()
        .route_controls()
        .set_f32_policy(F32TriadPolicy::AllowDeterministicTf32);
    let mut out_dtype = vec![0.0f32; batch * cfg.d_model];
    let mut out_policy = vec![0.0f32; batch * cfg.d_model];
    for _ in 0..3 {
        by_dtype.step(&input, &mut out_dtype).expect("Tf32 step");
        by_policy
            .step(&input, &mut out_policy)
            .expect("policy step");
        assert_eq!(bits(&out_dtype), bits(&out_policy));
    }
    assert!(out_dtype.iter().all(|value| value.is_finite()));

    let sess = session(cfg.d_model);
    let mut train_dtype =
        MambaTrainer::new_full(0, &weights, cfg, sess, WeightDtype::Tf32).expect("Tf32 trainer");
    let mut train_policy =
        MambaTrainer::new_full(0, &weights, cfg, sess, WeightDtype::F32).expect("F32 trainer");
    train_policy
        .ctx()
        .route_controls()
        .set_f32_policy(F32TriadPolicy::AllowDeterministicTf32);
    let x = ramp(sess.batch * sess.seq_len * cfg.d_model, 2);
    let d_temporal = ramp(sess.batch * sess.seq_len * cfg.d_model, 3);
    for _ in 0..2 {
        train_dtype.step(&x, &d_temporal).expect("Tf32 train step");
        train_policy
            .step(&x, &d_temporal)
            .expect("policy train step");
    }
    let a = train_dtype.snapshot_master().expect("Tf32 master");
    let b = train_policy.snapshot_master().expect("policy master");
    assert_eq!(bits(&a.input_proj_w), bits(&b.input_proj_w));
    for (layer_a, layer_b) in a.layers.iter().zip(&b.layers) {
        assert_eq!(bits(&layer_a.in_proj_w), bits(&layer_b.in_proj_w));
        assert_eq!(bits(&layer_a.out_proj_w), bits(&layer_b.out_proj_w));
        assert_eq!(bits(&layer_a.x_proj_w), bits(&layer_b.x_proj_w));
    }
}
