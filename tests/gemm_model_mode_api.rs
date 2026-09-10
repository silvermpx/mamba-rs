#![cfg(feature = "cuda")]

#[cfg(feature = "hf")]
#[path = "common/hf_synthetic.rs"]
mod hf_synthetic;

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::inference::{
    GpuMambaBackbone, GpuMambaInference, GpuMambaInferenceMixed,
};
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::{
    GpuMamba3Backbone, Mamba3GpuInferenceEngine, Mamba3GpuInferenceMixed,
};
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::weights::MambaWeights;
use std::process::Command;

#[cfg(feature = "hf")]
use hf_synthetic::write_synthetic_checkpoint;
#[cfg(feature = "hf")]
use mamba_rs::module::gpu_lm::GpuMambaLM;
#[cfg(feature = "hf")]
use mamba_rs::module::gpu_lm3::{GpuMamba3LM, Mamba3LmBuild};

const CHILD_CASE: &str = "MAMBA_RS_TEST_MODEL_MODE_CASE";
const GEMM_ENV: [&str; 8] = [
    "MAMBA_RS_GEMM_MODE",
    "MAMBA_RS_BATCH_INVARIANT",
    "MAMBA_RS_FAST_GEMM",
    "MAMBA_RS_BI_TENSOR_CORES",
    "MAMBA_RS_BI_F32_POLICY",
    "MAMBA_RS_BI_HALF_POLICY",
    "MAMBA_RS_BI_GEMM_FAMILY",
    "MAMBA_RS_ARCH_RUNG",
];

fn small_m1_cfg() -> MambaConfig {
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

fn small_m3_cfg() -> Mamba3Config {
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

fn m1_weights(cfg: &MambaConfig, identity: bool) -> MambaWeights {
    let mut weights = MambaWeights::init(cfg, cfg.d_model, 0x9060_0001);
    if identity {
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
    }
    weights
}

fn m3_weights(cfg: &Mamba3Config, identity: bool) -> Mamba3Weights {
    let mut weights = Mamba3Weights::init(cfg, cfg.d_model, 0x9060_0003);
    if identity {
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
    } else {
        weights.input_proj_w.fill(0.0);
        for i in 0..cfg.d_model {
            weights.input_proj_w[i * cfg.d_model + i] = 1.0;
        }
        weights.input_proj_b.fill(0.0);
    }
    weights
}

fn session(d_model: usize) -> TrainSessionCfg {
    TrainSessionCfg {
        input_dim: d_model,
        batch: 1,
        seq_len: 2,
        lr: 1e-4,
        weight_decay: 0.0,
    }
}

fn assert_route(ctx: &GpuCtx, mode: GemmMode, family: BiGemmFamily) {
    assert_eq!(ctx.gemm_mode(), mode);
    assert_eq!(ctx.bi_gemm_family(), family);
}

fn construction_error<T>(result: Result<T, String>, label: &str) -> String {
    match result {
        Ok(_) => panic!("{label}: construction unexpectedly succeeded"),
        Err(error) => error,
    }
}

#[test]
fn m1_default_constructor_uses_inference_family() {
    let cfg = small_m1_cfg();
    let weights = m1_weights(&cfg, true);

    let backbone = GpuMambaBackbone::new(0, &weights, cfg, cfg.d_model, 1)
        .expect("construct M1 F32 backbone from owned weights");

    assert_eq!(backbone.ctx().gemm_mode(), GemmMode::Deterministic);
    assert_eq!(backbone.ctx().bi_gemm_family(), BiGemmFamily::Inference);
}

#[test]
fn constructor_environment_cases_are_process_isolated() {
    let cases: [(&str, &[(&str, &str)]); 6] = [
        ("absent", &[]),
        ("empty_family", &[("MAMBA_RS_BI_GEMM_FAMILY", "")]),
        (
            "conflict_explicit",
            &[
                ("MAMBA_RS_GEMM_MODE", "deterministic"),
                ("MAMBA_RS_FAST_GEMM", "true"),
                ("MAMBA_RS_BI_GEMM_FAMILY", "triad"),
            ],
        ),
        ("invalid_mode", &[("MAMBA_RS_GEMM_MODE", "invalid")]),
        ("invalid_family", &[("MAMBA_RS_BI_GEMM_FAMILY", "invalid")]),
        ("errors", &[]),
    ];
    for (case, vars) in cases {
        let mut command = Command::new(std::env::current_exe().expect("current test binary"));
        command.args([
            "--exact",
            "constructor_env_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]);
        command.env(CHILD_CASE, case);
        for name in GEMM_ENV {
            command.env_remove(name);
        }
        for (name, value) in vars {
            command.env(name, value);
        }
        let output = command.output().expect("spawn isolated constructor case");
        assert!(
            output.status.success(),
            "constructor child {case} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("1 passed; 0 failed; 0 ignored"),
            "constructor child {case} did not run exactly one test\nstdout:\n{stdout}"
        );
        eprintln!("constructor env case {case}: 1 passed");
    }
}

#[test]
#[ignore = "invoked by process-isolated constructor parent"]
fn constructor_env_child() {
    let case = std::env::var(CHILD_CASE).expect("isolated constructor child case marker");
    match case.as_str() {
        "absent" => absent_environment_uses_role_defaults(),
        "empty_family" => empty_family_preserves_triad_compatibility(),
        "conflict_explicit" => explicit_overloads_bypass_conflicting_environment(),
        "invalid_mode" => explicit_overload_bypasses_invalid_mode(),
        "invalid_family" => explicit_overload_bypasses_invalid_family(),
        "errors" => explicit_overloads_preserve_validation_errors(),
        other => panic!("unknown constructor child case {other}"),
    }
}

fn absent_environment_uses_role_defaults() {
    let m1_cfg = small_m1_cfg();
    let m1_model_weights = m1_weights(&m1_cfg, true);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let model = GpuMambaBackbone::new_with_dtype(
            0,
            &m1_model_weights,
            m1_cfg,
            m1_cfg.d_model,
            1,
            dtype,
        )
        .expect("M1 default model");
        assert_route(
            model.ctx(),
            GemmMode::Deterministic,
            BiGemmFamily::Inference,
        );
    }
    let m1_train_weights = m1_weights(&m1_cfg, false);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let trainer =
            MambaTrainer::new_full(0, &m1_train_weights, m1_cfg, session(m1_cfg.d_model), dtype)
                .expect("M1 default trainer");
        assert_route(trainer.ctx(), GemmMode::Deterministic, BiGemmFamily::Triad);
    }

    let m3_cfg = small_m3_cfg();
    let m3_model_weights = m3_weights(&m3_cfg, true);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let model = GpuMamba3Backbone::new_with_dtype(
            0,
            &m3_model_weights,
            m3_cfg,
            m3_cfg.d_model,
            1,
            dtype,
        )
        .expect("M3 default model");
        assert_route(
            model.ctx(),
            GemmMode::Deterministic,
            BiGemmFamily::Inference,
        );
    }
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let weights = m3_weights(&m3_cfg, dtype != WeightDtype::F32);
        let trainer = Mamba3Trainer::new_full(0, &weights, m3_cfg, session(m3_cfg.d_model), dtype)
            .expect("M3 default trainer");
        assert_route(trainer.ctx(), GemmMode::Deterministic, BiGemmFamily::Triad);
    }
}

fn empty_family_preserves_triad_compatibility() {
    let m1_cfg = small_m1_cfg();
    let m1 = GpuMambaBackbone::new(0, &m1_weights(&m1_cfg, true), m1_cfg, m1_cfg.d_model, 1)
        .expect("M1 empty-family model");
    assert_route(m1.ctx(), GemmMode::Deterministic, BiGemmFamily::Triad);

    let m3_cfg = small_m3_cfg();
    let m3 = GpuMamba3Backbone::new(0, &m3_weights(&m3_cfg, true), m3_cfg, m3_cfg.d_model, 1)
        .expect("M3 empty-family model");
    assert_route(m3.ctx(), GemmMode::Deterministic, BiGemmFamily::Triad);
}

fn explicit_overloads_bypass_conflicting_environment() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let m1_cfg = small_m1_cfg();
    let m1_model = m1_weights(&m1_cfg, true);
    let m1_train = m1_weights(&m1_cfg, false);

    let m1_engine = GpuMambaInference::new_with_mode(
        &device,
        &m1_model,
        m1_cfg,
        m1_cfg.d_model,
        1,
        GemmMode::CublasFast,
    )
    .expect("M1 explicit f32 engine");
    assert_route(
        m1_engine.ctx(),
        GemmMode::CublasFast,
        BiGemmFamily::Inference,
    );
    assert_eq!(m1_engine.ctx().state_cap(), 16);

    let m1_mixed = GpuMambaInferenceMixed::new_with_mode(
        &device,
        &m1_model,
        m1_cfg,
        m1_cfg.d_model,
        1,
        WeightDtype::F16,
        GemmMode::CublasPedantic,
    )
    .expect("M1 explicit mixed engine");
    assert_route(
        m1_mixed.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );

    let m1_backbone = GpuMambaBackbone::new_with_mode(
        0,
        &m1_model,
        m1_cfg,
        m1_cfg.d_model,
        1,
        GemmMode::CublasFast,
    )
    .expect("M1 explicit f32 backbone");
    assert_route(
        m1_backbone.ctx(),
        GemmMode::CublasFast,
        BiGemmFamily::Inference,
    );
    let m1_bf16 = GpuMambaBackbone::new_with_dtype_and_mode(
        0,
        &m1_model,
        m1_cfg,
        m1_cfg.d_model,
        1,
        WeightDtype::Bf16,
        GemmMode::CublasPedantic,
    )
    .expect("M1 explicit bf16 backbone");
    assert_route(
        m1_bf16.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );

    let m1_trainer = MambaTrainer::new_full_with_mode(
        0,
        &m1_train,
        m1_cfg,
        session(m1_cfg.d_model),
        WeightDtype::F16,
        GemmMode::CublasFast,
    )
    .expect("M1 explicit f16 trainer");
    assert_route(m1_trainer.ctx(), GemmMode::CublasFast, BiGemmFamily::Triad);
    assert_eq!(m1_trainer.ctx().state_cap(), 16);

    let m3_cfg = small_m3_cfg();
    let m3_model = m3_weights(&m3_cfg, true);
    let m3_engine = Mamba3GpuInferenceEngine::new_with_mode(
        &device,
        &m3_model,
        m3_cfg,
        m3_cfg.d_model,
        1,
        GemmMode::CublasPedantic,
    )
    .expect("M3 explicit f32 engine");
    assert_route(
        m3_engine.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );
    let m3_mixed = Mamba3GpuInferenceMixed::new_with_mode(
        &device,
        &m3_model,
        m3_cfg,
        m3_cfg.d_model,
        1,
        WeightDtype::F16,
        GemmMode::CublasFast,
    )
    .expect("M3 explicit mixed engine");
    assert_route(
        m3_mixed.ctx(),
        GemmMode::CublasFast,
        BiGemmFamily::Inference,
    );
    let m3_backbone = GpuMamba3Backbone::new_with_mode(
        0,
        &m3_model,
        m3_cfg,
        m3_cfg.d_model,
        1,
        GemmMode::CublasFast,
    )
    .expect("M3 explicit f32 backbone");
    assert_route(
        m3_backbone.ctx(),
        GemmMode::CublasFast,
        BiGemmFamily::Inference,
    );
    let m3_bf16 = GpuMamba3Backbone::new_with_dtype_and_mode(
        0,
        &m3_model,
        m3_cfg,
        m3_cfg.d_model,
        1,
        WeightDtype::Bf16,
        GemmMode::CublasPedantic,
    )
    .expect("M3 explicit bf16 backbone");
    assert_route(
        m3_bf16.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );
    let m3_trainer = Mamba3Trainer::new_full_with_mode(
        0,
        &m3_weights(&m3_cfg, true),
        m3_cfg,
        session(m3_cfg.d_model),
        WeightDtype::F16,
        GemmMode::CublasPedantic,
    )
    .expect("M3 explicit f16 trainer");
    assert_route(
        m3_trainer.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Triad,
    );

    #[cfg(feature = "hf")]
    exercise_lm_overloads();

    let conflict = construction_error(
        GpuMambaBackbone::new(0, &m1_model, m1_cfg, m1_cfg.d_model, 1),
        "env-aware construction must reject conflicting selectors",
    );
    assert!(
        conflict.contains("conflicts"),
        "unexpected conflict: {conflict}"
    );
}

#[cfg(feature = "hf")]
fn exercise_lm_overloads() {
    let checkpoint = tempfile::tempdir().expect("synthetic checkpoint dir");
    write_synthetic_checkpoint(checkpoint.path(), false);
    let m1_f32 = GpuMambaLM::from_hf_with_mode(checkpoint.path(), 0, GemmMode::CublasFast)
        .expect("M1 LM f32 explicit");
    assert_route(m1_f32.ctx(), GemmMode::CublasFast, BiGemmFamily::Inference);
    let m1_bf16 = GpuMambaLM::from_hf_with_dtype_and_mode(
        checkpoint.path(),
        0,
        WeightDtype::Bf16,
        GemmMode::CublasPedantic,
    )
    .expect("M1 LM bf16 explicit");
    assert_route(
        m1_bf16.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );
    let m1_f16 = GpuMambaLM::from_hf_with_dtype_batch_and_mode(
        checkpoint.path(),
        0,
        WeightDtype::F16,
        1,
        GemmMode::CublasFast,
    )
    .expect("M1 LM f16 explicit");
    assert_route(m1_f16.ctx(), GemmMode::CublasFast, BiGemmFamily::Inference);

    let cfg = small_m3_cfg();
    let weights = m3_weights(&cfg, true);
    let vocab = 16;
    let embed = vec![0.01; vocab * cfg.d_model];
    let m3_f32 = GpuMamba3LM::from_weights_with_mode(
        &weights,
        cfg,
        embed.clone(),
        None,
        vocab,
        0,
        GemmMode::CublasPedantic,
    )
    .expect("M3 LM f32 explicit");
    assert_route(
        m3_f32.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );
    let m3_bf16 = GpuMamba3LM::build_with_mode(
        Mamba3LmBuild {
            cpu_weights: &weights,
            cfg,
            embed,
            lm_head: None,
            vocab_size: vocab,
            gpu_ordinal: 0,
            dtype: WeightDtype::Bf16,
            batch: 1,
        },
        GemmMode::CublasFast,
    )
    .expect("M3 LM bf16 explicit");
    assert_route(m3_bf16.ctx(), GemmMode::CublasFast, BiGemmFamily::Inference);
}

fn explicit_overload_bypasses_invalid_mode() {
    let cfg = small_m1_cfg();
    let weights = m1_weights(&cfg, true);
    let explicit =
        GpuMambaBackbone::new_with_mode(0, &weights, cfg, cfg.d_model, 1, GemmMode::CublasFast)
            .expect("explicit mode ignores invalid mode env");
    assert_route(
        explicit.ctx(),
        GemmMode::CublasFast,
        BiGemmFamily::Inference,
    );
    let error = construction_error(
        GpuMambaBackbone::new(0, &weights, cfg, cfg.d_model, 1),
        "env lane rejects invalid mode",
    );
    assert!(error.contains("MAMBA_RS_GEMM_MODE"), "{error}");
}

fn explicit_overload_bypasses_invalid_family() {
    let cfg = small_m3_cfg();
    let weights = m3_weights(&cfg, true);
    let explicit = GpuMamba3Backbone::new_with_mode(
        0,
        &weights,
        cfg,
        cfg.d_model,
        1,
        GemmMode::CublasPedantic,
    )
    .expect("explicit mode ignores invalid family env");
    assert_route(
        explicit.ctx(),
        GemmMode::CublasPedantic,
        BiGemmFamily::Inference,
    );
    let error = construction_error(
        GpuMamba3Backbone::new(0, &weights, cfg, cfg.d_model, 1),
        "env lane rejects invalid family",
    );
    assert!(error.contains("MAMBA_RS_BI_GEMM_FAMILY"), "{error}");
}

fn explicit_overloads_preserve_validation_errors() {
    let mut invalid_m1 = small_m1_cfg();
    invalid_m1.d_state = 257;
    let weights = m1_weights(&small_m1_cfg(), true);
    let env_error = construction_error(
        GpuMambaBackbone::new(0, &weights, invalid_m1, invalid_m1.d_model, 1),
        "M1 invalid state capacity",
    );
    let explicit_error = construction_error(
        GpuMambaBackbone::new_with_mode(
            0,
            &weights,
            invalid_m1,
            invalid_m1.d_model,
            1,
            GemmMode::CublasFast,
        ),
        "M1 explicit invalid state capacity",
    );
    assert_eq!(env_error, explicit_error);

    let mut invalid_m3 = small_m3_cfg();
    invalid_m3.d_state = 257;
    let invalid_m3_weights = m3_weights(&small_m3_cfg(), true);
    let env_error = construction_error(
        GpuMamba3Backbone::new(0, &invalid_m3_weights, invalid_m3, invalid_m3.d_model, 1),
        "M3 invalid state capacity",
    );
    let explicit_error = construction_error(
        GpuMamba3Backbone::new_with_mode(
            0,
            &invalid_m3_weights,
            invalid_m3,
            invalid_m3.d_model,
            1,
            GemmMode::CublasPedantic,
        ),
        "M3 explicit invalid state capacity",
    );
    assert_eq!(env_error, explicit_error);

    let cfg = small_m3_cfg();
    let identity = m3_weights(&cfg, true);
    let expected = "f32 M3 trainer requires an explicit input projection";
    let env_error = construction_error(
        Mamba3Trainer::new_full(0, &identity, cfg, session(cfg.d_model), WeightDtype::F32),
        "M3 f32 env projection rejection",
    );
    let explicit_error = construction_error(
        Mamba3Trainer::new_full_with_mode(
            0,
            &identity,
            cfg,
            session(cfg.d_model),
            WeightDtype::F32,
            GemmMode::CublasFast,
        ),
        "M3 f32 explicit projection rejection",
    );
    assert!(env_error.contains(expected), "{env_error}");
    assert_eq!(env_error, explicit_error);
}
