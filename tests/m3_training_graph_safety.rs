#![cfg(feature = "cuda")]

use std::any::Any;

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::BiGemmFamily;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::{Mamba3Trainer, TrainSessionCfg};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn deterministic(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn tiny_cfg() -> Mamba3Config {
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

fn captured_triad_trainer(dtype: WeightDtype) -> (Mamba3Trainer, Vec<f32>, Vec<f32>) {
    let cfg = tiny_cfg();
    let batch = 2;
    let seq_len = 64;
    let input_dim = cfg.d_model;
    let n = batch * seq_len * input_dim;
    let cpu = Mamba3Weights::init(&cfg, input_dim, 0x6A50_0000 ^ dtype as u64);
    let input = deterministic(n, 0xA11C_E001);
    let d_temporal = deterministic(n, 0xD7E0_0001);
    let mut trainer = Mamba3Trainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        dtype,
    )
    .expect("construct Mamba3 trainer");

    trainer
        .step(&input, &d_temporal)
        .expect("eager warmup step");
    trainer.ctx().set_batch_invariant(true);
    trainer.ctx().set_bi_gemm_family(BiGemmFamily::Triad);
    trainer.ctx().set_bi_tensor_cores(false);
    trainer.ctx().set_fast_gemm(false);
    trainer
        .capture_graph()
        .expect("capture triad training graph");
    (trainer, input, d_temporal)
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "non-string panic payload".to_owned(),
        },
    }
}

fn assert_route_change_rejected(dtype: WeightDtype, expected_message: &str) {
    let (mut trainer, input, d_temporal) = captured_triad_trainer(dtype);
    trainer.ctx().set_batch_invariant(false);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.step(&input, &d_temporal)
    }))
    .expect_err("route change must panic before graph replay");
    let message = panic_message(panic);
    assert!(
        message.contains(expected_message),
        "unexpected panic message: {message}"
    );
}

#[test]
fn m3_bf16_training_graph_rejects_route_change() {
    assert_route_change_rejected(
        WeightDtype::Bf16,
        "M3 training_graph replay: GEMM route changed since capture",
    );
}

#[test]
fn m3_f16_training_graph_rejects_route_change() {
    assert_route_change_rejected(
        WeightDtype::F16,
        "M3 f16 graph replay: GEMM route changed since capture",
    );
}

#[test]
fn m3_f32_training_graph_rejects_route_change() {
    assert_route_change_rejected(
        WeightDtype::F32,
        "M3 f32 training_graph replay: GEMM route changed since capture",
    );
}

#[test]
fn m3_f16_training_graph_captures_with_presized_triad_scratch() {
    let (mut trainer, input, d_temporal) = captured_triad_trainer(WeightDtype::F16);
    let metrics = trainer
        .step(&input, &d_temporal)
        .expect("replay f16 triad training graph");
    assert!(metrics.graph_replayed, "captured graph must replay");
}

#[test]
fn m3_f16_training_graph_rejects_half_staging_reallocation() {
    let (mut trainer, input, d_temporal) = captured_triad_trainer(WeightDtype::F16);
    trainer
        .ctx()
        .ensure_half_staging(8 * 1024 * 1024)
        .expect("grow half staging");

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.step(&input, &d_temporal)
    }))
    .expect_err("half staging relocation must panic before graph replay");
    let message = panic_message(panic);
    assert!(
        message.contains("M3 f16 graph replay: half_staging pointer changed since capture"),
        "unexpected panic message: {message}"
    );
}

#[test]
fn m3_f16_training_graph_rejects_bi_upcast_scratch_reallocation() {
    let (mut trainer, input, d_temporal) = captured_triad_trainer(WeightDtype::F16);
    let (m, k, n) = (128, 257, 131);
    let gemm_input = DtypedBuf::zeros(&trainer.ctx().stream, m * k, WeightDtype::F16)
        .expect("allocate GEMM input");
    let weight = DtypedBuf::zeros(&trainer.ctx().stream, k * n, WeightDtype::F16)
        .expect("allocate GEMM weight");
    let output = DtypedBuf::zeros(&trainer.ctx().stream, m * n, WeightDtype::F16)
        .expect("allocate GEMM output");

    gpu_gemm_typed_forward_raw(
        trainer.ctx(),
        TypedPtr {
            ptr: output.cached_ptr(),
            dtype: WeightDtype::F16,
        },
        TypedPtr {
            ptr: gemm_input.cached_ptr(),
            dtype: WeightDtype::F16,
        },
        TypedPtr {
            ptr: weight.cached_ptr(),
            dtype: WeightDtype::F16,
        },
        None,
        (128, 257, 131),
    )
    .expect("grow triad upcast scratch");

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.step(&input, &d_temporal)
    }))
    .expect_err("upcast scratch relocation must panic before graph replay");
    let message = panic_message(panic);
    assert!(
        message.contains("M3 f16 graph replay: bi_upcast_scratch pointer changed since capture"),
        "unexpected panic message: {message}"
    );
}
