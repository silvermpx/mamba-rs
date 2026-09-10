#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TiedLmDims, gpu_gemm_ex_tied_lm_head_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

const BATCH: usize = 2;
const D_MODEL: usize = 1;
const VOCAB_PADDED: usize = 96;

fn assert_tied_half_inputs_preserve_f32_product_bit(dtype: WeightDtype, precision: i32) {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);

    let input = 1.0f32 + 2.0f32.powi(-precision);
    let expected_bits = (input * input).to_bits();
    let temporal = DtypedBuf::zeros(&ctx.stream, BATCH * D_MODEL, dtype)
        .expect("allocate typed temporal owner");
    let embed = DtypedBuf::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL, dtype)
        .expect("allocate typed embedding owner");
    let logits = DtypedBuf::zeros(&ctx.stream, BATCH * VOCAB_PADDED, WeightDtype::F32)
        .expect("allocate F32 logits owner");
    temporal
        .upload_f32(&ctx.stream, &vec![input; BATCH * D_MODEL])
        .expect("upload exactly representable temporal input");
    embed
        .upload_f32(&ctx.stream, &vec![input; VOCAB_PADDED * D_MODEL])
        .expect("upload exactly representable embedding input");

    gpu_gemm_ex_tied_lm_head_raw(
        &ctx,
        logits.cached_ptr(),
        temporal.cached_ptr(),
        embed.cached_ptr(),
        dtype,
        TiedLmDims {
            batch: BATCH,
            d_model: D_MODEL,
            vocab_padded: VOCAB_PADDED,
        },
    )
    .expect("deterministic tied half-input GEMM with direct F32 output");

    let mut actual = vec![0.0f32; BATCH * VOCAB_PADDED];
    logits
        .download_f32(&ctx.stream, &mut actual)
        .expect("download F32 logits");
    assert!(
        actual.iter().all(|value| value.to_bits() == expected_bits),
        "{dtype:?} tied logits lost the low product bit: expected {expected_bits:#010x}, got {:?}",
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_bf16_inputs_preserve_true_f32_product_bits() {
    assert_tied_half_inputs_preserve_f32_product_bit(WeightDtype::Bf16, 7);
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_f16_inputs_preserve_true_f32_product_bits() {
    assert_tied_half_inputs_preserve_f32_product_bit(WeightDtype::F16, 10);
}
