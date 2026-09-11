#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TiedLmDims, gpu_gemm_ex_tied_lm_head_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
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
    let half_rounded_product_bits = match dtype {
        WeightDtype::Bf16 => half::bf16::from_f32(input * input).to_f32().to_bits(),
        WeightDtype::F16 => half::f16::from_f32(input * input).to_f32().to_bits(),
        WeightDtype::F32 => unreachable!(),
    };
    assert_ne!(
        expected_bits, half_rounded_product_bits,
        "fixture must expose a product bit lost by a half-output round trip"
    );
    let temporal = DtypedBuf::zeros(&ctx.stream, BATCH * D_MODEL, dtype)
        .expect("allocate typed temporal owner");
    let embed = DtypedBuf::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL, dtype)
        .expect("allocate typed embedding owner");
    let logits = DtypedBuf::zeros(&ctx.stream, BATCH * VOCAB_PADDED, WeightDtype::F32)
        .expect("allocate F32 logits owner");
    temporal
        .upload_f32(&ctx.stream, &[input; BATCH * D_MODEL])
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

fn assert_irregular_tied_half_f32_output_is_exact_and_guarded(dtype: WeightDtype, precision: i32) {
    const IRREGULAR_D: usize = 37;
    const PREFIX: usize = 5;
    const SUFFIX: usize = 7;
    const GUARD: f32 = -9_876.5;

    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);

    let step = 2.0f32.powi(-precision);
    let temporal_host = (0..BATCH * IRREGULAR_D)
        .map(|index| 1.0 + ((index * 3 + index / IRREGULAR_D) % 7) as f32 * step)
        .collect::<Vec<_>>();
    let embed_host = (0..VOCAB_PADDED * IRREGULAR_D)
        .map(|index| 1.0 + ((index * 5 + index / IRREGULAR_D * 2) % 11) as f32 * step)
        .collect::<Vec<_>>();
    let expected = (0..BATCH)
        .flat_map(|row| {
            let temporal_host = &temporal_host;
            let embed_host = &embed_host;
            (0..VOCAB_PADDED).map(move |column| {
                let mut sum = 0.0f32;
                for reduction in 0..IRREGULAR_D {
                    sum = temporal_host[row * IRREGULAR_D + reduction]
                        .mul_add(embed_host[column * IRREGULAR_D + reduction], sum);
                }
                sum
            })
        })
        .collect::<Vec<_>>();
    let temporal = DtypedBuf::zeros(&ctx.stream, temporal_host.len(), dtype)
        .expect("allocate irregular temporal owner");
    let embed = DtypedBuf::zeros(&ctx.stream, embed_host.len(), dtype)
        .expect("allocate irregular embedding owner");
    temporal
        .upload_f32(&ctx.stream, &temporal_host)
        .expect("upload irregular temporal");
    embed
        .upload_f32(&ctx.stream, &embed_host)
        .expect("upload irregular embedding");
    let initial = vec![GUARD; PREFIX + expected.len() + SUFFIX];
    let mut logits =
        GpuBuffer::from_cpu(&ctx.stream, &initial).expect("allocate guarded F32 logits owner");
    let logits_ptr = logits.raw_ptr_at(&ctx.stream, PREFIX);
    let dims = TiedLmDims {
        batch: BATCH,
        d_model: IRREGULAR_D,
        vocab_padded: VOCAB_PADDED,
    };

    gpu_gemm_ex_tied_lm_head_raw(
        &ctx,
        logits_ptr,
        temporal.cached_ptr(),
        embed.cached_ptr(),
        dtype,
        dims,
    )
    .expect("first irregular tied half-input GEMM");
    let first = logits
        .to_cpu(&ctx.stream)
        .expect("download first guarded logits");
    logits
        .upload(&ctx.stream, &initial)
        .expect("reset guarded logits");
    gpu_gemm_ex_tied_lm_head_raw(
        &ctx,
        logits_ptr,
        temporal.cached_ptr(),
        embed.cached_ptr(),
        dtype,
        dims,
    )
    .expect("repeat irregular tied half-input GEMM");
    let second = logits
        .to_cpu(&ctx.stream)
        .expect("download repeated guarded logits");

    assert_eq!(
        first[PREFIX..PREFIX + expected.len()]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "{dtype:?} irregular tied output"
    );
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "{dtype:?} irregular tied output repeat"
    );
    assert!(first[..PREFIX].iter().all(|&value| value == GUARD));
    assert!(
        first[PREFIX + expected.len()..]
            .iter()
            .all(|&value| value == GUARD)
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_bf16_irregular_inputs_keep_exact_f32_bits_and_guards() {
    assert_irregular_tied_half_f32_output_is_exact_and_guarded(WeightDtype::Bf16, 7);
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_f16_irregular_inputs_keep_exact_f32_bits_and_guards() {
    assert_irregular_tied_half_f32_output_is_exact_and_guarded(WeightDtype::F16, 10);
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_half_zero_reduction_uses_f32_epilogue_without_inputs() {
    const PREFIX: usize = 3;
    const SUFFIX: usize = 4;
    const GUARD: f32 = 4_321.0;
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    let active_len = BATCH * VOCAB_PADDED;
    let initial = vec![GUARD; PREFIX + active_len + SUFFIX];

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let logits = GpuBuffer::from_cpu(&ctx.stream, &initial)
            .expect("allocate guarded zero-reduction logits");
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.raw_ptr_at(&ctx.stream, PREFIX),
            0,
            0,
            dtype,
            TiedLmDims {
                batch: BATCH,
                d_model: 0,
                vocab_padded: VOCAB_PADDED,
            },
        )
        .expect("zero-reduction tied half-input GEMM");
        let actual = logits
            .to_cpu(&ctx.stream)
            .expect("download zero-reduction logits");
        assert!(actual[..PREFIX].iter().all(|&value| value == GUARD));
        assert!(
            actual[PREFIX..PREFIX + active_len]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            actual[PREFIX + active_len..]
                .iter()
                .all(|&value| value == GUARD)
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_tied_half_rejects_null_and_overflow_before_execution() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    let temporal =
        DtypedBuf::zeros(&ctx.stream, BATCH * D_MODEL, WeightDtype::Bf16).expect("temporal owner");
    let embed = DtypedBuf::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL, WeightDtype::Bf16)
        .expect("embedding owner");
    let logits = GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("logits owner");
    let dims = TiedLmDims {
        batch: BATCH,
        d_model: D_MODEL,
        vocab_padded: VOCAB_PADDED,
    };

    assert!(
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            0,
            temporal.cached_ptr(),
            embed.cached_ptr(),
            WeightDtype::Bf16,
            dims,
        )
        .is_err(),
        "null logits must reject"
    );
    assert!(
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            0,
            embed.cached_ptr(),
            WeightDtype::Bf16,
            dims,
        )
        .is_err(),
        "null nonempty temporal must reject"
    );
    assert!(
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            temporal.cached_ptr(),
            0,
            WeightDtype::Bf16,
            dims,
        )
        .is_err(),
        "null nonempty embedding must reject"
    );
    assert!(
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            temporal.cached_ptr(),
            embed.cached_ptr(),
            WeightDtype::Bf16,
            TiedLmDims {
                batch: usize::MAX,
                ..dims
            },
        )
        .is_err(),
        "overflowing tied dimensions must reject"
    );
}
