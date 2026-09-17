#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_bi_forward_ptr, gpu_gemm_bi_forward_raw, gpu_gemm_bi_tied_lm_head_raw,
    gpu_gemm_typed_forward_raw,
};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

const BATCH: usize = 3;
const D_MODEL: usize = 37;
const VOCAB_PADDED: usize = 96;

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn nn_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = (0..BATCH * D_MODEL)
        .map(|index| (index as i32 % 7 - 3) as f32)
        .collect::<Vec<_>>();
    let w = (0..D_MODEL * VOCAB_PADDED)
        .map(|index| ((index * 3 + (index / VOCAB_PADDED) * 2) as i32 % 11 - 5) as f32)
        .collect::<Vec<_>>();
    let bias = (0..VOCAB_PADDED)
        .map(|column| (column as i32 % 3 - 1) as f32)
        .collect::<Vec<_>>();
    let expected = (0..BATCH)
        .flat_map(|row| {
            let x = &x;
            let w = &w;
            let bias = &bias;
            (0..VOCAB_PADDED).map(move |column| {
                let mut value = bias[column];
                for reduction in 0..D_MODEL {
                    value += x[row * D_MODEL + reduction] * w[reduction * VOCAB_PADDED + column];
                }
                value
            })
        })
        .collect();
    (x, w, bias, expected)
}

fn nt_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = (0..BATCH * D_MODEL)
        .map(|index| (index as i32 % 5 - 2) as f32)
        .collect::<Vec<_>>();
    let embedding = (0..VOCAB_PADDED * D_MODEL)
        .map(|index| ((index * 3 + index / D_MODEL) as i32 % 7 - 3) as f32)
        .collect::<Vec<_>>();
    let expected = (0..BATCH)
        .flat_map(|row| {
            let hidden = &hidden;
            let embedding = &embedding;
            (0..VOCAB_PADDED).map(move |column| {
                let mut value = 0.0f32;
                for reduction in 0..D_MODEL {
                    value +=
                        hidden[row * D_MODEL + reduction] * embedding[column * D_MODEL + reduction];
                }
                value
            })
        })
        .collect();
    (hidden, embedding, expected)
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_triad_public_forward_ptr_records_nn_route() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.route_controls().set_family(BiGemmFamily::Triad);

    let x = GpuBuffer::zeros(&ctx.stream, BATCH * D_MODEL).expect("allocate X[3,37]");
    let w = GpuBuffer::zeros(&ctx.stream, D_MODEL * VOCAB_PADDED).expect("allocate W[37,96]");
    let mut y = GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("allocate Y[3,96]");

    let trace = ctx
        .record_eager_gemm_trace(|| {
            gpu_gemm_bi_forward_ptr(
                &ctx,
                &mut y,
                x.cached_ptr(),
                w.cached_ptr(),
                None,
                (BATCH, D_MODEL, VOCAB_PADDED),
            )
        })
        .expect("deterministic Triad raw NN forward");

    assert!(!trace.routes().is_empty(), "raw NN recorded no route");
    assert!(
        trace.routes().iter().all(|route| {
            route.op == ResolvedGemmOp::Nn
                && route.shape == (BATCH, D_MODEL, VOCAB_PADDED)
                && route.strides == (D_MODEL, VOCAB_PADDED, VOCAB_PADDED)
        }),
        "raw NN recorded unexpected routes: {:?}",
        trace.routes()
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_triad_tied_f32_raw_records_nt_route_and_strides() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.route_controls().set_family(BiGemmFamily::Triad);

    let hidden = GpuBuffer::zeros(&ctx.stream, BATCH * D_MODEL).expect("allocate hidden[3,37]");
    let embedding =
        GpuBuffer::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL).expect("allocate embedding[96,37]");
    let logits =
        GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("allocate logits[3,96]");

    let trace = ctx
        .record_eager_gemm_trace(|| {
            gpu_gemm_bi_tied_lm_head_raw(
                &ctx,
                logits.cached_ptr(),
                hidden.cached_ptr(),
                embedding.cached_ptr(),
                BATCH,
                D_MODEL,
                VOCAB_PADDED,
            )
        })
        .expect("deterministic Triad tied F32 NT forward");

    assert!(!trace.routes().is_empty(), "tied F32 NT recorded no route");
    assert!(
        trace.routes().iter().all(|route| {
            route.op == ResolvedGemmOp::Nt
                && route.shape == (BATCH, VOCAB_PADDED, D_MODEL)
                && route.strides == (D_MODEL, D_MODEL, VOCAB_PADDED)
        }),
        "tied F32 NT recorded unexpected routes: {:?}",
        trace.routes()
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn f32_nn_buffer_and_interior_pointer_wrappers_match_cpu_and_repeat_for_both_families() {
    const INPUT_PREFIX: usize = 5;
    const INPUT_SUFFIX: usize = 7;
    let (x_host, w_host, bias_host, expected) = nn_fixture();
    let expected_bits = f32_bits(&expected);

    for family in [BiGemmFamily::Triad, BiGemmFamily::Inference] {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic)
            .expect("select deterministic GEMM mode");
        ctx.route_controls().set_family(family);

        let x = GpuBuffer::from_cpu(&ctx.stream, &x_host).expect("upload contiguous NN input");
        let mut guarded_x = vec![101.0f32; INPUT_PREFIX + x_host.len() + INPUT_SUFFIX];
        guarded_x[INPUT_PREFIX..INPUT_PREFIX + x_host.len()].copy_from_slice(&x_host);
        let guarded_x =
            GpuBuffer::from_cpu(&ctx.stream, &guarded_x).expect("upload guarded NN input");
        let w = GpuBuffer::from_cpu(&ctx.stream, &w_host).expect("upload NN weights");
        let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("upload NN bias");
        let mut buffered = GpuBuffer::zeros(&ctx.stream, expected.len()).expect("NN output");
        let mut borrowed = GpuBuffer::zeros(&ctx.stream, expected.len()).expect("NN output");

        gpu_gemm_bi_forward_raw(
            &ctx,
            &mut buffered,
            &x,
            w.cached_ptr(),
            Some(bias.cached_ptr()),
            (BATCH, D_MODEL, VOCAB_PADDED),
        )
        .expect("buffer-taking NN wrapper");
        let trace = ctx
            .record_eager_gemm_trace(|| {
                gpu_gemm_bi_forward_ptr(
                    &ctx,
                    &mut borrowed,
                    guarded_x.raw_ptr_at(&ctx.stream, INPUT_PREFIX),
                    w.cached_ptr(),
                    Some(bias.cached_ptr()),
                    (BATCH, D_MODEL, VOCAB_PADDED),
                )
            })
            .expect("borrowed-pointer NN wrapper");
        let borrowed_once = f32_bits(
            &borrowed
                .to_cpu(&ctx.stream)
                .expect("download first borrowed NN output"),
        );
        gpu_gemm_bi_forward_ptr(
            &ctx,
            &mut borrowed,
            guarded_x.raw_ptr_at(&ctx.stream, INPUT_PREFIX),
            w.cached_ptr(),
            Some(bias.cached_ptr()),
            (BATCH, D_MODEL, VOCAB_PADDED),
        )
        .expect("repeat borrowed-pointer NN wrapper");

        let buffered_bits = f32_bits(
            &buffered
                .to_cpu(&ctx.stream)
                .expect("download buffered NN output"),
        );
        let borrowed_twice = f32_bits(
            &borrowed
                .to_cpu(&ctx.stream)
                .expect("download repeated borrowed NN output"),
        );
        assert_eq!(buffered_bits, expected_bits, "{family:?} buffered NN");
        assert_eq!(borrowed_once, expected_bits, "{family:?} borrowed NN");
        assert_eq!(borrowed_twice, borrowed_once, "{family:?} repeated NN");
        if family == BiGemmFamily::Triad {
            assert!(!trace.routes().is_empty(), "Triad NN recorded no route");
            assert!(
                trace
                    .routes()
                    .iter()
                    .all(|route| route.op == ResolvedGemmOp::Nn),
                "Triad NN recorded unexpected routes: {:?}",
                trace.routes()
            );
        }
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn tied_f32_guarded_output_matches_cpu_and_repeats_with_nt_routes_for_both_families() {
    const OUTPUT_PREFIX: usize = 5;
    const OUTPUT_SUFFIX: usize = 7;
    const GUARD: f32 = -12_345.0;
    let (hidden_host, embedding_host, expected) = nt_fixture();
    let expected_bits = f32_bits(&expected);

    for family in [BiGemmFamily::Triad, BiGemmFamily::Inference] {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic)
            .expect("select deterministic GEMM mode");
        ctx.route_controls().set_family(family);

        let hidden =
            GpuBuffer::from_cpu(&ctx.stream, &hidden_host).expect("upload tied hidden input");
        let embedding =
            GpuBuffer::from_cpu(&ctx.stream, &embedding_host).expect("upload tied embedding input");
        let initial = vec![GUARD; OUTPUT_PREFIX + expected.len() + OUTPUT_SUFFIX];
        let mut guarded_output =
            GpuBuffer::from_cpu(&ctx.stream, &initial).expect("upload guarded tied output");
        let output_ptr = guarded_output.raw_ptr_at(&ctx.stream, OUTPUT_PREFIX);

        let trace = ctx
            .record_eager_gemm_trace(|| {
                gpu_gemm_bi_tied_lm_head_raw(
                    &ctx,
                    output_ptr,
                    hidden.cached_ptr(),
                    embedding.cached_ptr(),
                    BATCH,
                    D_MODEL,
                    VOCAB_PADDED,
                )
            })
            .expect("first tied NT wrapper call");
        let first = guarded_output
            .to_cpu(&ctx.stream)
            .expect("download first tied output");
        guarded_output
            .upload(&ctx.stream, &initial)
            .expect("reset guarded tied output");
        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            output_ptr,
            hidden.cached_ptr(),
            embedding.cached_ptr(),
            BATCH,
            D_MODEL,
            VOCAB_PADDED,
        )
        .expect("repeat tied NT wrapper call");
        let second = guarded_output
            .to_cpu(&ctx.stream)
            .expect("download repeated tied output");

        assert_eq!(
            f32_bits(&first[OUTPUT_PREFIX..OUTPUT_PREFIX + expected.len()]),
            expected_bits,
            "{family:?} tied NT active output"
        );
        assert_eq!(
            f32_bits(&second),
            f32_bits(&first),
            "{family:?} tied repeat"
        );
        assert!(first[..OUTPUT_PREFIX].iter().all(|&value| value == GUARD));
        assert!(
            first[OUTPUT_PREFIX + expected.len()..]
                .iter()
                .all(|&value| value == GUARD)
        );
        assert!(!trace.routes().is_empty(), "{family:?} tied NT route");
        assert!(
            trace.routes().iter().all(|route| {
                route.op == ResolvedGemmOp::Nt
                    && route.shape == (BATCH, VOCAB_PADDED, D_MODEL)
                    && route.strides == (D_MODEL, D_MODEL, VOCAB_PADDED)
            }),
            "{family:?} tied NT recorded unexpected routes: {:?}",
            trace.routes()
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn tied_f32_zero_reduction_zeros_only_the_guarded_output_subspan_for_both_families() {
    const PREFIX: usize = 3;
    const SUFFIX: usize = 4;
    const GUARD: f32 = 9_876.0;
    let active_len = BATCH * VOCAB_PADDED;

    for family in [BiGemmFamily::Triad, BiGemmFamily::Inference] {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic)
            .expect("select deterministic GEMM mode");
        ctx.route_controls().set_family(family);
        let initial = vec![GUARD; PREFIX + active_len + SUFFIX];
        let output = GpuBuffer::from_cpu(&ctx.stream, &initial).expect("guarded zero-K output");

        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            output.raw_ptr_at(&ctx.stream, PREFIX),
            0,
            0,
            BATCH,
            0,
            VOCAB_PADDED,
        )
        .expect("zero-reduction tied NT");
        let actual = output.to_cpu(&ctx.stream).expect("download zero-K output");
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
fn all_f32_typed_nn_zero_reduction_applies_bias_once_inside_output_guards() {
    const PREFIX: usize = 3;
    const SUFFIX: usize = 4;
    const GUARD: f32 = -7_654.0;
    let bias_host = (0..VOCAB_PADDED)
        .map(|column| (column as i32 % 9 - 4) as f32)
        .collect::<Vec<_>>();
    let expected = (0..BATCH)
        .flat_map(|_| bias_host.iter().copied())
        .collect::<Vec<_>>();

    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    let initial = vec![GUARD; PREFIX + expected.len() + SUFFIX];
    let output = GpuBuffer::from_cpu(&ctx.stream, &initial).expect("guarded typed NN output");
    let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("typed NN bias");

    gpu_gemm_typed_forward_raw(
        &ctx,
        TypedPtr {
            ptr: output.raw_ptr_at(&ctx.stream, PREFIX),
            dtype: mamba_rs::mamba_ssm::gpu::WeightDtype::F32,
        },
        TypedPtr {
            ptr: 0,
            dtype: mamba_rs::mamba_ssm::gpu::WeightDtype::F32,
        },
        TypedPtr {
            ptr: 0,
            dtype: mamba_rs::mamba_ssm::gpu::WeightDtype::F32,
        },
        Some(bias.cached_ptr()),
        (BATCH, 0, VOCAB_PADDED),
    )
    .expect("all-F32 typed zero-reduction NN");
    let actual = output
        .to_cpu(&ctx.stream)
        .expect("download typed zero-K NN output");
    assert!(actual[..PREFIX].iter().all(|&value| value == GUARD));
    assert_eq!(
        f32_bits(&actual[PREFIX..PREFIX + expected.len()]),
        f32_bits(&expected),
        "zero-reduction NN bias must be applied exactly once"
    );
    assert!(
        actual[PREFIX + expected.len()..]
            .iter()
            .all(|&value| value == GUARD)
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn tied_f32_raw_rejects_null_and_overflowing_spans_before_launch() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    let hidden = GpuBuffer::zeros(&ctx.stream, BATCH * D_MODEL).expect("hidden owner");
    let embedding = GpuBuffer::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL).expect("embedding owner");
    let output = GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("output owner");

    assert!(
        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            0,
            hidden.cached_ptr(),
            embedding.cached_ptr(),
            BATCH,
            D_MODEL,
            VOCAB_PADDED,
        )
        .is_err(),
        "null output must reject"
    );
    assert!(
        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            output.cached_ptr(),
            0,
            embedding.cached_ptr(),
            BATCH,
            D_MODEL,
            VOCAB_PADDED,
        )
        .is_err(),
        "null nonempty hidden must reject"
    );
    assert!(
        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            output.cached_ptr(),
            hidden.cached_ptr(),
            embedding.cached_ptr(),
            usize::MAX,
            D_MODEL,
            VOCAB_PADDED,
        )
        .is_err(),
        "overflowing output span must reject"
    );
}
