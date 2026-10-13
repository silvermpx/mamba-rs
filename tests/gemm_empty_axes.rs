//! Every GEMM entry settles an empty axis as BLAS does, in every mode and
//! deterministic family: an empty output writes nothing, an empty reduction
//! leaves the bias (zeros without one) in a written output, and an
//! accumulated weight gradient takes nothing from an empty axis.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::blas::{
    TiedLmDims, TypedPtr, gemm_bi_backward_dw_typed, gemm_bi_backward_dx_typed,
    gemm_bi_forward_raw, gemm_bi_forward_typed, gpu_gemm_bi_backward_dw_grad,
    gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_dx_raw, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_bi_forward_raw, gpu_gemm_bi_tied_lm_head_raw, gpu_gemm_ex_backward_dx_typed,
    gpu_gemm_ex_tied_lm_head_raw, gpu_gemm_forward_dispatch, gpu_gemm_typed_forward_raw,
};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer, GradSlice};
use mamba_rs::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;

const SENTINEL: f32 = 7.0;

/// The contexts every GEMM entry runs under: both deterministic families
/// and both vendor modes.
fn contexts(device: &GpuDevice) -> Vec<(String, GpuCtx)> {
    let mut out = Vec::new();
    for family in [BiGemmFamily::Triad, BiGemmFamily::Inference] {
        let ctx = GpuCtx::new_with_mode(device, GemmMode::Deterministic).unwrap();
        ctx.route_controls().set_family(family);
        out.push((format!("deterministic {family:?}"), ctx));
    }
    for mode in [GemmMode::CublasFast, GemmMode::CublasPedantic] {
        let ctx = GpuCtx::new_with_mode(device, mode).unwrap();
        out.push((format!("{mode:?}"), ctx));
    }
    out
}

fn filled(ctx: &GpuCtx, n: usize, v: f32) -> GpuBuffer {
    GpuBuffer::from_cpu(&ctx.stream, &vec![v; n.max(1)]).unwrap()
}

fn host(ctx: &GpuCtx, b: &GpuBuffer) -> Vec<f32> {
    ctx.stream.synchronize().unwrap();
    b.to_cpu(&ctx.stream).unwrap()
}

fn typed(ctx: &GpuCtx, n: usize, v: f32, dtype: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, n.max(1), dtype).unwrap();
    b.upload_f32(&ctx.stream, &vec![v; n.max(1)]).unwrap();
    b
}

fn typed_host(ctx: &GpuCtx, b: &DtypedBuf) -> Vec<f32> {
    ctx.stream.synchronize().unwrap();
    let mut out = vec![0f32; b.len_elems()];
    b.download_f32(&ctx.stream, &mut out).unwrap();
    out
}

fn tp(b: &DtypedBuf) -> TypedPtr {
    TypedPtr {
        ptr: b.cached_ptr(),
        dtype: b.dtype(),
    }
}

/// The f32 forward and dispatch: an empty batch or output writes nothing,
/// an empty reduction writes the bias in every row, or zeros.
#[test]
fn f32_forward_settles_empty_axes() {
    let device = GpuDevice::new(0).unwrap();
    for (label, ctx) in contexts(&device) {
        let (rows, cols) = (3usize, 5usize);
        let bias_host: Vec<f32> = (0..cols).map(|j| j as f32 + 0.5).collect();
        let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).unwrap();
        let (x, w) = (filled(&ctx, 0, 1.0), filled(&ctx, 0, 1.0));
        for w_dtype in [None, Some(WeightDtype::Bf16)] {
            let run = |y: &mut GpuBuffer, b: Option<u64>, dims| match w_dtype {
                None => gpu_gemm_bi_forward_raw(&ctx, y, &x, w.cached_ptr(), b, dims),
                Some(dt) => gpu_gemm_forward_dispatch(&ctx, y, &x, w.cached_ptr(), dt, b, dims),
            };
            // An empty reduction: the bias, then zeros.
            let mut y = filled(&ctx, rows * cols, SENTINEL);
            run(&mut y, Some(bias.cached_ptr()), (rows, 0, cols)).unwrap();
            let want: Vec<f32> = (0..rows).flat_map(|_| bias_host.clone()).collect();
            assert_eq!(host(&ctx, &y), want, "{label} {w_dtype:?} K=0 with bias");
            run(&mut y, None, (rows, 0, cols)).unwrap();
            assert_eq!(
                host(&ctx, &y),
                vec![0.0; rows * cols],
                "{label} {w_dtype:?} K=0"
            );
            // An empty output: nothing written.
            for dims in [(0, 4, cols), (rows, 4, 0)] {
                let mut y = filled(&ctx, 1, SENTINEL);
                run(&mut y, Some(bias.cached_ptr()), dims).unwrap();
                assert_eq!(
                    host(&ctx, &y),
                    vec![SENTINEL],
                    "{label} {w_dtype:?} {dims:?}"
                );
            }
        }
    }
}

/// The f32 backward: dX of an empty reduction is zeros, of an empty
/// output nothing; dW and the bias gradient accumulate nothing over an
/// empty axis.
#[test]
fn f32_backward_settles_empty_axes() {
    let device = GpuDevice::new(0).unwrap();
    for (label, ctx) in contexts(&device) {
        let (rows, n_in) = (3usize, 4usize);
        let (dy, x, w) = (
            filled(&ctx, 0, 1.0),
            filled(&ctx, 0, 1.0),
            filled(&ctx, 0, 1.0),
        );
        let mut dx = filled(&ctx, rows * n_in, SENTINEL);
        gpu_gemm_bi_backward_dx_raw(&ctx, &mut dx, &dy, w.cached_ptr(), rows, n_in, 0).unwrap();
        assert_eq!(
            host(&ctx, &dx),
            vec![0.0; rows * n_in],
            "{label} dX over N=0"
        );
        for (b, k, n) in [(0, n_in, 2), (rows, 0, 2)] {
            let mut dx = filled(&ctx, 1, SENTINEL);
            gpu_gemm_bi_backward_dx_raw(&ctx, &mut dx, &dy, w.cached_ptr(), b, k, n).unwrap();
            assert_eq!(host(&ctx, &dx), vec![SENTINEL], "{label} dX {b}x{k}x{n}");
        }
        let dw_buf = filled(&ctx, n_in * 2, SENTINEL);
        let db_buf = filled(&ctx, 2, SENTINEL);
        let (dw, db) = (
            GradSlice::from_raw(dw_buf.cached_ptr(), dw_buf.len()),
            GradSlice::from_raw(db_buf.cached_ptr(), db_buf.len()),
        );
        for (b, k, n) in [(0, n_in, 2), (rows, 0, 2), (rows, n_in, 0)] {
            gpu_gemm_bi_backward_dw_grad(&ctx, &dw, &dy, &x, b, k, n).unwrap();
        }
        let mut dx = filled(&ctx, 1, SENTINEL);
        gpu_gemm_bi_backward_grad_raw(
            &ctx,
            &mut dx,
            (&dw, Some(&db)),
            &dy,
            &x,
            w.cached_ptr(),
            (0, n_in, 2),
        )
        .unwrap();
        assert_eq!(host(&ctx, &dw_buf), vec![SENTINEL; n_in * 2], "{label} dW");
        assert_eq!(host(&ctx, &db_buf), vec![SENTINEL; 2], "{label} db");
    }
}

/// The typed entries: the forward writes the bias in the output's storage
/// over an empty reduction, dX zeros, dW nothing; an empty output writes
/// nothing.
#[test]
fn typed_entries_settle_empty_axes() {
    let device = GpuDevice::new(0).unwrap();
    for (label, ctx) in contexts(&device) {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let (rows, cols) = (3usize, 5usize);
            let bias_host: Vec<f32> = (0..cols).map(|j| j as f32 + 0.5).collect();
            let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).unwrap();
            let (x, w) = (typed(&ctx, 0, 1.0, dtype), typed(&ctx, 0, 1.0, dtype));
            let want: Vec<f32> = (0..rows).flat_map(|_| bias_host.clone()).collect();
            let y = typed(&ctx, rows * cols, SENTINEL, dtype);
            gpu_gemm_typed_forward_raw(
                &ctx,
                tp(&y),
                tp(&x),
                tp(&w),
                Some(bias.cached_ptr()),
                (rows, 0, cols),
            )
            .unwrap();
            assert_eq!(
                typed_host(&ctx, &y),
                want,
                "{label} {dtype:?} typed forward"
            );
            if ctx.gemm_mode() == GemmMode::Deterministic {
                let y = typed(&ctx, rows * cols, SENTINEL, dtype);
                gemm_bi_forward_typed(&ctx, tp(&y), tp(&x), tp(&w), 0, (rows, 0, cols)).unwrap();
                assert_eq!(
                    typed_host(&ctx, &y),
                    vec![0.0; rows * cols],
                    "{label} {dtype:?}"
                );
                let y = typed(&ctx, rows * cols, SENTINEL, dtype);
                gemm_bi_forward_raw(&ctx, tp(&y), tp(&x), tp(&w), None, (rows, 0, cols)).unwrap();
                assert_eq!(
                    typed_host(&ctx, &y),
                    vec![0.0; rows * cols],
                    "{label} {dtype:?}"
                );
            }
            // dX: zeros over an empty reduction, nothing over an empty output.
            let dx = typed(&ctx, rows * 4, SENTINEL, dtype);
            let dy = typed(&ctx, 0, 1.0, dtype);
            gpu_gemm_ex_backward_dx_typed(&ctx, tp(&dx), tp(&dy), tp(&w), rows, 4, 0).unwrap();
            assert_eq!(
                typed_host(&ctx, &dx),
                vec![0.0; rows * 4],
                "{label} {dtype:?} dX"
            );
            let dx = typed(&ctx, 1, SENTINEL, dtype);
            gpu_gemm_ex_backward_dx_typed(&ctx, tp(&dx), tp(&dy), tp(&w), 0, 4, 2).unwrap();
            assert_eq!(
                typed_host(&ctx, &dx),
                vec![SENTINEL],
                "{label} {dtype:?} dX M=0"
            );
            // dW accumulates nothing.
            let dw_buf = filled(&ctx, 8, SENTINEL);
            let dw = GradSlice::from_raw(dw_buf.cached_ptr(), dw_buf.len());
            gpu_gemm_bi_backward_dw_grad_typed(&ctx, &dw, tp(&dy), tp(&x), 0, 4, 2).unwrap();
            if ctx.gemm_mode() == GemmMode::Deterministic {
                gemm_bi_backward_dw_typed(&ctx, dw.ptr(), tp(&dy), tp(&x), (rows, 0, 2)).unwrap();
                let dx = typed(&ctx, rows * 4, SENTINEL, dtype);
                gemm_bi_backward_dx_typed(&ctx, tp(&dx), tp(&dy), tp(&w), (rows, 4, 0)).unwrap();
                assert_eq!(
                    typed_host(&ctx, &dx),
                    vec![0.0; rows * 4],
                    "{label} {dtype:?} dX"
                );
            }
            assert_eq!(
                host(&ctx, &dw_buf),
                vec![SENTINEL; 8],
                "{label} {dtype:?} dW"
            );
        }
    }
}

/// The tied heads: zeros for a model width of zero, nothing for an empty
/// batch or vocabulary.
#[test]
fn tied_heads_settle_empty_axes() {
    let device = GpuDevice::new(0).unwrap();
    for (label, ctx) in contexts(&device) {
        let (batch, vocab) = (2usize, 64usize);
        let (t, e) = (filled(&ctx, 0, 1.0), filled(&ctx, 0, 1.0));
        let logits = filled(&ctx, batch * vocab, SENTINEL);
        gpu_gemm_bi_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            t.cached_ptr(),
            e.cached_ptr(),
            batch,
            0,
            vocab,
        )
        .unwrap();
        assert_eq!(
            host(&ctx, &logits),
            vec![0.0; batch * vocab],
            "{label} f32 head"
        );
        let logits = filled(&ctx, batch * vocab, SENTINEL);
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            t.cached_ptr(),
            e.cached_ptr(),
            WeightDtype::Bf16,
            TiedLmDims {
                batch,
                d_model: 0,
                vocab_padded: vocab,
            },
        )
        .unwrap();
        assert_eq!(
            host(&ctx, &logits),
            vec![0.0; batch * vocab],
            "{label} bf16 head"
        );
        for (b, v) in [(0, vocab), (batch, 0)] {
            let logits = filled(&ctx, 1, SENTINEL);
            gpu_gemm_bi_tied_lm_head_raw(
                &ctx,
                logits.cached_ptr(),
                t.cached_ptr(),
                e.cached_ptr(),
                b,
                8,
                v,
            )
            .unwrap();
            assert_eq!(host(&ctx, &logits), vec![SENTINEL], "{label} {b}x{v}");
        }
    }
}
