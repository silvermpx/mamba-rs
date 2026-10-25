//! A bf16 or f16 forward's output rows, and an input gradient's, do not
//! depend on how many rows the batch holds: each prefix of a batch gives the
//! bits the whole batch gives for those rows, on every layer shape, with and
//! without a bias, with the tensor cores on and off. The forward runs
//! through the models' entry under the trainer's family and an inference
//! context's; the prefixes cross every row count a tile, a ladder rung or a
//! measured cell is chosen by, the measured Ada cells among them.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::blas::{TypedPtr, gemm_bi_backward_dx_typed, gpu_gemm_typed_forward_raw};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32) * 0.5 - 0.25
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn typed(ctx: &GpuCtx, values: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buffer = DtypedBuf::zeros(&ctx.stream, values.len(), dtype).unwrap();
    buffer.upload_f32(&ctx.stream, values).unwrap();
    buffer
}

fn pointer(buffer: &DtypedBuf, dtype: WeightDtype) -> TypedPtr {
    TypedPtr {
        ptr: buffer.cached_ptr(),
        dtype,
    }
}

/// The forward's output, read back as f32: every half value converts to
/// f32 exactly, so equal f32 bits are equal half bits.
fn forward(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    x: &[f32],
    w: &[f32],
    bias: Option<&GpuBuffer>,
    dims: (usize, usize, usize),
) -> Vec<f32> {
    let (rows, _, n) = dims;
    let x = typed(ctx, x, dtype);
    let w = typed(ctx, w, dtype);
    let y = DtypedBuf::zeros(&ctx.stream, rows * n, dtype).unwrap();
    gpu_gemm_typed_forward_raw(
        ctx,
        pointer(&y, dtype),
        pointer(&x, dtype),
        pointer(&w, dtype),
        bias.map(|bias| bias.cached_ptr()),
        dims,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; rows * n];
    y.download_f32(&ctx.stream, &mut out).unwrap();
    out
}

fn input_gradient(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    dy: &[f32],
    w: &[f32],
    dims: (usize, usize, usize),
) -> Vec<f32> {
    let (rows, k, _) = dims;
    let dy = typed(ctx, dy, dtype);
    let w = typed(ctx, w, dtype);
    let dx = DtypedBuf::zeros(&ctx.stream, rows * k, dtype).unwrap();
    gemm_bi_backward_dx_typed(
        ctx,
        pointer(&dx, dtype),
        pointer(&dy, dtype),
        pointer(&w, dtype),
        dims,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; rows * k];
    dx.download_f32(&ctx.stream, &mut out).unwrap();
    out
}

/// Layers as (inputs, outputs): narrow, shallow, deep and wide, with
/// reductions and widths off every tile edge. (1024, 128) is a deep
/// reduction into 128 outputs, once split between the scalar and the
/// tensor-core tier by the row count; (40, 512) and (16, 2048) are input
/// gradients narrower than a tile; (128, 512), (256, 128), (768, 3072),
/// (1536, 768) and (384, 1928) carry measured Ada cells at 1024, 2048 and
/// 4621 rows.
const LAYERS: [(usize, usize); 17] = [
    (64, 96),
    (96, 40),
    (128, 16),
    (256, 128),
    (256, 2112),
    (384, 1536),
    (512, 256),
    (768, 768),
    (1536, 384),
    (833, 513),
    (1024, 128),
    (40, 512),
    (16, 2048),
    (128, 512),
    (768, 3072),
    (1536, 768),
    (384, 1928),
];
const BATCH: usize = 4700;
const PREFIXES: [usize; 15] = [
    1, 7, 31, 32, 33, 127, 128, 129, 255, 256, 600, 1024, 2048, 4095, 4621,
];
const DTYPES: [WeightDtype; 2] = [WeightDtype::Bf16, WeightDtype::F16];

fn context(device: &GpuDevice) -> GpuCtx {
    let ctx = GpuCtx::new(device).unwrap();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx
}

#[test]
fn half_forward_rows_do_not_depend_on_the_batch() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = context(&device);
    for (family, tensor_cores) in [
        (BiGemmFamily::Triad, true),
        (BiGemmFamily::Triad, false),
        (BiGemmFamily::Inference, true),
    ] {
        ctx.route_controls().set_family(family);
        ctx.route_controls().set_tensor_cores(tensor_cores);
        for dtype in DTYPES {
            for (index, (k, n)) in LAYERS.into_iter().enumerate() {
                let seed = 0x4a1f_0000 + index as u64 * 16;
                let x = synth(BATCH * k, seed);
                let w = synth(k * n, seed + 1);
                let mut bias = GpuBuffer::zeros(&ctx.stream, n).unwrap();
                bias.upload(&ctx.stream, &synth(n, seed + 2)).unwrap();
                for bias in [None, Some(&bias)] {
                    let whole = forward(&ctx, dtype, &x, &w, bias, (BATCH, k, n));
                    for rows in PREFIXES {
                        let prefix = forward(&ctx, dtype, &x[..rows * k], &w, bias, (rows, k, n));
                        assert_eq!(
                            bits(&prefix),
                            bits(&whole[..rows * n]),
                            "{family:?} tensor cores {tensor_cores} {dtype:?} layer {k}x{n}, \
                             bias {}, {rows} rows",
                            bias.is_some(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn half_input_gradient_rows_do_not_depend_on_the_batch() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = context(&device);
    for tensor_cores in [true, false] {
        ctx.route_controls().set_tensor_cores(tensor_cores);
        for dtype in DTYPES {
            for (index, (k, n)) in LAYERS.into_iter().enumerate() {
                let seed = 0x5b2e_0000 + index as u64 * 16;
                let dy = synth(BATCH * n, seed);
                let w = synth(k * n, seed + 1);
                let whole = input_gradient(&ctx, dtype, &dy, &w, (BATCH, k, n));
                for rows in PREFIXES {
                    let prefix = input_gradient(&ctx, dtype, &dy[..rows * n], &w, (rows, k, n));
                    assert_eq!(
                        bits(&prefix),
                        bits(&whole[..rows * k]),
                        "tensor cores {tensor_cores} {dtype:?} layer {k}x{n}, {rows} rows",
                    );
                }
            }
        }
    }
}
