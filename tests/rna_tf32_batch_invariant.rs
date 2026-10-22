//! Under the strict TF32 policy a forward's output rows do not depend on
//! how many rows the batch holds: each prefix of a batch gives the bits the
//! whole batch gives for those rows, on every layer shape, with and without
//! a bias. The prefixes run from one row to the whole batch, so every tile
//! the architecture's table picks meets every other one row for row. The products really run on TF32 (they differ from the exact
//! policy's), and the gradients run under the policy without falling to the
//! exact family.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::blas::{
    gpu_gemm_f32_backward_dw_ptrs, gpu_gemm_f32_backward_dx_ptrs, gpu_gemm_f32_forward_ptrs,
};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::{F32TriadPolicy, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;

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

fn upload(ctx: &GpuCtx, values: &[f32]) -> GpuBuffer {
    let mut buffer = GpuBuffer::zeros(&ctx.stream, values.len()).unwrap();
    buffer.upload(&ctx.stream, values).unwrap();
    buffer
}

fn forward(
    ctx: &GpuCtx,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    dims: (usize, usize, usize),
) -> Vec<f32> {
    let (rows, _, n) = dims;
    let x = upload(ctx, x);
    let w = upload(ctx, w);
    let bias = bias.map(|bias| upload(ctx, bias));
    let y = GpuBuffer::zeros(&ctx.stream, rows * n).unwrap();
    unsafe {
        gpu_gemm_f32_forward_ptrs(
            ctx,
            y.cached_ptr(),
            x.cached_ptr(),
            w.cached_ptr(),
            bias.as_ref().map(|bias| bias.cached_ptr()),
            dims,
        )
        .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    y.to_cpu(&ctx.stream).unwrap()
}

/// Layers whose rows, at the batch sizes below, take every tile of an
/// architecture's forward table: the narrow and 64 x 64 portable tiles, the
/// multistage tiles of a few thousand rows, the wide tiles of tall batches.
const LAYERS: [(usize, usize); 9] = [
    (64, 96),
    (96, 40),
    (128, 64),
    (256, 128),
    (256, 256),
    (256, 2112),
    (384, 1536),
    (512, 256),
    (768, 768),
];
const BATCH: usize = 8192;
const PREFIXES: [usize; 10] = [1, 7, 255, 256, 600, 1024, 2048, 4096, 5000, 8191];

#[test]
fn strict_tf32_forward_rows_do_not_depend_on_the_batch() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = GpuCtx::new(&device).unwrap();
    for (index, (k, n)) in LAYERS.into_iter().enumerate() {
        let seed = 0x7f32_0000 + index as u64 * 16;
        let x = synth(BATCH * k, seed);
        let w = synth(k * n, seed + 1);
        let bias = synth(n, seed + 2);
        for bias in [None, Some(bias.as_slice())] {
            ctx.route_controls()
                .set_f32_policy(F32TriadPolicy::RnaTf32BatchInvariant);
            let whole = forward(&ctx, &x, &w, bias, (BATCH, k, n));
            for rows in PREFIXES {
                let prefix = forward(&ctx, &x[..rows * k], &w, bias, (rows, k, n));
                assert_eq!(
                    bits(&prefix),
                    bits(&whole[..rows * n]),
                    "layer {k}x{n}, bias {}, {rows} rows",
                    bias.is_some(),
                );
            }
            ctx.route_controls()
                .set_f32_policy(F32TriadPolicy::ExactScalarFma);
            let exact = forward(&ctx, &x, &w, bias, (BATCH, k, n));
            assert_ne!(
                bits(&exact),
                bits(&whole),
                "layer {k}x{n}: the strict policy ran no TF32 product",
            );
        }
    }
}

#[test]
fn strict_tf32_gradients_run_under_the_policy() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = GpuCtx::new(&device).unwrap();
    ctx.route_controls()
        .set_f32_policy(F32TriadPolicy::RnaTf32BatchInvariant);
    for (index, (k, n)) in LAYERS.into_iter().enumerate() {
        for rows in [7, 128, BATCH] {
            let seed = 0x9d0_0000 + index as u64 * 16 + rows as u64;
            let x = upload(&ctx, &synth(rows * k, seed));
            let w = upload(&ctx, &synth(k * n, seed + 1));
            let dy = upload(&ctx, &synth(rows * n, seed + 2));
            let dw = upload(&ctx, &synth(k * n, seed + 3));
            let dx = GpuBuffer::zeros(&ctx.stream, rows * k).unwrap();
            let dims = (rows, k, n);
            let run = || unsafe {
                gpu_gemm_f32_backward_dx_ptrs(
                    &ctx,
                    dx.cached_ptr(),
                    dy.cached_ptr(),
                    w.cached_ptr(),
                    dims,
                )?;
                gpu_gemm_f32_backward_dw_ptrs(
                    &ctx,
                    dw.cached_ptr(),
                    dy.cached_ptr(),
                    x.cached_ptr(),
                    dims,
                )
            };
            run().unwrap_or_else(|error| panic!("layer {k}x{n}, {rows} rows: {error}"));
            ctx.stream.synchronize().unwrap();
            let values = dx.to_cpu(&ctx.stream).unwrap();
            assert!(
                values.iter().all(|v| v.is_finite()),
                "layer {k}x{n}, {rows} rows"
            );
        }
    }
}
