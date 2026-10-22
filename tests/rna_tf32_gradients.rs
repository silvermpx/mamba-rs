//! Under the strict TF32 policy the gradients take the routes the shape
//! rules pick - the eight-way weight-gradient split, the Ada's pre-rounded
//! weight-gradient tiles, the four-way input-gradient split, the Ada's
//! multistage input-gradient tiles and the narrow tiles - on ragged shapes as
//! well as whole ones. Every product stays
//! within TF32's rounding of an f64 reference, and two runs of one product
//! write the same bits.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::blas::{gpu_gemm_f32_backward_dw_ptrs, gpu_gemm_f32_backward_dx_ptrs};
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

/// Each output against its f64 sum: TF32 keeps ten mantissa bits of each
/// operand, so an error past a few units of 2^-11 of the absolute products
/// is a wrong product, not rounding.
fn assert_close(label: &str, got: &[f32], want: &[f64], magnitude: &[f64]) {
    for (index, ((g, w), m)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let bound = m * 2f64.powi(-9) + 1e-6;
        assert!(
            (f64::from(*g) - w).abs() <= bound,
            "{label} [{index}]: {g} against {w} (bound {bound})"
        );
    }
}

fn weight_gradient(
    ctx: &GpuCtx,
    dims: (usize, usize, usize),
    dy: &[f32],
    x: &[f32],
    dw0: &[f32],
) -> Vec<f32> {
    let dy = upload(ctx, dy);
    let x = upload(ctx, x);
    let dw = upload(ctx, dw0);
    unsafe {
        gpu_gemm_f32_backward_dw_ptrs(ctx, dw.cached_ptr(), dy.cached_ptr(), x.cached_ptr(), dims)
            .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    dw.to_cpu(&ctx.stream).unwrap()
}

fn input_gradient(ctx: &GpuCtx, dims: (usize, usize, usize), dy: &[f32], w: &[f32]) -> Vec<f32> {
    let (m, k, _) = dims;
    let dy = upload(ctx, dy);
    let w = upload(ctx, w);
    let dx = GpuBuffer::zeros(&ctx.stream, m * k).unwrap();
    unsafe {
        gpu_gemm_f32_backward_dx_ptrs(ctx, dx.cached_ptr(), dy.cached_ptr(), w.cached_ptr(), dims)
            .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    dx.to_cpu(&ctx.stream).unwrap()
}

#[test]
fn strict_tf32_gradients_are_tf32_products_and_repeat_their_bits() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = GpuCtx::new(&device).unwrap();
    ctx.route_controls()
        .set_f32_policy(F32TriadPolicy::RnaTf32BatchInvariant);

    // (batch, inputs, outputs): the eight-way split (deep batches, one of
    // them ragged in every axis), the pre-rounded wide and square tiles,
    // the narrow tile, and a ragged output no pre-rounded tile stages.
    for (index, dims) in [
        (4096, 128, 796),
        (2048, 512, 256),
        (3001, 385, 772),
        (1024, 384, 1536),
        (1024, 833, 384),
        (64, 256, 1024),
        (513, 64, 100),
        (1027, 95, 131),
    ]
    .into_iter()
    .enumerate()
    {
        let (m, k, n) = dims;
        let seed = 0x7e57_0000 + index as u64 * 8;
        let dy = synth(m * n, seed);
        let x = synth(m * k, seed + 1);
        let dw0 = synth(k * n, seed + 2);
        let first = weight_gradient(&ctx, dims, &dy, &x, &dw0);
        let second = weight_gradient(&ctx, dims, &dy, &x, &dw0);
        assert_eq!(bits(&first), bits(&second), "dW {dims:?} repeats");
        let mut want = vec![0f64; k * n];
        let mut magnitude = vec![0f64; k * n];
        for row in 0..m {
            for i in 0..k {
                let a = f64::from(x[row * k + i]);
                for j in 0..n {
                    let product = a * f64::from(dy[row * n + j]);
                    want[i * n + j] += product;
                    magnitude[i * n + j] += product.abs();
                }
            }
        }
        for (slot, start) in want.iter_mut().zip(&dw0) {
            *slot += f64::from(*start);
        }
        assert_close(&format!("dW {dims:?}"), &first, &want, &magnitude);
    }

    // (batch, inputs, outputs): the four-way split (whole and ragged), the
    // Ada's multistage tiles (both sizes and a ragged output), the narrow
    // tile past its column bound and under its reduction floor, and the
    // 64 x 64 tile.
    for (index, dims) in [
        (64, 384, 1536),
        (1024, 1536, 384),
        (4096, 256, 2112),
        (777, 300, 260),
        (100, 250, 300),
        (64, 1536, 384),
        (64, 384, 101),
        (1024, 384, 1536),
        (513, 97, 129),
    ]
    .into_iter()
    .enumerate()
    {
        let (m, k, n) = dims;
        let seed = 0xd1_0000 + index as u64 * 8;
        let dy = synth(m * n, seed);
        let w = synth(k * n, seed + 1);
        let first = input_gradient(&ctx, dims, &dy, &w);
        let second = input_gradient(&ctx, dims, &dy, &w);
        assert_eq!(bits(&first), bits(&second), "dX {dims:?} repeats");
        let mut want = vec![0f64; m * k];
        let mut magnitude = vec![0f64; m * k];
        for row in 0..m {
            for i in 0..k {
                for j in 0..n {
                    let product = f64::from(dy[row * n + j]) * f64::from(w[i * n + j]);
                    want[row * k + i] += product;
                    magnitude[row * k + i] += product.abs();
                }
            }
        }
        assert_close(&format!("dX {dims:?}"), &first, &want, &magnitude);
    }
}
