//! An exact input gradient whose width is not a whole number of 32 columns
//! runs its first columns through the transpose and the split sum, and the
//! last ones, the column tail, as one `fma` chain per output from the first
//! output gradient up. Every tail column must equal that chain on the CPU
//! bit for bit, with one tail column and with thirty-one, over a few rows
//! and over a thousand.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::blas::gpu_gemm_f32_backward_dx_ptrs;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
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

fn upload(ctx: &GpuCtx, values: &[f32]) -> GpuBuffer {
    let mut buffer = GpuBuffer::zeros(&ctx.stream, values.len()).unwrap();
    buffer.upload(&ctx.stream, values).unwrap();
    buffer
}

#[test]
fn exact_input_gradient_tail_columns_equal_their_cpu_chain() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = GpuCtx::new(&device).unwrap();
    // (batch, inputs, outputs): 31 and 30 tail columns over 32 rows, one
    // tail column over 64 and over 1024 rows.
    for (index, dims) in [
        (32, 95, 128),
        (32, 94, 128),
        (64, 833, 384),
        (1024, 833, 384),
    ]
    .into_iter()
    .enumerate()
    {
        let (m, k, n) = dims;
        let seed = 0x7a11_0000 + index as u64 * 8;
        let dy = synth(m * n, seed);
        let w = synth(k * n, seed + 1);
        let dy_buffer = upload(&ctx, &dy);
        let w_buffer = upload(&ctx, &w);
        let dx = GpuBuffer::zeros(&ctx.stream, m * k).unwrap();
        unsafe {
            gpu_gemm_f32_backward_dx_ptrs(
                &ctx,
                dx.cached_ptr(),
                dy_buffer.cached_ptr(),
                w_buffer.cached_ptr(),
                dims,
            )
            .unwrap_or_else(|error| panic!("{dims:?}: {error}"));
        }
        ctx.stream.synchronize().unwrap();
        let got = dx.to_cpu(&ctx.stream).unwrap();
        let first_tail = k - k % 32;
        for row in 0..m {
            for column in first_tail..k {
                let mut chain = 0.0f32;
                for output in 0..n {
                    chain = dy[row * n + output].mul_add(w[column * n + output], chain);
                }
                assert_eq!(
                    got[row * k + column].to_bits(),
                    chain.to_bits(),
                    "{dims:?}: row {row}, tail column {column}"
                );
            }
        }
    }
}
