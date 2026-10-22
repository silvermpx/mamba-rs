//! Under the batch-invariant exact policy every row of a forward equals the
//! CPU fold of its layer, bit for bit, at every row count: a shallow layer
//! as one `fma` chain seeded with the bias (one `+0.0` where the reduction
//! is not a whole number of 16-input slabs), a deep or narrow one as
//! 64-input chunks summed in order (a chunk closed with `+0.0` where it is
//! not a whole number of 32-input slabs) before the bias. Rows that pass the
//! split scratch go in pieces and fold the same way.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::blas::gpu_gemm_f32_forward_ptrs;
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

fn chunked(k: usize, n: usize) -> bool {
    k >= 512 || (n < 128 && k >= 384)
}

/// The fold of one row, as a CPU twin computes it.
fn fold_row(w: &[f32], x: &[f32], bias: Option<&[f32]>, k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|column| {
            if chunked(k, n) {
                let mut sum = 0.0f32;
                for chunk in 0..k.div_ceil(64) {
                    let (first, end) = (chunk * 64, ((chunk + 1) * 64).min(k));
                    let mut partial = 0.0f32;
                    for input in first..end {
                        partial = x[input].mul_add(w[input * n + column], partial);
                    }
                    if !(end - first).is_multiple_of(32) {
                        partial += 0.0;
                    }
                    sum = if chunk == 0 { partial } else { sum + partial };
                }
                bias.map_or(sum, |b| sum + b[column])
            } else {
                let mut acc = bias.map_or(0.0, |b| b[column]);
                for input in 0..k {
                    acc = x[input].mul_add(w[input * n + column], acc);
                }
                if !k.is_multiple_of(16) {
                    acc += 0.0;
                }
                acc
            }
        })
        .collect()
}

fn upload(ctx: &GpuCtx, values: &[f32]) -> GpuBuffer {
    let mut buffer = GpuBuffer::zeros(&ctx.stream, values.len()).unwrap();
    buffer.upload(&ctx.stream, values).unwrap();
    buffer
}

/// Shallow chain layers, deep and narrow chunk layers, ragged reductions.
const LAYERS: [(usize, usize); 10] = [
    (33, 17),
    (64, 96),
    (96, 40),
    (200, 300),
    (256, 2112),
    (384, 101),
    (384, 384),
    (512, 256),
    (833, 384),
    (3000, 40),
];
const ROWS: [usize; 10] = [1, 7, 16, 17, 64, 65, 128, 513, 1024, 4096];

#[test]
fn exact_batch_invariant_rows_equal_their_cpu_fold() {
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let ctx = GpuCtx::new(&device).unwrap();
    ctx.route_controls()
        .set_f32_policy(F32TriadPolicy::ExactFmaBatchInvariant);
    for (index, (k, n)) in LAYERS.into_iter().enumerate() {
        let seed = 0xe7ac_0000 + index as u64 * 16;
        let w = synth(k * n, seed);
        let bias = synth(n, seed + 1);
        let w_buffer = upload(&ctx, &w);
        let bias_buffer = upload(&ctx, &bias);
        // A deep layer also runs past one piece of the split scratch.
        let extra = if chunked(k, n) && n >= 256 { 10_000 } else { 0 };
        for rows in ROWS.into_iter().chain((extra > 0).then_some(extra)) {
            let x = synth(rows * k, seed + 2 + rows as u64);
            let x_buffer = upload(&ctx, &x);
            for with_bias in [false, true] {
                let y = GpuBuffer::zeros(&ctx.stream, rows * n).unwrap();
                unsafe {
                    gpu_gemm_f32_forward_ptrs(
                        &ctx,
                        y.cached_ptr(),
                        x_buffer.cached_ptr(),
                        w_buffer.cached_ptr(),
                        with_bias.then(|| bias_buffer.cached_ptr()),
                        (rows, k, n),
                    )
                    .unwrap_or_else(|error| panic!("{rows}x{k}x{n}: {error}"));
                }
                ctx.stream.synchronize().unwrap();
                let got = y.to_cpu(&ctx.stream).unwrap();
                let bias = with_bias.then_some(bias.as_slice());
                for row in 0..rows {
                    let want = fold_row(&w, &x[row * k..(row + 1) * k], bias, k, n);
                    for (column, (g, e)) in
                        got[row * n..(row + 1) * n].iter().zip(&want).enumerate()
                    {
                        assert_eq!(
                            g.to_bits(),
                            e.to_bits(),
                            "{rows}x{k}x{n} bias {with_bias}: row {row} column {column}: {g} vs {e}",
                        );
                    }
                }
            }
        }
    }
}
