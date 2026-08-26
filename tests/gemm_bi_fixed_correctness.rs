//! Correctness of the FIXED-tile batch-invariant GEMM
//! (`kernels/gemm_bi_fixed.cu`, `BiGemmFamily::Fixed`) against a CPU
//! reference, across shapes that exercise the tile tails.
//!
//! The family had no direct test while it sat off every dispatch path;
//! it has one now, and it is the gate any tile-geometry change must
//! pass before a benchmark number means anything.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32) * 0.4 - 0.2
        })
        .collect()
}

/// `y[m][n] = sum_k x[m][k] * w[k][n]`, ascending k.
fn cpu_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w[kk * n + j];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

#[test]
#[ignore = "needs a CUDA device"]
fn fixed_tile_matches_cpu_across_tails() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let stream = ctx.stream.clone();

    // (m, k, n): exact tile multiples first, then every tail combination
    // (m not a multiple of the M tile, n not of the N tile, k not of the
    // K tile), then a production-shaped case.
    let shapes: &[(usize, usize, usize)] = &[
        (64, 32, 64),
        (128, 64, 128),
        (127, 64, 128),
        (128, 64, 127),
        (128, 63, 128),
        (127, 63, 127),
        (321, 96, 193),
        (4621, 384, 384),
    ];

    for &(m, k, n) in shapes {
        let x_host = synth(m * k, 0xA11CE ^ m as u64);
        let w_host = synth(k * n, 0xB0B ^ n as u64);
        let want = cpu_ref(&x_host, &w_host, m, k, n);

        let x = GpuBuffer::from_cpu(&stream, &x_host).expect("x");
        let w = GpuBuffer::from_cpu(&stream, &w_host).expect("w");
        let mut y = GpuBuffer::zeros(&stream, m * n).expect("y");

        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
        gpu_sgemm_forward_raw(&ctx, &mut y, &x, w.raw_ptr(&stream), None, (m, k, n))
            .expect("fixed forward");
        let got = y.to_cpu(&stream).expect("d2h");
        ctx.set_batch_invariant(false);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);

        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        for i in 0..m {
            for j in 0..n {
                let d = (got[i * n + j] - want[i * n + j]).abs();
                if d > worst {
                    worst = d;
                    worst_at = (i, j);
                }
            }
        }
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        println!(
            "m={m:<5} k={k:<4} n={n:<5} worst |gpu-cpu| = {worst:.3e} at {worst_at:?} (scale {scale:.3e})"
        );
        assert!(
            worst <= 2e-4 * scale,
            "m={m} k={k} n={n}: fixed tile differs from the CPU reference by {worst:.3e} \
             at {worst_at:?} (scale {scale:.3e})"
        );
    }
}
