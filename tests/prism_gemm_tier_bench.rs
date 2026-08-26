//! Prism serve shapes: cuBLAS vs `sgemm_bi` (training triad, f32) vs
//! `gemm_bi` (the WMMA batch-invariant GEMM) — latency and cross-kernel
//! drift at the exact GEMMs the m3 classify prefill runs.
//!
//! M = T_TOTAL per page (4621 = 57x81 patches + 4 registers); the batched
//! rows show whether a bucket boundary is ever crossed.
//!
//! Run on the GPU box:
//!   cargo test --release --features cuda --test prism_gemm_tier_bench \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::BiGemmFamily;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

const ITERS: usize = 30;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32) * 0.2 - 0.1
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device"]
fn prism_shapes_cublas_vs_sgemm_bi_vs_gemm_bi() {
    let dev = GpuDevice::new(0).expect("cuda device");

    // The three projections of one m3 classify layer at d_model=384,
    // expand=2 (d_inner=768), plus the input projection from the 32x32
    // grayscale patch (INPUT_DIM=1024). in_proj_out_dim is reported by
    // the engine as ip; 1928 is the e111 value (2*768 + 2*128 + 3*24 + 64).
    let t_page = 4621usize;
    let shapes: &[(&str, usize, usize)] = &[
        ("input_proj  K=1024 N=384", 1024, 384),
        ("in_proj     K=384  N=1928", 384, 1928),
        ("out_proj    K=768  N=384", 768, 384),
    ];

    for &(label, k, n) in shapes {
        for pages in [1usize, 4] {
            // Fresh ctx per cell: disable_tf32 is one-way on a handle, so
            // reusing the ctx would silently turn every later "TF32" arm
            // into a second f32 arm (the exact mislabeling the first run
            // of this bench shipped).
            let ctx = GpuCtx::new(&dev).expect("ctx");
            let stream = ctx.stream.clone();
            let m = t_page * pages;
            let x = GpuBuffer::from_cpu(&stream, &synth(m * k, 0xA11CE)).expect("x");
            let w = GpuBuffer::from_cpu(&stream, &synth(k * n, 0xB0B)).expect("w");
            let mut y = GpuBuffer::zeros(&stream, m * n).expect("y");

            let mut timed = |tag: &str, bi: bool, family: BiGemmFamily| -> (f64, Vec<f32>) {
                ctx.set_batch_invariant(bi);
                ctx.set_bi_gemm_family(family);
                let once = |y: &mut GpuBuffer| {
                    gpu_sgemm_forward_raw(&ctx, y, &x, w.raw_ptr(&stream), None, (m, k, n))
                        .expect("forward")
                };
                once(&mut y);
                stream.synchronize().expect("sync");
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    once(&mut y);
                }
                stream.synchronize().expect("sync");
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let out = y.to_cpu(&stream).expect("d2h");
                println!("  {tag:<12} {us:9.1} us");
                (us, out)
            };

            println!("{label}  M={m} ({pages} page(s))");
            let (us_cublas, y_cublas) = timed("cuBLAS-TF32", false, BiGemmFamily::Triad);
            ctx.disable_tf32();
            let (us_cublas_f32, _y_cublas_f32) = timed("cuBLAS-f32", false, BiGemmFamily::Triad);
            let _ = us_cublas_f32;
            let (us_triad, y_triad) = timed("bi:triad", true, BiGemmFamily::Triad);
            let (us_fixed, y_fixed) = timed("bi:fixed", true, BiGemmFamily::Fixed);
            ctx.set_batch_invariant(false);
            ctx.set_bi_gemm_family(BiGemmFamily::Triad);

            let drift = |a: &[f32], b: &[f32]| -> f32 {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            println!(
                "  ratios vs cuBLAS: triad {:.2}x, fixed {:.2}x | drift triad {:.2e}, fixed {:.2e}",
                us_triad / us_cublas,
                us_fixed / us_cublas,
                drift(&y_cublas, &y_triad),
                drift(&y_cublas, &y_fixed),
            );
        }
    }
}
