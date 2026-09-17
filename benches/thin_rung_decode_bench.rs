//! The Thin16 rung against matvec_bi and Tile64 at
//! decode shapes. The rung only earns its place if it beats matvec on
//! the wall clock - the prediction (deep cp.async supplies the
//! memory-level parallelism a register-capped scalar CTA cannot) is not
//! a result until this prints it.
#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_bi_forward_raw, gpu_gemm_typed_forward_raw,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    TcFwdOperands, TcTile, gemm_bi_forward_tc_with_tile,
};

const ITERS: usize = 200;

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

fn thin16_vs_matvec_decode() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Triad);

    // Decode shapes: (m, k, n) across model widths.
    let shapes: &[(usize, usize, usize)] = &[
        (1, 768, 2560),
        (1, 1536, 768),
        (1, 2560, 2560),
        (4, 768, 2560),
        (16, 768, 2304),
        (32, 1536, 1536),
        // The Thin16/Tile64 crossover band - the ladder selector's
        // threshold is pinned from these rows, not from theory.
        (48, 768, 2304),
        (64, 768, 2304),
        (96, 768, 2304),
        (128, 768, 2304),
        (256, 768, 2304),
        (48, 1536, 1536),
        (64, 1536, 1536),
        (128, 1536, 1536),
    ];

    println!(
        "{:>4} {:>5} {:>5} | {:>10} {:>10} {:>10} | ratio thin/matvec",
        "m", "k", "n", "matvec us", "thin16 us", "tile64 us"
    );
    for &(m, k, n) in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, m * k, WeightDtype::Bf16).expect("A");
        a.upload_f32(&ctx.stream, &synth(m * k, 0xA11CE))
            .expect("A up");
        let w = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::Bf16).expect("W");
        w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B))
            .expect("W up");
        let c = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("C");

        // matvec_bi through the public typed route - the SCALAR tier
        // (this ctx never enables the TC tier), where m < 128 still
        // routes to matvec.
        let mv = |c: &DtypedBuf| {
            gpu_gemm_typed_forward_raw(
                &ctx,
                TypedPtr {
                    ptr: c.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: a.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: w.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                None,
                (m, k, n),
            )
            .expect("matvec route");
        };
        let tc = |c: &DtypedBuf, tile: TcTile| {
            gemm_bi_forward_tc_with_tile(
                &ctx.stream,
                &ctx.kernels,
                &TcFwdOperands {
                    y: TypedPtr {
                        ptr: c.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    x: TypedPtr {
                        ptr: a.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    w: TypedPtr {
                        ptr: w.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    bias_ptr: 0,
                },
                (m, k, n),
                tile,
            )
            .expect("tc launch");
        };

        let time = |f: &dyn Fn()| -> f64 {
            f();
            ctx.stream.synchronize().expect("sync");
            let t0 = Instant::now();
            for _ in 0..ITERS {
                f();
            }
            ctx.stream.synchronize().expect("sync");
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };

        let t_mv = time(&|| mv(&c));
        let t_16 = time(&|| tc(&c, TcTile::Thin16));
        let t_64 = time(&|| tc(&c, TcTile::Tile64));
        println!(
            "{m:>4} {k:>5} {n:>5} | {t_mv:>10.2} {t_16:>10.2} {t_64:>10.2} | {:.2}x",
            t_16 / t_mv
        );
    }
}

/// G5 measurement: the Tile128 rung at prefill-class bf16 shapes (page
/// rows x model dims). Forced-tile timings so the 3-stage swizzled NN
/// pipeline is measured directly against the Tile64 twin.
fn tile128_prefill_bench() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Triad);

    let shapes: &[(usize, usize, usize)] = &[
        (4621, 384, 1928),
        (4621, 768, 2304),
        (4621, 1928, 384),
        (2048, 768, 2304),
        (2048, 2304, 768),
    ];

    println!(
        "{:>5} {:>5} {:>5} | {:>11} {:>11}",
        "m", "k", "n", "tile128 us", "tile64 us"
    );
    for &(m, k, n) in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, m * k, WeightDtype::Bf16).expect("A");
        a.upload_f32(&ctx.stream, &synth(m * k, 0xA11CE))
            .expect("A up");
        let w = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::Bf16).expect("W");
        w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B))
            .expect("W up");
        let c = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("C");

        let tc = |tile: TcTile| {
            gemm_bi_forward_tc_with_tile(
                &ctx.stream,
                &ctx.kernels,
                &TcFwdOperands {
                    y: TypedPtr {
                        ptr: c.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    x: TypedPtr {
                        ptr: a.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    w: TypedPtr {
                        ptr: w.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    bias_ptr: 0,
                },
                (m, k, n),
                tile,
            )
            .expect("tc launch");
        };
        let time = |tile: TcTile| -> f64 {
            tc(tile);
            ctx.stream.synchronize().expect("sync");
            let t0 = Instant::now();
            for _ in 0..ITERS {
                tc(tile);
            }
            ctx.stream.synchronize().expect("sync");
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        let t128 = time(TcTile::Tile128);
        let t64 = time(TcTile::Tile64);

        // cuBLAS arms on the same shapes: PEDANTIC bf16 (the deterministic
        // baseline everyone ships) and fast-TC bf16 + TF32 f32 (the
        // non-deterministic speed ceilings).
        let cublas_bf16 = |fast: bool| -> f64 {
            ctx.set_gemm_mode(if fast {
                GemmMode::CublasFast
            } else {
                GemmMode::CublasPedantic
            })
            .unwrap();
            let run = || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    TypedPtr {
                        ptr: c.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    TypedPtr {
                        ptr: a.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    TypedPtr {
                        ptr: w.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    None,
                    (m, k, n),
                )
                .expect("cublas route");
            };
            run();
            ctx.stream.synchronize().expect("sync");
            let t0 = Instant::now();
            for _ in 0..ITERS {
                run();
            }
            ctx.stream.synchronize().expect("sync");
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        let t_ped = cublas_bf16(false);
        let t_fast = cublas_bf16(true);

        // TF32: f32 operands through the vendor TF32 route.
        let af = GpuBuffer::from_cpu(&ctx.stream, &synth(m * k, 0xA11CE)).expect("Af");
        let wf = GpuBuffer::from_cpu(&ctx.stream, &synth(k * n, 0xB0B)).expect("Wf");
        let mut cf = GpuBuffer::zeros(&ctx.stream, m * n).expect("Cf");
        let t_tf32 = {
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let mut run = || {
                gpu_gemm_bi_forward_raw(
                    &ctx,
                    &mut cf,
                    &af,
                    wf.raw_ptr(&ctx.stream),
                    None,
                    (m, k, n),
                )
                .expect("tf32 route");
            };
            run();
            ctx.stream.synchronize().expect("sync");
            let t0 = Instant::now();
            for _ in 0..ITERS {
                run();
            }
            ctx.stream.synchronize().expect("sync");
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        println!(
            "{m:>5} {k:>5} {n:>5} | {t128:>11.2} {t64:>11.2} | ped {t_ped:>9.2} fast {t_fast:>9.2} tf32 {t_tf32:>9.2} | ours/ped {:.2}x ours/tf32 {:.2}x",
            t128 / t_ped,
            t128 / t_tf32
        );
    }
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("thin16_vs_matvec_decode") {
        thin16_vs_matvec_decode();
    }
    if run("tile128_prefill_bench") {
        tile128_prefill_bench();
    }
}
