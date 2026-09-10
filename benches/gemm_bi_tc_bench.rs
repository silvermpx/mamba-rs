//! Tensor-core tier wall clock: tc64 against tc128 on the small trainer
//! shapes, the tensor-core against the scalar triad paths, and the kernel
//! attribute and golden-hash baseline. The tier's contract tests live in
//! tests/gemm_bi_tc.rs.

#![cfg(feature = "cuda")]

use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;
#[path = "../tests/common/digest.rs"]
mod digest;
#[path = "../tests/common/evidence.rs"]
mod evidence;
#[path = "../tests/common/evidence_digest.rs"]
mod evidence_digest;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}
fn quantize(v: &[f32], dt: WeightDtype) -> Vec<f32> {
    match dt {
        WeightDtype::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        WeightDtype::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        WeightDtype::F32 => v.to_vec(),
    }
}
struct Ctx {
    ctx: GpuCtx,
}
impl Ctx {
    fn new() -> Self {
        let device = GpuDevice::new(0).expect("gpu");
        let ctx = GpuCtx::new(&device).expect("ctx");
        Self { ctx }
    }

    fn typed_buf(&self, data: &[f32], dt: WeightDtype) -> DtypedBuf {
        let b = DtypedBuf::zeros(&self.ctx.stream, data.len(), dt).unwrap();
        b.upload_f32(&self.ctx.stream, data).unwrap();
        b
    }

    fn f32_buf(&self, data: &[f32]) -> GpuBuffer {
        let mut b = GpuBuffer::zeros(&self.ctx.stream, data.len()).unwrap();
        b.upload(&self.ctx.stream, data).unwrap();
        b
    }
}
/// Launch the TC forward on quantized inputs; returns Y upcast to f32.
fn run_tc(
    t: &Ctx,
    dt: WeightDtype,
    dims: (usize, usize, usize),
    qx: &[f32],
    qw: &[f32],
    bias: Option<&GpuBuffer>,
) -> Vec<f32> {
    let (m, _k, n) = dims;
    let xt = t.typed_buf(qx, dt);
    let wt = t.typed_buf(qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    gemm_bi_triad::gemm_bi_forward_tc(
        &t.ctx.stream,
        &t.ctx.kernels,
        TypedPtr {
            ptr: yt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: xt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: wt.cached_ptr(),
            dtype: dt,
        },
        bias.map_or(0, |b| b.cached_ptr()),
        dims,
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut out).unwrap();
    out
}
fn bench_tc64_vs_tc128_small_shapes() {
    use gemm_bi_triad::TcTile;
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    // (m, k, n, label) — trainer GEMMs of the d128/d256 benches plus the
    // crossover region (tiles128 between 32 and 192).
    for (m, k, n, label) in [
        (1024usize, 128usize, 512usize, "d128 in_proj fwd (t128=32)"),
        (1024, 256, 128, "d128 out_proj fwd (t128=8)"),
        (2048, 256, 1024, "d256 in_proj fwd (t128=128)"),
        (2048, 512, 256, "d256 out_proj fwd (t128=32)"),
        (2048, 768, 512, "crossover (t128=64)"),
        (2048, 1536, 768, "d768 out_proj fwd (t128=96)"),
        (2048, 768, 3072, "d768 in_proj fwd (t128=384)"),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let dyt = t.typed_buf(&qdy, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let dw = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
        let tp = |b: &DtypedBuf| TypedPtr {
            ptr: b.cached_ptr(),
            dtype: dt,
        };
        let (xtp, wtp, dytp, ytp, dxtp) = (tp(&xt), tp(&wt), tp(&dyt), tp(&yt), tp(&dxt));

        let iters = 200;
        let time_path = |label: &str, f: &dyn Fn()| -> f64 {
            for _ in 0..5 {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  {label}: {us:.1} us");
            us
        };

        eprintln!("[{label}] M{m} K{k} N{n}");
        let ops = gemm_bi_triad::TcFwdOperands {
            y: ytp,
            x: xtp,
            w: wtp,
            bias_ptr: 0,
        };
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("fwd {tile:?}"), &|| {
                gemm_bi_triad::gemm_bi_forward_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    &ops,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("dW {tile:?}"), &|| {
                gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw.cached_ptr(),
                    dytp,
                    xtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("dX {tile:?}"), &|| {
                gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dxtp,
                    dytp,
                    wtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        // Scalar-tier reference on the same shape (what TC64 must beat).
        use mamba_rs::mamba_ssm::gpu::blas::{
            gemm_bi_backward_dw_typed, gemm_bi_backward_dx_typed, gemm_bi_forward_typed,
        };
        time_path("fwd scalar bi", &|| {
            gemm_bi_forward_typed(&t.ctx, ytp, xtp, wtp, 0, (m, k, n)).unwrap();
        });
        time_path("dW scalar bi", &|| {
            gemm_bi_backward_dw_typed(&t.ctx, dw.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
        });
        time_path("dX scalar bi", &|| {
            gemm_bi_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
        });
    }
}
fn bench_tc_vs_scalar_paths() {
    use mamba_rs::mamba_ssm::gpu::blas::gemm_bi_forward_typed;
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    for (m, k, n) in [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (2048, 768, 512),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let (ytp, xtp, wtp) = (
            TypedPtr {
                ptr: yt.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: xt.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: wt.cached_ptr(),
                dtype: dt,
            },
        );

        let iters = 50;
        let time_path = |label: &str, f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  {label}: {us:.1} us");
            us
        };

        eprintln!("[M{m} K{k} N{n}]");
        let scalar = time_path("scalar bi (native/fallback)", &|| {
            gemm_bi_forward_typed(&t.ctx, ytp, xtp, wtp, 0, (m, k, n)).unwrap();
        });
        let tc = time_path("tensor-core bi", &|| {
            gemm_bi_triad::gemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                ytp,
                xtp,
                wtp,
                0,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  TC speedup vs scalar bi: {:.2}x", scalar / tc);

        // Backward twins on the same shape.
        use mamba_rs::mamba_ssm::gpu::blas::{
            gemm_bi_backward_dw_typed, gemm_bi_backward_dx_typed,
        };
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let dyt = t.typed_buf(&qdy, dt);
        let dytp = TypedPtr {
            ptr: dyt.cached_ptr(),
            dtype: dt,
        };
        let dw = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
        let dxtp = TypedPtr {
            ptr: dxt.cached_ptr(),
            dtype: dt,
        };
        let dw_s = time_path("dW scalar bi", &|| {
            gemm_bi_backward_dw_typed(&t.ctx, dw.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
        });
        let dw_tc = time_path("dW tensor-core", &|| {
            gemm_bi_triad::gemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                dytp,
                xtp,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  dW TC speedup: {:.2}x", dw_s / dw_tc);
        let dx_s = time_path("dX scalar bi", &|| {
            gemm_bi_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
        });
        let dx_tc = time_path("dX tensor-core", &|| {
            gemm_bi_triad::gemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dxtp,
                dytp,
                wtp,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  dX TC speedup: {:.2}x", dx_s / dx_tc);
    }
}
/// Baseline instrumentation for TC-kernel optimization work: function
/// attributes (regs / spills / static smem), bit-level golden hashes
/// of the full TC triad on the contract shapes, and baseline fwd timings.
/// Run before and after each optimization step; goldens must not move on steps
/// that promise bit-identity (swizzle, dynsmem, BK=64 on K%64 ∉ (0,32] shapes).
fn step0_tc_attrs_and_goldens() {
    use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
    use std::time::Instant;
    let t = Ctx::new();
    let k = &t.ctx.kernels;

    println!("== function attributes ==");
    let fams = [
        ("nn_tc", &k.gemm_bi_nn_tc_typed),
        ("tn_tc", &k.gemm_bi_tn_tc_typed),
        ("nt_tc", &k.gemm_bi_nt_tc_typed),
        ("nn_tc64", &k.gemm_bi_nn_tc64_typed),
        ("tn_tc64", &k.gemm_bi_tn_tc64_typed),
        ("nt_tc64", &k.gemm_bi_nt_tc64_typed),
    ];
    for (name, kern) in fams {
        for (dtn, f) in [("bf16", &kern.bf16), ("f16", &kern.f16)] {
            println!(
                "{name}_{dtn}: regs={} local={} smem={} maxthr={}",
                f.num_regs().unwrap(),
                f.local_size_bytes().unwrap(),
                f.shared_size_bytes().unwrap(),
                f.max_threads_per_block().unwrap(),
            );
        }
    }

    println!("== golden hashes (fwd/dW/dX bits) ==");
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k_, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),
            (300, 768, 3072),
            (2048, 768, 3072),
        ] {
            let qx = quantize(&det(m * k_, 11, 1.0), dt);
            let qw = quantize(&det(k_ * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);
            let y = run_tc(&t, dt, (m, k_, n), &qx, &qw, Some(&b32));
            let h_fwd = digest::fnv1a_f32(&y);

            let qdy = quantize(&det(m * n, 44, 0.5), dt);
            let dyt = t.typed_buf(&qdy, dt);
            let xt = t.typed_buf(&qx, dt);
            let wt = t.typed_buf(&qw, dt);

            let dw = GpuBuffer::zeros(&t.ctx.stream, k_ * n).unwrap();
            gemm_bi_triad::gemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dyt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: dt,
                },
                (m, k_, n),
            )
            .unwrap();
            let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k_, dt).unwrap();
            gemm_bi_triad::gemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: dxt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: dyt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: dt,
                },
                (m, k_, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let dwh = dw.to_cpu(&t.ctx.stream).unwrap();
            let mut dxh = vec![0.0f32; m * k_];
            dxt.download_f32(&t.ctx.stream, &mut dxh).unwrap();
            let h_dw = digest::fnv1a_f32(&dwh);
            let h_dx = digest::fnv1a_f32(&dxh);
            println!("[{dt:?} M{m} K{k_} N{n}] fwd={h_fwd:016x} dW={h_dw:016x} dX={h_dx:016x}");
            let cell = format!("{dt:?}.M{m}K{k_}N{n}");
            evidence_digest::record_digest("gemm_bi_tc", "goldens.fwd", &cell, h_fwd)
                .expect("acceptance evidence");
            evidence_digest::record_digest("gemm_bi_tc", "goldens.dW", &cell, h_dw)
                .expect("acceptance evidence");
            evidence_digest::record_digest("gemm_bi_tc", "goldens.dX", &cell, h_dx)
                .expect("acceptance evidence");
        }
    }

    println!("== baseline fwd timings (bf16, 128-tile route) ==");
    for (m, k_, n) in [(2048usize, 768usize, 3072usize), (4096, 1536, 3072)] {
        let qx = quantize(&det(m * k_, 11, 1.0), WeightDtype::Bf16);
        let qw = quantize(&det(k_ * n, 22, 0.5), WeightDtype::Bf16);
        let xt = t.typed_buf(&qx, WeightDtype::Bf16);
        let wt = t.typed_buf(&qw, WeightDtype::Bf16);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, WeightDtype::Bf16).unwrap();
        let run = || {
            gemm_bi_triad::gemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: yt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                0,
                (m, k_, n),
            )
            .unwrap();
        };
        for _ in 0..3 {
            run();
        }
        t.ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..50 {
            run();
        }
        t.ctx.stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        let tflops = 2.0 * m as f64 * k_ as f64 * n as f64 / (us * 1e-6) / 1e12;
        println!("fwd M{m} K{k_} N{n}: {us:.1} us = {tflops:.1} TFLOPS");
    }
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("bench_tc64_vs_tc128_small_shapes") {
        bench_tc64_vs_tc128_small_shapes();
    }
    if run("bench_tc_vs_scalar_paths") {
        bench_tc_vs_scalar_paths();
    }
    if run("step0_tc_attrs_and_goldens") {
        step0_tc_attrs_and_goldens();
    }
}
