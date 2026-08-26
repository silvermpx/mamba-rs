//! G3 measurement: the Thin16 rung against matvec_bi and Tile64 at
//! decode shapes. The rung only earns its place if it beats matvec on
//! the wall clock - the prediction (deep cp.async supplies the
//! memory-level parallelism a register-capped scalar CTA cannot) is not
//! a result until this prints it.
#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    TcFwdOperands, TcTile, sgemm_bi_forward_tc_with_tile,
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

#[test]
#[ignore = "needs a CUDA device"]
fn thin16_vs_matvec_decode() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);

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
        // routes to matvec after G4.
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
            sgemm_bi_forward_tc_with_tile(
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
