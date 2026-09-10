//! The WMMA / mma.sync arithmetic-contract census.
//!
//! Question 1: is the WMMA m16n16k16 tile (the Fixed family's bf16/f16
//! path, `kernels/gemm_bi_inference/`) bit-identical to the inline
//! `mma.sync.m16n8k16` K-slab (the triad's TC tier)? `wmma::mma_sync` on
//! an m16n16k16 fragment lowers to a pair of m16n8k16 covering the full
//! k=16, so per element the chain MAY already be the same ascending
//! sequence — in which case retiring the WMMA path is golden-free.
//!
//! Measured on sm_89: bit-identical on EVERY shape, tails and
//! the 8.9M-element production shape included - pinned as an assertion.
//! wmma::mma_sync on m16n16k16 lowers to the same ascending m16n8k16
//! pair the TC tier issues explicitly; the two paths are one arithmetic
//! contract and the WMMA macro can be retired golden-free.
//!
//! Run on the GPU box:
//!   cargo test --release --features cuda --test gemm_bi_contract_census \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gemm_bi_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::gemm_bi_forward_tc;

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

fn bits(buf: &DtypedBuf, elems: usize, ctx: &GpuCtx) -> Vec<u32> {
    let mut full = vec![0.0f32; elems];
    buf.download_f32(&ctx.stream, &mut full).expect("D2H");
    full.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs a CUDA device"]
fn census_wmma_vs_mma_sync() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_batch_invariant(true);

    // (m, k, n): TC coverage needs M >= 64 && N >= 64. Tile multiples,
    // tails on every axis, and the production shapes.
    let shapes: &[(usize, usize, usize)] = &[
        (64, 64, 64),
        (128, 384, 384),
        (127, 384, 384),
        (128, 383, 384),
        (128, 384, 383),
        (129, 65, 127),
        (4621, 384, 1928),
        (128, 768, 2304),
    ];

    let mut all_equal = true;
    for &(m, k, n) in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, m * k, WeightDtype::Bf16).expect("A");
        a.upload_f32(&ctx.stream, &synth(m * k, 0xA11CE ^ m as u64))
            .expect("A up");
        let w = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::Bf16).expect("W");
        w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64))
            .expect("W up");
        let c_wmma = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("Cw");
        let c_tc = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("Ct");

        // Route 1: the Fixed family's WMMA tile.
        ctx.set_bi_gemm_family(BiGemmFamily::Inference);
        let run_wmma = |c: &DtypedBuf| {
            gemm_bi_forward_raw(
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
            .expect("wmma forward");
        };
        run_wmma(&c_wmma);
        let b1 = bits(&c_wmma, m * n, &ctx);
        run_wmma(&c_wmma);
        assert_eq!(
            b1,
            bits(&c_wmma, m * n, &ctx),
            "WMMA not run-stable at {m}x{k}x{n}"
        );

        // Route 2: the triad's mma.sync TC tier, called directly.
        let run_tc = |c: &DtypedBuf| {
            gemm_bi_forward_tc(
                &ctx.stream,
                &ctx.kernels,
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
                0,
                (m, k, n),
            )
            .expect("tc forward")
        };
        let tile = run_tc(&c_tc);
        let b2 = bits(&c_tc, m * n, &ctx);
        run_tc(&c_tc);
        assert_eq!(
            b2,
            bits(&c_tc, m * n, &ctx),
            "TC not run-stable at {m}x{k}x{n}"
        );

        let equal = b1 == b2;
        all_equal &= equal;
        let diff = b1.iter().zip(&b2).filter(|(x, y)| x != y).count();
        println!(
            "{m:>5}x{k:<4}x{n:<5} tile={tile:?}  WMMA == mma.sync: {equal}  (diff elems: {diff}/{})",
            m * n
        );
    }
    println!("CENSUS VERDICT: WMMA bit-identical to mma.sync on all shapes: {all_equal}");
    // PINNED (measured on sm_89): the two paths are ONE
    // arithmetic contract - wmma::mma_sync on m16n16k16 lowers to the
    // same ascending m16n8k16 pair the TC tier issues explicitly, so per
    // element the reduction chain is identical. Retiring the WMMA macro
    // is therefore golden-free. If this ever breaks (a toolchain changes
    // the WMMA lowering), the unification plan must re-golden instead.
    assert!(
        all_equal,
        "WMMA and mma.sync diverged - the golden-free retirement no longer holds"
    );
}

/// G3 ladder-legality gate: the Thin16 rung must be byte-identical to
/// Tile64 on every shape - same arithmetic contract, different schedule.
/// This is the license that makes an M-indexed rung selection legal.
#[test]
#[ignore = "needs a CUDA device"]
fn census_thin16_vs_tile64() {
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        TcFwdOperands, TcTile, gemm_bi_forward_tc_with_tile,
    };

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");

    // Decode-class shapes plus tails and one prefill-class control; the
    // last two pin the exact ladder-gate floors tc_pick_tile_forward now
    // exposes (K below one slab at the N=32 column floor, and the
    // rows >= 64 && cols < 64 arm).
    let shapes: &[(usize, usize, usize)] = &[
        (1, 768, 2560),
        (4, 1536, 768),
        (16, 768, 2304),
        (17, 65, 33),
        (32, 384, 384),
        (129, 384, 1928),
        (128, 383, 383),
        (5, 63, 32),
        (300, 768, 40),
    ];

    for &(m, k, n) in shapes {
        // The magnitude-heterogeneous probe from census_matvec_vs_thin16:
        // byte-identity attested on a tame input is not attestation (the
        // matvec lesson) - the probe must be the exposing one.
        let mut a_host = synth(m * k, 0xA11CE ^ m as u64);
        for (i, v) in a_host.iter_mut().enumerate() {
            match i % 4 {
                0 => *v = 4096.0,
                1 => *v = -4096.0,
                2 => *v *= 512.0,
                _ => {}
            }
        }
        let a = DtypedBuf::zeros(&ctx.stream, m * k, WeightDtype::Bf16).expect("A");
        a.upload_f32(&ctx.stream, &a_host).expect("A up");
        let w = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::Bf16).expect("W");
        w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64))
            .expect("W up");
        let c16 = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("C16");
        let c64 = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("C64");

        let run = |c: &DtypedBuf, tile: TcTile| {
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
        run(&c16, TcTile::Thin16);
        run(&c64, TcTile::Tile64);
        let b16 = bits(&c16, m * n, &ctx);
        let b64 = bits(&c64, m * n, &ctx);
        let diff = b16.iter().zip(&b64).filter(|(x, y)| x != y).count();
        println!(
            "{m:>4}x{k:<5}x{n:<5} Thin16 == Tile64: {} (diff {diff}/{})",
            diff == 0,
            m * n
        );
        assert_eq!(
            diff, 0,
            "Thin16 diverged from Tile64 at {m}x{k}x{n} - the rung is NOT \
             bit-identical and may not join the ladder"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn census_backward_tile64_tail_contract() {
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        TcTile, gemm_bi_backward_dw_tc_with_tile, gemm_bi_backward_dx_tc_with_tile,
    };

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let shapes = [
        (1024usize, 8usize, 256usize),
        (2048, 16, 512),
        (2048, 48, 1536),
        (1024, 256, 40),
        (2048, 512, 48),
        (32, 256, 1024),
    ];

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for &(m, k, n) in &shapes {
            let x = DtypedBuf::zeros(&ctx.stream, m * k, dtype).expect("X");
            x.upload_f32(&ctx.stream, &synth(m * k, 0x1100 ^ m as u64))
                .expect("X upload");
            let dy = DtypedBuf::zeros(&ctx.stream, m * n, dtype).expect("dY");
            dy.upload_f32(&ctx.stream, &synth(m * n, 0x2200 ^ n as u64))
                .expect("dY upload");
            let w = DtypedBuf::zeros(&ctx.stream, k * n, dtype).expect("W");
            w.upload_f32(&ctx.stream, &synth(k * n, 0x3300 ^ k as u64))
                .expect("W upload");

            let initial = synth(k * n, 0x4400 ^ (k * n) as u64);
            let mut dw64 = GpuBuffer::zeros(&ctx.stream, k * n).expect("dW64");
            let mut dw128 = GpuBuffer::zeros(&ctx.stream, k * n).expect("dW128");
            dw64.upload(&ctx.stream, &initial).expect("dW64 upload");
            dw128.upload(&ctx.stream, &initial).expect("dW128 upload");
            for (output, tile) in [(&dw64, TcTile::Tile64), (&dw128, TcTile::Tile128)] {
                gemm_bi_backward_dw_tc_with_tile(
                    &ctx.stream,
                    &ctx.kernels,
                    output.cached_ptr(),
                    TypedPtr {
                        ptr: dy.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: x.cached_ptr(),
                        dtype,
                    },
                    (m, k, n),
                    tile,
                )
                .expect("TN launch");
            }
            ctx.stream.synchronize().expect("TN sync");
            assert_eq!(
                dw64.to_cpu(&ctx.stream)
                    .expect("dW64 download")
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                dw128
                    .to_cpu(&ctx.stream)
                    .expect("dW128 download")
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{dtype:?} TN tile contract changed at M{m} K{k} N{n}"
            );

            let sentinel = vec![3.0f32; m * k];
            let dx64 = DtypedBuf::zeros(&ctx.stream, m * k, dtype).expect("dX64");
            let dx128 = DtypedBuf::zeros(&ctx.stream, m * k, dtype).expect("dX128");
            dx64.upload_f32(&ctx.stream, &sentinel)
                .expect("dX64 upload");
            dx128
                .upload_f32(&ctx.stream, &sentinel)
                .expect("dX128 upload");
            for (output, tile) in [(&dx64, TcTile::Tile64), (&dx128, TcTile::Tile128)] {
                gemm_bi_backward_dx_tc_with_tile(
                    &ctx.stream,
                    &ctx.kernels,
                    TypedPtr {
                        ptr: output.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: dy.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: w.cached_ptr(),
                        dtype,
                    },
                    (m, k, n),
                    tile,
                )
                .expect("NT launch");
            }
            ctx.stream.synchronize().expect("NT sync");
            assert_eq!(
                bits(&dx64, m * k, &ctx),
                bits(&dx128, m * k, &ctx),
                "{dtype:?} NT tile contract changed at M{m} K{k} N{n}"
            );
        }
    }
}

/// The decisive census for the decode ladder: is matvec_bi bit-identical
/// to the TC chain? If yes, matvec and the TC rungs are ONE arithmetic
/// contract and an M-keyed pick between them is legal scheduling; if no,
/// matvec at small M is a documented bucketed exception.
#[test]
#[ignore = "needs a CUDA device"]
fn census_matvec_vs_thin16() {
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_typed_forward_raw;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        TcFwdOperands, TcTile, gemm_bi_forward_tc_with_tile,
    };

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    // Force matvec through the typed route: the Fixed check is first, so
    // flip family to Triad only for the matvec launch below.

    let shapes: &[(usize, usize, usize)] = &[
        (1, 768, 2560),
        (1, 1536, 768),
        (3, 65, 33),
        (16, 768, 2304),
        (32, 384, 384),
    ];

    let mut all_equal = true;
    for &(m, k, n) in shapes {
        // Adversarial + cancellation mix, the strongest probes we have.
        let mut a_host = synth(m * k, 0xA11CE ^ m as u64);
        for (i, v) in a_host.iter_mut().enumerate() {
            match i % 4 {
                0 => *v = 4096.0,
                1 => *v = -4096.0,
                2 => *v *= 512.0,
                _ => {}
            }
        }
        let a = DtypedBuf::zeros(&ctx.stream, m * k, WeightDtype::Bf16).expect("A");
        a.upload_f32(&ctx.stream, &a_host).expect("A up");
        let w = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::Bf16).expect("W");
        w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64))
            .expect("W up");
        let c_mv = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("Cm");
        let c_tc = DtypedBuf::zeros(&ctx.stream, m * n, WeightDtype::Bf16).expect("Ct");

        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        gpu_gemm_typed_forward_raw(
            &ctx,
            TypedPtr {
                ptr: c_mv.cached_ptr(),
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

        gemm_bi_forward_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            &TcFwdOperands {
                y: TypedPtr {
                    ptr: c_tc.cached_ptr(),
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
            TcTile::Thin16,
        )
        .expect("thin16 launch");

        let bm = bits(&c_mv, m * n, &ctx);
        let bt = bits(&c_tc, m * n, &ctx);
        let diff = bm.iter().zip(&bt).filter(|(x, y)| x != y).count();
        let equal = diff == 0;
        all_equal &= equal;
        println!(
            "{m:>4}x{k:<5}x{n:<5} matvec == Thin16: {equal} (diff {diff}/{})",
            m * n
        );
    }
    println!("CENSUS VERDICT: matvec bit-identical to the TC chain: {all_equal}");
}
