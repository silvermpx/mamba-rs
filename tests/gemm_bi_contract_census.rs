//! G1 of the 0.6.10 program — the contract census.
//!
//! Question 1: is the WMMA m16n16k16 tile (the Fixed family's bf16/f16
//! path, `gemm_bi_fixed.cu`) bit-identical to the inline
//! `mma.sync.m16n8k16` K-slab (the triad's TC tier)? `wmma::mma_sync` on
//! an m16n16k16 fragment lowers to a pair of m16n8k16 covering the full
//! k=16, so per element the chain MAY already be the same ascending
//! sequence — in which case retiring the WMMA path is golden-free.
//!
//! Census of 2026-08-26 (sm_89): bit-identical on EVERY shape, tails and
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
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::sgemm_bi_forward_tc;

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
        ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
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
            sgemm_bi_forward_tc(
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
    // PINNED (census of 2026-08-26, sm_89): the two paths are ONE
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
