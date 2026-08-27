//! The bridge census between the two kernels: the FIXED inference
//! ladder (`kernels/gemm_bi_fixed/`, GBF namespace) must stay
//! byte-identical to the triad forward tiles it was copied from at the
//! split point. This suite is permanent - it is what makes the
//! two-kernel split safe: any arithmetic drift between the inference
//! kernel and the training kernel's forward is a loud red, never a
//! silent divergence.
//!
//! Also gates the GBF safety layer: a 2-byte-misaligned typed subview
//! must produce the SAME bits as the aligned layout of the same values
//! (the hardened gate routes it to the scalar stage; the unhardened
//! fast path staged wrong bytes silently).
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{FixedTile, fixed_forward};
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    TcFwdOperands, TcTile, sgemm_bi_forward_tc_with_tile,
};

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 0.4 - 0.2
        })
        .collect()
}

/// The exposing probe (the matvec-census lesson: tame inputs hide real
/// arithmetic differences).
fn exposing(n: usize, seed: u64) -> Vec<f32> {
    let mut v = synth(n, seed);
    for (i, x) in v.iter_mut().enumerate() {
        match i % 4 {
            0 => *x = 4096.0,
            1 => *x = -4096.0,
            2 => *x *= 512.0,
            _ => {}
        }
    }
    v
}

fn bits(ctx: &GpuCtx, buf: &DtypedBuf, n: usize) -> Vec<u32> {
    let mut h = vec![0.0f32; n];
    buf.download_f32(&ctx.stream, &mut h).unwrap();
    h.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs a CUDA device"]
fn bridge_fixed_ladder_bit_identical_to_triad() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");

    // (m, k, n, expected fixed tile, matching forced triad tile)
    let cases: &[(usize, usize, usize, FixedTile, TcTile)] = &[
        (5, 768, 2560, FixedTile::Tc16, TcTile::Thin16),
        (64, 384, 384, FixedTile::Tc16, TcTile::Thin16),
        (300, 768, 40, FixedTile::Tc16, TcTile::Thin16),
        (129, 384, 1928, FixedTile::Tc64, TcTile::Tile64),
        (100, 383, 383, FixedTile::Tc64, TcTile::Tile64),
        (4621, 768, 2304, FixedTile::Tc128, TcTile::Tile128),
        (2048, 1024, 3072, FixedTile::Tc128, TcTile::Tile128),
    ];
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for &(m, k, n, want_fixed, force_triad) in cases {
            let a = DtypedBuf::zeros(&ctx.stream, m * k, dt).unwrap();
            a.upload_f32(&ctx.stream, &exposing(m * k, 0xA11CE ^ m as u64))
                .unwrap();
            let w = DtypedBuf::zeros(&ctx.stream, k * n, dt).unwrap();
            w.upload_f32(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64))
                .unwrap();
            let c_fixed = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();
            let c_triad = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

            let tp = |b: &DtypedBuf| TypedPtr {
                ptr: b.cached_ptr(),
                dtype: dt,
            };
            let got = fixed_forward(&ctx, tp(&c_fixed), tp(&a), tp(&w), None, (m, k, n))
                .expect("fixed forward");
            assert_eq!(got, want_fixed, "tile pick at M{m} K{k} N{n}");
            sgemm_bi_forward_tc_with_tile(
                &ctx.stream,
                &ctx.kernels,
                &TcFwdOperands {
                    y: tp(&c_triad),
                    x: tp(&a),
                    w: tp(&w),
                    bias_ptr: 0,
                },
                (m, k, n),
                force_triad,
            )
            .expect("triad forced tile");
            ctx.stream.synchronize().unwrap();

            let bf = bits(&ctx, &c_fixed, m * n);
            let bt = bits(&ctx, &c_triad, m * n);
            let bad = bf.iter().zip(&bt).filter(|(x, y)| x != y).count();
            assert_eq!(
                bad,
                0,
                "{dt:?} M{m} K{k} N{n} {want_fixed:?}: {bad}/{} bits differ from the triad twin",
                m * n
            );
        }
    }
    println!("bridge: fixed ladder == triad tiles, all cases bitwise");
}

#[test]
#[ignore = "needs a CUDA device"]
fn misaligned_subview_matches_aligned_bits() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let dt = WeightDtype::Bf16;
    let (m, k, n) = (129usize, 384usize, 384usize);

    let host_a = exposing(m * k, 7);
    let host_w = synth(k * n, 9);

    // Aligned reference.
    let a0 = DtypedBuf::zeros(&ctx.stream, m * k, dt).unwrap();
    a0.upload_f32(&ctx.stream, &host_a).unwrap();
    let w = DtypedBuf::zeros(&ctx.stream, k * n, dt).unwrap();
    w.upload_f32(&ctx.stream, &host_w).unwrap();
    let c0 = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

    // The same A values one ELEMENT into a bigger buffer: base is 2-byte
    // aligned, stride stays a multiple of 8 - exactly the operand class
    // the unhardened fast gate mis-staged.
    let mut padded = vec![0.0f32; m * k + 8];
    padded[1..1 + m * k].copy_from_slice(&host_a);
    let a1 = DtypedBuf::zeros(&ctx.stream, m * k + 8, dt).unwrap();
    a1.upload_f32(&ctx.stream, &padded).unwrap();
    let a1_view = a1.cached_ptr() + dt.size_bytes() as u64;
    let c1 = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

    let tp = |p: u64| TypedPtr { ptr: p, dtype: dt };
    fixed_forward(
        &ctx,
        tp(c0.cached_ptr()),
        tp(a0.cached_ptr()),
        tp(w.cached_ptr()),
        None,
        (m, k, n),
    )
    .unwrap();
    fixed_forward(
        &ctx,
        tp(c1.cached_ptr()),
        tp(a1_view),
        tp(w.cached_ptr()),
        None,
        (m, k, n),
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let b0 = bits(&ctx, &c0, m * n);
    let b1 = bits(&ctx, &c1, m * n);
    let bad = b0.iter().zip(&b1).filter(|(x, y)| x != y).count();
    assert_eq!(
        bad,
        0,
        "misaligned subview diverged from aligned bits on {bad}/{} elements",
        m * n
    );
    println!("safety gate: misaligned subview == aligned bits");
}
