//! The cross-M bitwise invariance matrix.
//!
//! The law under test: the bits of
//! output row `r` depend only on `A[r, :]`, the whole `B` operand and the
//! plan — never on `M`, never on `r`, never on which other rows shared
//! the launch.
//!
//! Every reachable family is STRICT: the test derives the OBSERVED
//! boundary set (the ladder values where the probe rows' bits change) and
//! asserts it is empty. The families once declared bucket tables, edges
//! where an M-keyed rule switched the arithmetic (the typed route's M=128
//! matvec/TC break, the f32 split-K and slim entries); one deterministic
//! mode removed them, so any boundary now is a defect.
//!
//! No absolute bits are captured — the suite asserts relations only, so
//! it stays green across a legitimate arithmetic re-golden (that is
//! S4's job, not this file's).
//!
//! Run on the GPU box (`GpuCtx` is !Sync — one ctx per test fn):
//!   cargo test --release --features cuda --test gemm_bi_invariance_matrix \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cuda")]

#[path = "common/evidence.rs"]
mod evidence;

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::blas::{
    TypedPtr, gemm_bi_forward_raw, gpu_gemm_bi_forward_raw, gpu_gemm_typed_forward_raw,
};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;

/// The reduced M ladder: every entry sits on (or adjacent to) a known
/// dispatch edge — matvec row-block 4, FRAG_M 16, ultra-thin 32, tile 64,
/// THE typed break 128, slim-force 512, split-K cap 1024, the serve page.
const M_LADDER: &[usize] = &[
    1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025,
    2048, 4621,
];

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

/// One family under test: launches the prefix `A[0..m]` and returns the
/// bits of the first `min(m, CMP_ROWS) * n` outputs. Comparing the whole
/// common prefix, not just row 0, is what gives the walk its
/// sensitivity: a family switch that differs on ~0.1% of elements is
/// near-invisible to a single row (the first version of this suite
/// missed a real switch exactly that way) and unmissable over 128 rows.
type LaunchFn<'a> = dyn FnMut(usize) -> Vec<u32> + 'a;

const CMP_ROWS: usize = 128;

/// Walk the ladder, return the observed boundary set: every ladder m at
/// which the common-prefix bits differ from the previous ladder m. The
/// prefix rows are the SAME input rows in both launches (one fixed A,
/// prefix launches), so per the L2 law their bits must not move.
fn observed_boundaries(launch: &mut LaunchFn<'_>, k: usize, n: usize) -> Vec<usize> {
    let mut prev: Option<Vec<u32>> = None;
    let mut bounds = Vec::new();
    for &m in M_LADDER {
        let bits = launch(m);
        assert_eq!(
            bits.len(),
            m.min(CMP_ROWS) * n,
            "prefix size at m={m} k={k}"
        );
        if let Some(p) = &prev {
            let common = p.len().min(bits.len());
            if p[..common] != bits[..common] {
                bounds.push(m);
            }
        }
        prev = Some(bits);
    }
    bounds
}

fn assert_strict(family: &str, k: usize, n: usize, observed: &[usize]) {
    evidence::record(
        "gemm_bi_invariance_matrix",
        family,
        &format!("K{k}N{n}"),
        &observed
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(","),
    )
    .expect("acceptance evidence");
    assert!(
        observed.is_empty(),
        "{family} (K={k} N={n}): the probe rows' bits changed at M in \
         {observed:?} — an M-keyed arithmetic switch"
    );
}

/// Shared fixture: A[M_MAX, K] with a fixed probe vector in row 0 and
/// deterministic noise elsewhere, B[K, N]; C sized for the largest M.
struct Fixture {
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
    n: usize,
}

impl Fixture {
    fn new(ctx: &GpuCtx, k: usize, n: usize) -> Self {
        let m_max = *M_LADDER.last().expect("ladder non-empty");
        let mut a_host = synth(m_max * k, 0xA5EED ^ (k * n) as u64);
        // The probe vector: fixed, independent of (K, N) noise.
        for (i, slot) in a_host[..k].iter_mut().enumerate() {
            *slot = ((i % 17) as f32 - 8.0) * 0.03;
        }
        let stream = &ctx.stream;
        Fixture {
            a: GpuBuffer::from_cpu(stream, &a_host).expect("A"),
            b: GpuBuffer::from_cpu(stream, &synth(k * n, 0xB0B ^ n as u64)).expect("B"),
            c: GpuBuffer::zeros(stream, m_max * n).expect("C"),
            n,
        }
    }

    fn prefix_bits(&self, ctx: &GpuCtx, m: usize) -> Vec<u32> {
        let full = self.c.to_cpu(&ctx.stream).expect("C D2H");
        full[..m.min(CMP_ROWS) * self.n]
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
}

/// Typed fixture: the same operands quantized once to `dt` (RNE via the
/// device upload), so every M sees identical 16-bit input bytes.
struct TypedFixture {
    a: DtypedBuf,
    b: DtypedBuf,
    c: DtypedBuf,
    c_elems: usize,
    n: usize,
}

impl TypedFixture {
    fn new(ctx: &GpuCtx, k: usize, n: usize, dt: WeightDtype) -> Self {
        let m_max = *M_LADDER.last().expect("ladder non-empty");
        // The EXPOSING probe (established by census_matvec_vs_thin16):
        // +-4096 at stride 4 with x512 spikes between them. Adjacent-pair
        // cancellation is useless here - contiguous-ascending partitions
        // cancel the pairs exactly in every association - so the pattern
        // interleaves signs and magnitudes at co-prime strides instead.
        // This is what turned the matvec/TC family difference from
        // invisible to 0.1-0.3% of elements.
        let mut a_host = synth(m_max * k, 0xA5EED ^ (k * n) as u64);
        for (i, v) in a_host.iter_mut().enumerate() {
            match i % 4 {
                0 => *v = 4096.0,
                1 => *v = -4096.0,
                2 => *v *= 512.0,
                _ => {}
            }
        }
        let stream = &ctx.stream;
        let a = DtypedBuf::zeros(stream, m_max * k, dt).expect("A");
        a.upload_f32(stream, &a_host).expect("A up");
        let b = DtypedBuf::zeros(stream, k * n, dt).expect("B");
        b.upload_f32(stream, &synth(k * n, 0xB0B ^ n as u64))
            .expect("B up");
        let c = DtypedBuf::zeros(stream, m_max * n, dt).expect("C");
        TypedFixture {
            a,
            b,
            c,
            c_elems: m_max * n,
            n,
        }
    }

    fn prefix_bits(&self, ctx: &GpuCtx, m: usize) -> Vec<u32> {
        // download_f32 widens losslessly and injectively: equal f32 bits
        // <=> equal 16-bit bits. This IS the raw-byte compare.
        let mut full = vec![0.0f32; self.c_elems];
        self.c.download_f32(&ctx.stream, &mut full).expect("C D2H");
        full[..m.min(CMP_ROWS) * self.n]
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
}

fn ctx_new() -> (GpuDevice, GpuCtx) {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    (dev, ctx)
}

const SHAPES: &[(usize, usize)] = &[
    (64, 64),
    (63, 128),
    (384, 384),
    (1024, 384),
    (384, 1928),
    (768, 384),
];

/// Inference family, f32 FFMA tile: batch-invariant BY CONSTRUCTION — the
/// declaration is Strict, and this test is what turns that phrase from
/// prose into a gate (its only prior test was tolerance-based).
#[test]
#[ignore = "needs a CUDA device"]
fn inference_f32_is_strictly_invariant() {
    let (_dev, ctx) = ctx_new();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Inference);
    for &(k, n) in SHAPES {
        let fx = Fixture::new(&ctx, k, n);
        let mut launch = |m: usize| -> Vec<u32> {
            gemm_bi_forward_raw(
                &ctx,
                TypedPtr {
                    ptr: fx.c.raw_ptr(&ctx.stream),
                    dtype: WeightDtype::F32,
                },
                TypedPtr {
                    ptr: fx.a.raw_ptr(&ctx.stream),
                    dtype: WeightDtype::F32,
                },
                TypedPtr {
                    ptr: fx.b.raw_ptr(&ctx.stream),
                    dtype: WeightDtype::F32,
                },
                None,
                (m, k, n),
            )
            .expect("inference f32 forward");
            fx.prefix_bits(&ctx, m)
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_strict("Inference/f32", k, n, &obs);
        println!("Inference/f32   K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// Inference family, WMMA bf16 tile (contract C3): also declared Strict — one
/// CTA per tile, fixed K sequence, no M anywhere in the dispatch.
#[test]
#[ignore = "needs a CUDA device"]
fn inference_bf16_is_strictly_invariant() {
    let (_dev, ctx) = ctx_new();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Inference);
    for &(k, n) in &[(64usize, 64usize), (384, 384), (768, 2304)] {
        let fx = TypedFixture::new(&ctx, k, n, WeightDtype::Bf16);
        let mut launch = |m: usize| -> Vec<u32> {
            gemm_bi_forward_raw(
                &ctx,
                TypedPtr {
                    ptr: fx.c.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: fx.a.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: fx.b.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                None,
                (m, k, n),
            )
            .expect("inference bf16 forward");
            fx.prefix_bits(&ctx, m)
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_strict("Inference/bf16", k, n, &obs);
        println!("Inference/bf16  K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// Triad family, f32: strict on every architecture. Each layer folds the
/// same way at every row count, cut layers in their pieces, so no row
/// count moves a row's bits.
#[test]
#[ignore = "needs a CUDA device"]
fn triad_f32_is_strictly_invariant() {
    let (_dev, ctx) = ctx_new();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    for &(k, n) in SHAPES {
        let m_max = *M_LADDER.last().expect("ladder non-empty");
        let mut a_host = synth(m_max * k, 0xA5EED ^ (k * n) as u64);
        for (i, slot) in a_host[..k].iter_mut().enumerate() {
            *slot = ((i % 17) as f32 - 8.0) * 0.03;
        }
        let a = GpuBuffer::from_cpu(&ctx.stream, &a_host).expect("A");
        let b = GpuBuffer::from_cpu(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64)).expect("B");
        let mut c = GpuBuffer::zeros(&ctx.stream, m_max * n).expect("C");
        let mut launch = |m: usize| -> Vec<u32> {
            gpu_gemm_bi_forward_raw(&ctx, &mut c, &a, b.raw_ptr(&ctx.stream), None, (m, k, n))
                .expect("triad f32 forward");
            let full = c.to_cpu(&ctx.stream).expect("C D2H");
            full[..m.min(CMP_ROWS) * n]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_strict("Triad/f32", k, n, &obs);
        println!("Triad/f32   K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// The typed route (what the mixed forwards and the typed prefill call):
/// strict across all M with the tensor cores on and off. The layer alone
/// picks the arithmetic: the tile ladder (Thin16/Tile64/Tile128, the same
/// bits per element) for N >= 32 under the tensor cores, the scalar entry
/// for N >= 32 without them, and matvec for N < 32 on either tier.
///
/// The probe matters: adjacent +x/-x pairs sum exactly in any contiguous
/// ascending K-partition, and row 0 alone samples ~0.1% of the output, so
/// the fixture interleaves signs and magnitudes and the walk compares the
/// whole 128-row prefix. A row-0 walk once passed over the live M=128
/// matvec/TC break.
#[test]
#[ignore = "needs a CUDA device"]
fn typed_route_is_strictly_invariant_on_both_tiers() {
    let (_dev, ctx) = ctx_new();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    for tensor_cores in [true, false] {
        ctx.route_controls().set_tensor_cores(tensor_cores);
        let family = if tensor_cores {
            "Typed-route/bf16+tc"
        } else {
            "Typed-route/bf16"
        };
        for &(k, n) in &[
            (384usize, 384usize),
            (768, 2304),
            (1024, 384),
            (96, 40),
            (384, 16),
            (2048, 8),
        ] {
            let fx = TypedFixture::new(&ctx, k, n, WeightDtype::Bf16);
            let mut launch = |m: usize| -> Vec<u32> {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    TypedPtr {
                        ptr: fx.c.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    TypedPtr {
                        ptr: fx.a.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    TypedPtr {
                        ptr: fx.b.cached_ptr(),
                        dtype: WeightDtype::Bf16,
                    },
                    None,
                    (m, k, n),
                )
                .expect("typed route forward");
                fx.prefix_bits(&ctx, m)
            };
            let obs = observed_boundaries(&mut launch, k, n);
            assert_strict(family, k, n, &obs);
            println!("{family:<20} K={k:<5} N={n:<5} boundaries: {obs:?}");
        }
    }
}

/// Row-position invariance on the Strict family: the probe vector placed
/// at rows {0, 1, m/2, m-1} of one launch must produce identical bits at
/// every position (L2: bits depend on the row's vector, not its index).
#[test]
#[ignore = "needs a CUDA device"]
fn inference_f32_row_position_invariance() {
    let (_dev, ctx) = ctx_new();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Inference);
    let (k, n, m) = (384usize, 384usize, 129usize);
    let probe: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let positions = [0usize, 1, m / 2, m - 1];

    let mut reference: Option<Vec<u32>> = None;
    for &pos in &positions {
        let mut a_host = synth(m * k, 0x77 ^ pos as u64);
        a_host[pos * k..(pos + 1) * k].copy_from_slice(&probe);
        let a = GpuBuffer::from_cpu(&ctx.stream, &a_host).expect("A");
        let b = GpuBuffer::from_cpu(&ctx.stream, &synth(k * n, 0xB0B ^ n as u64)).expect("B");
        let mut c = GpuBuffer::zeros(&ctx.stream, m * n).expect("C");
        gemm_bi_forward_raw(
            &ctx,
            TypedPtr {
                ptr: c.raw_ptr(&ctx.stream),
                dtype: WeightDtype::F32,
            },
            TypedPtr {
                ptr: a.raw_ptr(&ctx.stream),
                dtype: WeightDtype::F32,
            },
            TypedPtr {
                ptr: b.raw_ptr(&ctx.stream),
                dtype: WeightDtype::F32,
            },
            None,
            (m, k, n),
        )
        .expect("inference f32 forward");
        let full = c.to_cpu(&ctx.stream).expect("C D2H");
        let bits: Vec<u32> = full[pos * n..(pos + 1) * n]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        match &reference {
            None => reference = Some(bits),
            Some(r) => assert_eq!(
                *r, bits,
                "probe row bits changed with its position (pos={pos}) — \
                 row-index leakage into the arithmetic"
            ),
        }
        let _ = &mut c;
    }
    println!(
        "Inference/f32 position invariance: {} positions identical",
        positions.len()
    );
}
