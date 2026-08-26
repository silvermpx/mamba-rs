//! S1 — THE cross-M bitwise invariance matrix (0.6.10 program, report
//! §7.2 turned into a gate).
//!
//! The law under test (docs-to-be invariance contract, L2): the bits of
//! output row `r` depend only on `A[r, :]`, the whole `B` operand and the
//! plan — never on `M`, never on `r`, never on which other rows shared
//! the launch.
//!
//! The load-bearing part is the VERDICT TABLE: each reachable family
//! declares its invariance class (`Strict` or `Bucketed`), the test
//! derives the OBSERVED boundary set (the ladder values where the probe
//! row's bits change) and asserts it against the declaration. A new
//! M-keyed heuristic, a moved bucket edge, or a silently removed one all
//! fail; removing a boundary legitimately (the G4 unification) requires
//! updating the contract row in the same commit.
//!
//! The typed route's declared boundary {128} is the LIVE defect this
//! encodes: `blas.rs` switches matvec -> typed-sgemm at M=128, two
//! different reduction graphs. This suite documents it today and becomes
//! its regression gate when the unification removes it.
//!
//! No absolute bits are captured — the suite asserts relations only, so
//! it stays green across a legitimate arithmetic re-golden (that is
//! S4's job, not this file's).
//!
//! Run on the GPU box (`GpuCtx` is !Sync — one ctx per test fn):
//!   cargo test --release --features cuda --test gemm_bi_invariance_matrix \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, gemm_bi_forward_raw, gpu_gemm_typed_forward_raw, gpu_sgemm_forward_raw,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

/// The reduced M ladder: every entry sits on (or adjacent to) a known
/// dispatch edge — matvec row-block 4, FRAG_M 16, ultra-thin 32, tile 64,
/// THE typed break 128, slim-force 512, split-K cap 1024, the prism page.
const M_LADDER: &[usize] = &[
    1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025,
    2048, 4621,
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Invariance {
    /// No boundary anywhere on the ladder — any bit change across M fails.
    Strict,
    /// Boundaries allowed ONLY at the declared first-m-of-new-bucket set.
    Bucketed(&'static [usize]),
}

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
/// PROBE ROW's output bits (row 0 — the row whose vector never changes).
type LaunchFn<'a> = dyn FnMut(usize) -> Vec<u32> + 'a;

/// Walk the ladder, return the observed boundary set: every ladder m at
/// which the probe row's bits differ from the previous ladder m.
fn observed_boundaries(launch: &mut LaunchFn<'_>, k: usize, n: usize) -> Vec<usize> {
    let mut prev: Option<Vec<u32>> = None;
    let mut bounds = Vec::new();
    for &m in M_LADDER {
        let bits = launch(m);
        assert_eq!(bits.len(), n, "probe row width at m={m} k={k}");
        if let Some(p) = &prev
            && *p != bits
        {
            bounds.push(m);
        }
        prev = Some(bits);
    }
    bounds
}

fn assert_contract(family: &str, k: usize, n: usize, declared: Invariance, observed: &[usize]) {
    match declared {
        Invariance::Strict => assert!(
            observed.is_empty(),
            "{family} (K={k} N={n}) declared STRICT but the probe row's bits \
             changed at M in {observed:?} — an M-keyed arithmetic switch"
        ),
        Invariance::Bucketed(edges) => {
            assert_eq!(
                observed, edges,
                "{family} (K={k} N={n}) declared bucket edges {edges:?} but \
                 observed {observed:?} — a new, moved or removed boundary \
                 must update the contract row in the SAME commit"
            );
        }
    }
}

/// Shared fixture: A[M_MAX, K] with a fixed probe vector in row 0 and
/// deterministic noise elsewhere, B[K, N]; C sized for the largest M.
struct Fixture {
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
    k: usize,
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
            k,
            n,
        }
    }

    fn row0_bits(&self, ctx: &GpuCtx) -> Vec<u32> {
        let full = self.c.to_cpu(&ctx.stream).expect("C D2H");
        full[..self.n].iter().map(|v| v.to_bits()).collect()
    }
}

/// Typed fixture: the same operands quantized once to `dt` (RNE via the
/// device upload), so every M sees identical 16-bit input bytes.
struct TypedFixture {
    a: DtypedBuf,
    b: DtypedBuf,
    c: DtypedBuf,
    c_elems: usize,
    k: usize,
    n: usize,
}

impl TypedFixture {
    fn new(ctx: &GpuCtx, k: usize, n: usize, dt: WeightDtype) -> Self {
        let m_max = *M_LADDER.last().expect("ladder non-empty");
        let mut a_host = synth(m_max * k, 0xA5EED ^ (k * n) as u64);
        for (i, slot) in a_host[..k].iter_mut().enumerate() {
            *slot = ((i % 17) as f32 - 8.0) * 0.03;
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
            k,
            n,
        }
    }

    fn row0_bits(&self, ctx: &GpuCtx) -> Vec<u32> {
        // download_f32 widens losslessly and injectively: equal f32 bits
        // <=> equal 16-bit bits. This IS the raw-byte compare.
        let mut full = vec![0.0f32; self.c_elems];
        self.c.download_f32(&ctx.stream, &mut full).expect("C D2H");
        full[..self.n].iter().map(|v| v.to_bits()).collect()
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

/// Fixed family, f32 FFMA tile: batch-invariant BY CONSTRUCTION — the
/// declaration is Strict, and this test is what turns that phrase from
/// prose into a gate (its only prior test was tolerance-based).
#[test]
#[ignore = "needs a CUDA device"]
fn fixed_f32_is_strictly_invariant() {
    let (_dev, ctx) = ctx_new();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
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
            .expect("fixed f32 forward");
            fx.row0_bits(&ctx)
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_contract("Fixed/f32", k, n, Invariance::Strict, &obs);
        println!("Fixed/f32   K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// Fixed family, WMMA bf16 tile (contract C3): also declared Strict — one
/// CTA per tile, fixed K sequence, no M anywhere in the dispatch.
#[test]
#[ignore = "needs a CUDA device"]
fn fixed_bf16_is_strictly_invariant() {
    let (_dev, ctx) = ctx_new();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
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
            .expect("fixed bf16 forward");
            fx.row0_bits(&ctx)
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_contract("Fixed/bf16", k, n, Invariance::Strict, &obs);
        println!("Fixed/bf16  K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// Triad family, f32: per-bucket invariance. The declaration is the
/// honest documentation of the dispatcher: any boundary OUTSIDE the
/// ladder's known-edge set is a new M-keyed heuristic and fails. Exact
/// per-(K,N) set equality (the dispatcher mirror predicate) is the
/// S1-final form; the subset gate already catches new and moved edges.
#[test]
#[ignore = "needs a CUDA device"]
fn triad_f32_boundaries_stay_on_known_edges() {
    let (_dev, ctx) = ctx_new();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    // First m of a new bucket can only be one of these (ultra-thin exit,
    // tile/narrow gates, split-K entry and cap, slim-force).
    const KNOWN_EDGES: &[usize] = &[16, 32, 33, 64, 65, 128, 129, 512, 513, 1024, 1025, 2048];
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
            gpu_sgemm_forward_raw(&ctx, &mut c, &a, b.raw_ptr(&ctx.stream), None, (m, k, n))
                .expect("triad f32 forward");
            let full = c.to_cpu(&ctx.stream).expect("C D2H");
            full[..n].iter().map(|v| v.to_bits()).collect()
        };
        let obs = observed_boundaries(&mut launch, k, n);
        for b in &obs {
            assert!(
                KNOWN_EDGES.contains(b),
                "Triad/f32 (K={k} N={n}): boundary at M={b} is not a known \
                 dispatch edge — a new M-keyed heuristic entered the triad"
            );
        }
        println!("Triad/f32   K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// The typed BI route (what the m3 mixed forward and the future typed
/// prefill call): matvec below M=128, typed triad at and above it — two
/// arithmetic graphs. Declared Bucketed({128}) — THE live defect, pinned.
/// The G4 unification flips this row to Strict in the same commit that
/// removes the branch.
#[test]
#[ignore = "needs a CUDA device"]
fn typed_route_break_is_exactly_m128() {
    let (_dev, ctx) = ctx_new();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    for &(k, n) in &[(384usize, 384usize), (768, 2304)] {
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
            fx.row0_bits(&ctx)
        };
        let obs = observed_boundaries(&mut launch, k, n);
        assert_contract("Typed-route/bf16", k, n, Invariance::Bucketed(&[128]), &obs);
        println!("Typed/bf16  K={k:<5} N={n:<5} boundaries: {obs:?}");
    }
}

/// Row-position invariance on the Strict family: the probe vector placed
/// at rows {0, 1, m/2, m-1} of one launch must produce identical bits at
/// every position (L2: bits depend on the row's vector, not its index).
#[test]
#[ignore = "needs a CUDA device"]
fn fixed_f32_row_position_invariance() {
    let (_dev, ctx) = ctx_new();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
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
        .expect("fixed f32 forward");
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
        "Fixed/f32 position invariance: {} positions identical",
        positions.len()
    );
}
