//! The FIXED family - the standalone deterministic INFERENCE kernel.
//!
//! Owns the fast NN-forward tile ladder in `kernels/gemm_bi_fixed.cu`
//! (the GBF tiles, byte-identical to each other per output element),
//! plus the legacy 64x64 WMMA tile as the narrow-N fallback and the f32
//! FFMA tile. One dispatcher, shape-keyed:
//!
//!   f32                  -> FFMA 64x64 tile (all M; the shipped serve
//!                           route - its bits do not move here)
//!   bf16/f16, N >= 32    -> the GBF ladder (Tc16 / Tc64 / Tc128), one
//!                           arithmetic family, STRICT across all M
//!   bf16/f16, N <  32    -> the legacy WMMA tile (all M, strict)
//!
//! The N-keyed split is a legal shape key; M keys only the ladder's
//! bit-identical rungs. Training never enters here - the triad family is
//! the training kernel's home.

use cudarc::driver::PushKernelArg;

use super::blas::TypedPtr;
use super::buffers::DtypedBuf;
use super::context::GpuCtx;
use super::dtype::WeightDtype;

type CUptr = cudarc::driver::sys::CUdeviceptr;

/// Which inference tile actually launched - returned so callers and
/// tests can assert launch reality; a kernel that silently never fires
/// must be impossible to miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedTile {
    /// 128x128 CTA, 256 threads, 2-stage cp.async, dynamic smem 71 680 B.
    Tc128,
    /// 128x256 CTA, 256 threads, 64x64 warp tiles (fragment reuse),
    /// XOR-swizzled dynamic smem 98 304 B. Bit-identical to Tc128.
    TcWn64,
    /// 64x64 CTA, 128 threads, static smem.
    Tc64,
    /// 16x32 thin/decode rung, 128 threads, 4-stage cp.async.
    Tc16,
    /// Legacy 64x64x32 WMMA tile - the narrow-N (< 32) fallback.
    Legacy,
    /// Hopper wgmma rung (sm_90a only): the arch's own numeric family.
    Sm90Wgmma,
    /// Datacenter-Blackwell tcgen05 rung (CC 10.x only): the arch's own
    /// numeric family.
    Sm100Tcgen,
}

/// Underfill threshold: below this many 128-tiles the 64-tile grid wins
/// (four times the CTAs at the same total FLOPs keep a wide GPU busy).
/// The rungs are bit-identical, so the pick can never change bits.
const TC64_PREFER_MAX_TILES128: u32 = 72;

/// Frozen per-architecture wave anchor for the wide-rung pick below.
/// A scheduling key over bit-identical rungs, never an arithmetic one;
/// the same frozen-cell law as the scalar tier's split anchor.
const LADDER_NUM_SMS: u32 = 142;

fn wave_eff(tiles: u32) -> f64 {
    let waves = tiles.div_ceil(LADDER_NUM_SMS);
    f64::from(tiles) / f64::from(waves * LADDER_NUM_SMS)
}

/// Shape-keyed pick for the bf16/f16 inference ladder. `None` = below
/// the ladder's N floor (the caller routes to the legacy tile).
/// The reduction depth `k` keys only the wide fragment-reuse rung - a
/// legal key like every other, since the rungs are bit-identical.
fn fixed_pick_tile(rows: usize, cols: usize, k: usize) -> Option<FixedTile> {
    if cols < 32 {
        return None;
    }
    if rows <= 64 || cols < 64 {
        return Some(FixedTile::Tc16);
    }
    if rows >= 128 && cols >= 128 {
        let tiles128 = (rows as u32).div_ceil(128) * (cols as u32).div_ceil(128);
        if tiles128 >= TC64_PREFER_MAX_TILES128 {
            // The wide 128x256 rung wins its census band by pure wave
            // arithmetic: its half-count grid must hold the 128-tile's
            // wave efficiency (the fragment-reuse bonus is then a free
            // 7-12 percent), and it collapses whenever its grid lands
            // just past a wave boundary. The K/N edge excises the one
            // deep-K cell the equal-efficiency rule mispredicts. Every
            // condition is drawn from the 24-point promotion grid in
            // the census, not from a model.
            if cols >= 1536 && (k <= 768 || cols <= 2304) {
                let tiles_wn = (rows as u32).div_ceil(128) * (cols as u32).div_ceil(256);
                if wave_eff(tiles_wn) >= wave_eff(tiles128) {
                    return Some(FixedTile::TcWn64);
                }
            }
            return Some(FixedTile::Tc128);
        }
    }
    Some(FixedTile::Tc64)
}

fn ladder_cfg(tile: FixedTile, rows: usize, cols: usize) -> cudarc::driver::LaunchConfig {
    let (bm, bn, threads, dyn_bytes) = match tile {
        FixedTile::Tc128 => (128u32, 128u32, 256u32, 71_680u32),
        FixedTile::TcWn64 => (128, 256, 256, 98_304),
        FixedTile::Tc64 => (64, 64, 128, 0),
        FixedTile::Tc16 => (16, 32, 128, 0),
        FixedTile::Sm90Wgmma => (64, 128, 128, 49_152),
        FixedTile::Sm100Tcgen => (128, 128, 128, 65_536),
        FixedTile::Legacy => unreachable!("legacy tile has its own launcher"),
    };
    let grid = (rows as u32).div_ceil(bm) * (cols as u32).div_ceil(bn);
    cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: dyn_bytes,
    }
}

/// Arguments shared by every fixed-family kernel (row-major NN).
struct FixedArgs {
    c: CUptr,
    a: CUptr,
    b: CUptr,
    bias: CUptr,
    m: i32,
    n: i32,
    k: i32,
}

fn launch_ladder(
    ctx: &GpuCtx,
    tile: FixedTile,
    dt: WeightDtype,
    args: &FixedArgs,
) -> Result<(), String> {
    let func = match tile {
        FixedTile::Tc128 => ctx.kernels.gemm_bi_nn_tc128_typed.get(dt),
        FixedTile::TcWn64 => ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt),
        FixedTile::Tc64 => ctx.kernels.gemm_bi_nn_tc64_typed.get(dt),
        FixedTile::Tc16 => ctx.kernels.gemm_bi_nn_tc16_typed.get(dt),
        FixedTile::Sm90Wgmma => ctx
            .kernels
            .gemm_bi_nn_sm90_typed
            .as_ref()
            .ok_or("wgmma rung not loaded on this arch")?
            .get(dt),
        FixedTile::Sm100Tcgen => ctx
            .kernels
            .gemm_bi_nn_sm100_typed
            .as_ref()
            .ok_or("tcgen05 rung not loaded on this arch")?
            .get(dt),
        FixedTile::Legacy => unreachable!("legacy tile has its own launcher"),
    };
    let cfg = ladder_cfg(tile, args.m as usize, args.n as usize);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let mut b = ctx.stream.launch_builder(func);
    b.arg(&args.c);
    b.arg(&args.a);
    b.arg(&args.b);
    b.arg(&args.bias);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&args.m);
    b.arg(&args.n);
    b.arg(&args.k);
    b.arg(&args.k);
    b.arg(&args.n);
    b.arg(&args.n);
    unsafe { b.launch(cfg) }.map_err(|e| format!("gemm_bi fixed ladder ({tile:?}): {e:?}"))?;
    Ok(())
}

/// One-time verdict for an architecture-specific rung: enabled, and
/// sane on this very device. The portable mma.sync ladder has run its
/// bit censuses on real silicon; the Hopper and datacenter-Blackwell
/// rungs may reach a customer's device before ours, so their first use
/// runs a tolerance probe against the portable ladder (they are
/// separate numeric families, so byte equality is not expected - but a
/// staging or descriptor defect is orders of magnitude, not ulps). A
/// failed probe disables the rung for the process and says so loudly;
/// MAMBA_RS_ARCH_RUNG=off disables it up front. Run-to-run and replay
/// bit identity hold either way: the verdict is fixed at first use.
static ARCH_RUNG_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn arch_rung_enabled(ctx: &GpuCtx, tile: FixedTile) -> bool {
    *ARCH_RUNG_OK.get_or_init(|| {
        if std::env::var("MAMBA_RS_ARCH_RUNG").is_ok_and(|v| v.trim() == "off") {
            eprintln!("gemm_bi: architecture rung disabled by MAMBA_RS_ARCH_RUNG=off");
            return false;
        }
        match arch_rung_self_check(ctx, tile) {
            Ok(()) => true,
            Err(e) => {
                eprintln!(
                    "gemm_bi: architecture rung FAILED its self-check and is \
                     disabled for this process ({e}); the portable ladder serves \
                     instead"
                );
                false
            }
        }
    })
}

fn arch_rung_self_check(ctx: &GpuCtx, tile: FixedTile) -> Result<(), String> {
    let (m, k, n) = (256usize, 512usize, 512usize);
    let st = &ctx.stream;
    let dt = WeightDtype::Bf16;
    let mk = |seed: u64, len: usize| -> Result<DtypedBuf, String> {
        let mut s = seed;
        let host: Vec<f32> = (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFF) as f32 / 65536.0) - 0.5
            })
            .collect();
        let b = DtypedBuf::zeros(st, len, dt)?;
        b.upload_f32(st, &host)?;
        Ok(b)
    };
    let a = mk(0x5EED1, m * k)?;
    let b = mk(0x5EED2, k * n)?;
    let c_ref = DtypedBuf::zeros(st, m * n, dt)?;
    let c_arch = DtypedBuf::zeros(st, m * n, dt)?;
    let run = |tile_sel: FixedTile, c: &DtypedBuf| -> Result<(), String> {
        let args = FixedArgs {
            c: c.cached_ptr(),
            a: a.cached_ptr(),
            b: b.cached_ptr(),
            bias: 0,
            m: m as i32,
            n: n as i32,
            k: k as i32,
        };
        launch_ladder(ctx, tile_sel, dt, &args)
    };
    run(FixedTile::Tc128, &c_ref)?;
    run(tile, &c_arch)?;
    st.synchronize()
        .map_err(|e| format!("self-check sync: {e:?}"))?;
    let mut hr = vec![0.0f32; m * n];
    c_ref.download_f32(st, &mut hr)?;
    let mut ha = vec![0.0f32; m * n];
    c_arch.download_f32(st, &mut ha)?;
    let mut worst = 0.0f32;
    for (r, x) in hr.iter().zip(&ha) {
        let rel = (r - x).abs() / (r.abs() + 1e-3);
        if rel > worst {
            worst = rel;
        }
    }
    if worst > 1e-2 {
        return Err(format!(
            "worst relative deviation {worst:.3e} vs the ladder"
        ));
    }
    Ok(())
}

/// The fixed family's NN forward: `C[M,N] = A[M,K] @ B[K,N] (+ bias)`,
/// deterministic and batch-invariant for every covered operand triple.
/// Returns the tile that actually launched.
pub fn fixed_forward(
    ctx: &GpuCtx,
    c: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: Option<CUptr>,
    dims: (usize, usize, usize),
) -> Result<FixedTile, String> {
    let (batch, n_in, n_out) = dims;
    let args = FixedArgs {
        c: c.ptr,
        a: x.ptr,
        b: w.ptr,
        bias: bias_ptr.unwrap_or(0),
        m: batch as i32,
        n: n_out as i32,
        k: n_in as i32,
    };
    let homogeneous_half = c.dtype != WeightDtype::F32 && c.dtype == x.dtype && x.dtype == w.dtype;
    // Architecture rungs: on Hopper and datacenter Blackwell the arch's
    // own tensor path is the numeric family for every eligible shape
    // (per-arch families are the documented law; batch invariance holds
    // inside each). Unaligned operands fall back to the portable
    // ladder, whose kernels carry their own scalar staging.
    if homogeneous_half && n_out >= 32 {
        let aligned = n_in.is_multiple_of(8)
            && n_out.is_multiple_of(8)
            && x.ptr.is_multiple_of(16)
            && w.ptr.is_multiple_of(16);
        let arch_tile = if ctx.kernels.gemm_bi_nn_sm100_typed.is_some() {
            Some(FixedTile::Sm100Tcgen)
        } else if ctx.kernels.gemm_bi_nn_sm90_typed.is_some() {
            Some(FixedTile::Sm90Wgmma)
        } else {
            None
        };
        if let Some(tile) = arch_tile
            && aligned
            && arch_rung_enabled(ctx, tile)
        {
            launch_ladder(ctx, tile, c.dtype, &args)?;
            return Ok(tile);
        }
    }
    if homogeneous_half && let Some(tile) = fixed_pick_tile(batch, n_out, n_in) {
        launch_ladder(ctx, tile, c.dtype, &args)?;
        return Ok(tile);
    }
    // Narrow-N half triples and every f32/mixed triple: the legacy
    // fixed-tile kernel (64x64x32, strict, no buckets) covers all M/N.
    let _ = args;
    super::blas::fixed_legacy_forward(ctx, c, x, w, bias_ptr, dims)?;
    Ok(FixedTile::Legacy)
}
