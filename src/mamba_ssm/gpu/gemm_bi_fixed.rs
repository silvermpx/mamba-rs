//! The FIXED family - the standalone deterministic INFERENCE kernel.
//!
//! Owns the fast NN-forward tile ladder in `kernels/gemm_bi_fixed.cu`
//! (the GBF tiles, byte-identical to each other per output element),
//! plus the legacy 64x64 WMMA tile as the narrow-N fallback and the f32
//! FFMA tile. One dispatcher, shape-keyed:
//!
//!   exact f32            -> FFMA 64x128 at two measured CC12.0/170-SM
//!                           points; portable FFMA 64x64 everywhere else
//!   bf16/f16, N >= 32    -> the GBF ladder (Tc16 / Tc64 / Tc128), one
//!                           arithmetic family, STRICT across all M
//!   bf16/f16, N <  32    -> the legacy WMMA tile (all M, strict)
//!
//! The N-keyed split is a legal shape key; M keys only the ladder's
//! bit-identical rungs. Training never enters here - the triad family is
//! the training kernel's home.

use std::collections::VecDeque;

use cudarc::driver::{DeviceRepr, PushKernelArg, sys};

use super::blas::TypedPtr;
use super::buffers::{DtypedBuf, ManagedAllocationEpochStamp, managed_allocation_epoch_for_ranges};
use super::context::GpuCtx;
use super::dtype::WeightDtype;

type CUptr = cudarc::driver::sys::CUdeviceptr;

/// Which inference tile actually launched - returned so callers and
/// tests can assert launch reality; a kernel that silently never fires
/// must be impossible to miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedTile {
    /// Exact-f32 64x128 CTA, two-stage async mainloop.
    F32N128S2,
    /// Portable deterministic TF32, 128x64 CTA, two-stage mainloop.
    Tf32M128S2,
    /// Portable deterministic TF32, 128x64 CTA, three-stage mainloop.
    Tf32M128S3,
    /// Force-only Triad TF32 wide tile, 128x128 CTA, three-stage mainloop.
    /// Uses the wide route's half-ulp operand conversion, including its NaN behavior.
    Tf32M128N128S3,
    /// Portable deterministic TF32, 64x64 CTA, two-stage mainloop.
    Tf32M64S2,
    /// Portable deterministic TF32, 64x64 CTA, three-stage mainloop.
    Tf32M64S3,
    /// Portable deterministic TF32 thin/decode tile, 16x32 CTA.
    Tf32M16S4,
    /// SM120 TMA TF32, 128x64 CTA, two-stage mainloop.
    Tf32Sm120M128S2,
    /// SM120 TMA TF32, 128x64 CTA, three-stage mainloop.
    Tf32Sm120M128S3,
    /// SM120 TMA TF32, 64x128 CTA, two-stage mainloop.
    Tf32Sm120M64N128S2,
    /// SM120 TMA TF32, 64x128 CTA, three-stage mainloop.
    Tf32Sm120M64N128S3,
    /// SM120 TMA TF32, 64x64 CTA with a dedicated producer warp.
    Tf32Sm120M64S2ProducerWarp,
    /// SM120 TMA TF32, 64x64 CTA, two-stage mainloop.
    Tf32Sm120M64S2,
    /// SM120 TMA BF16/F16 route selected from the qualified tile matrix.
    Sm120Half(FixedSm120HalfTile),
    /// 128x128 CTA, 256 threads, 2-stage cp.async, dynamic smem 71 680 B.
    Tc128,
    /// 128x256 CTA, 256 threads, 64x64 warp tiles (fragment reuse),
    /// XOR-swizzled dynamic smem 98 304 B. Bit-identical to Tc128.
    TcWn64,
    /// 128x128 CTA, 128 threads, 64x64 warp tiles (fragment reuse).
    TcW64,
    /// 64x64 CTA, 128 threads, static smem.
    Tc64,
    /// 16x32 thin/decode rung, 128 threads, 4-stage cp.async.
    Tc16,
    /// Portable fallback: exact-f32 64x64 FFMA or narrow-N half WMMA.
    Legacy,
    /// Hopper wgmma rung (sm_90a only): the arch's own numeric family.
    Sm90Wgmma,
    /// Datacenter-Blackwell tcgen05 rung (CC 10.x only): the arch's own
    /// numeric family.
    Sm100Tcgen,
}

/// Forced SM120 TMA BF16/F16 tile used by qualification and dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FixedSm120HalfTile {
    M64N64Bk64S2,
    M64N128Bk64S2,
    M128N64Bk32S3,
    M128N128Bk32S2,
    M128N128Bk32S3,
}

impl FixedSm120HalfTile {
    pub const ALL: [Self; 5] = [
        Self::M64N64Bk64S2,
        Self::M64N128Bk64S2,
        Self::M128N64Bk32S3,
        Self::M128N128Bk32S2,
        Self::M128N128Bk32S3,
    ];

    const fn geometry(self) -> (u32, u32, u32, u32) {
        match self {
            Self::M64N64Bk64S2 => (64, 64, 64, 2),
            Self::M64N128Bk64S2 => (64, 128, 64, 2),
            Self::M128N64Bk32S3 => (128, 64, 32, 3),
            Self::M128N128Bk32S2 => (128, 128, 32, 2),
            Self::M128N128Bk32S3 => (128, 128, 32, 3),
        }
    }

    const fn threads(self) -> u32 {
        let (m, n, bk, _) = self.geometry();
        if m == 128 && n == 128 && bk == 32 {
            256
        } else {
            (m / 32) * (n / 32) * 32
        }
    }

    const fn shared_mem_bytes(self) -> u32 {
        let (m, n, bk, stages) = self.geometry();
        128 + stages * (m + n) * bk * 2
    }
}

/// Operands for a forced Fixed-family qualification launch.
#[derive(Clone, Copy)]
pub struct FixedFwdOperands {
    pub c: TypedPtr,
    pub x: TypedPtr,
    pub w: TypedPtr,
    pub bias_ptr: Option<CUptr>,
}

/// Logical row-major NN shape for a Fixed-family launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Launch one explicitly selected Fixed-family rung for qualification.
pub fn fixed_forward_with_tile(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> Result<(), String> {
    if tile == FixedTile::Legacy {
        return super::blas::fixed_legacy_forward(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            operands.bias_ptr,
            (shape.m, shape.k, shape.n),
        );
    }
    if tile == FixedTile::F32N128S2 {
        if operands.c.dtype != WeightDtype::F32
            || operands.x.dtype != WeightDtype::F32
            || operands.w.dtype != WeightDtype::F32
        {
            return Err("forced Fixed F32 N128 launch requires f32 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_f32_n128_s2(ctx, &args);
    }
    if matches!(
        tile,
        FixedTile::Tf32M128S2
            | FixedTile::Tf32M128S3
            | FixedTile::Tf32M128N128S3
            | FixedTile::Tf32M64S2
            | FixedTile::Tf32M64S3
            | FixedTile::Tf32M16S4
            | FixedTile::Tf32Sm120M128S2
            | FixedTile::Tf32Sm120M128S3
            | FixedTile::Tf32Sm120M64N128S2
            | FixedTile::Tf32Sm120M64N128S3
            | FixedTile::Tf32Sm120M64S2ProducerWarp
            | FixedTile::Tf32Sm120M64S2
    ) {
        if operands.c.dtype != WeightDtype::F32
            || operands.x.dtype != WeightDtype::F32
            || operands.w.dtype != WeightDtype::F32
        {
            return Err("forced Fixed TF32 launch requires f32 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_tf32(ctx, tile, &args, false);
    }
    if let FixedTile::Sm120Half(sm120_tile) = tile {
        let half_inputs =
            operands.x.dtype != WeightDtype::F32 && operands.x.dtype == operands.w.dtype;
        let supported_output =
            operands.c.dtype == WeightDtype::F32 || operands.c.dtype == operands.x.dtype;
        if !half_inputs || !supported_output {
            return Err(
                "forced Fixed SM120 half launch requires bf16/f16 inputs and matching-half or f32 output"
                    .into(),
            );
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_sm120_half(ctx, sm120_tile, operands.x.dtype, operands.c.dtype, &args);
    }
    if !matches!(
        tile,
        FixedTile::Tc16 | FixedTile::Tc64 | FixedTile::Tc128 | FixedTile::TcW64 | FixedTile::TcWn64
    ) {
        return Err(format!("forced Fixed tile {tile:?} is not qualified"));
    }
    let half_inputs = operands.x.dtype != WeightDtype::F32 && operands.x.dtype == operands.w.dtype;
    let mixed_f32_output = operands.c.dtype == WeightDtype::F32;
    if !half_inputs
        || (!mixed_f32_output && operands.c.dtype != operands.x.dtype)
        || (mixed_f32_output
            && !matches!(tile, FixedTile::Tc16 | FixedTile::Tc64 | FixedTile::Tc128))
    {
        return Err("forced Fixed tensor-core launch has an unsupported dtype/tile pair".into());
    }
    let args = FixedArgs::try_new(operands, shape)?;
    if mixed_f32_output {
        launch_f32out_ladder(ctx, tile, operands.x.dtype, &args)
    } else {
        launch_ladder(ctx, tile, operands.c.dtype, &args)
    }
}

#[cfg(test)]
fn launch_sm120_tf32_custom_ldc_for_test(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    ldc: usize,
    pair_store: bool,
) -> Result<(), String> {
    if operands.c.dtype != WeightDtype::F32
        || operands.x.dtype != WeightDtype::F32
        || operands.w.dtype != WeightDtype::F32
    {
        return Err("Fixed SM120 TF32 custom-ldc test launch requires f32 operands".into());
    }
    if ldc < shape.n {
        return Err("Fixed SM120 TF32 custom-ldc test launch requires ldc >= N".into());
    }
    let ldc = i32::try_from(ldc).map_err(|_| "Fixed SM120 TF32 custom-ldc test ldc exceeds i32")?;
    let args = FixedArgs::try_new(operands, shape)?;
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    let kernels = ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .ok_or("Fixed SM120 TF32 kernels are not loaded")?;
    let rows = u32::try_from(args.m).map_err(|_| "Fixed SM120 TF32 M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed SM120 TF32 N is negative")?;
    let grid = rows
        .div_ceil(64)
        .checked_mul(cols.div_ceil(64))
        .ok_or("Fixed SM120 TF32 launch grid exceeds u32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 32_896,
    };
    let maps = if args.k == 0 {
        let zero = unsafe {
            FixedTensorMap(std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed().assume_init())
        };
        [zero, zero]
    } else {
        fixed_sm120_tf32_maps(ctx, &args, 64)?
    };
    let params = FixedSm120Tf32Params {
        m: args.m,
        k: args.k,
        n: args.n,
        ldc,
    };
    let function = if pair_store {
        &kernels.m64n64_s2_pair_store
    } else {
        &kernels.m64n64_s2
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&maps[0]);
    builder.arg(&maps[1]);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 TF32 custom-ldc test: {error:?}"))
}

/// Launch the former exact-f32 kernel for qualification against the
/// production two-stage route.
#[doc(hidden)]
pub fn fixed_forward_f32_legacy_baseline(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
) -> Result<(), String> {
    if operands.c.dtype != WeightDtype::F32
        || operands.x.dtype != WeightDtype::F32
        || operands.w.dtype != WeightDtype::F32
    {
        return Err("Fixed f32 legacy baseline requires f32 operands".into());
    }
    if shape.m == 0 || shape.n == 0 {
        return Ok(());
    }
    let m = i32::try_from(shape.m).map_err(|_| "Fixed M exceeds i32")?;
    let n = i32::try_from(shape.n).map_err(|_| "Fixed N exceeds i32")?;
    let k = i32::try_from(shape.k).map_err(|_| "Fixed K exceeds i32")?;
    let grid_m = u32::try_from(shape.m.div_ceil(64)).map_err(|_| "Fixed M grid exceeds u32")?;
    let grid_n = u32::try_from(shape.n.div_ceil(64)).map_err(|_| "Fixed N grid exceeds u32")?;
    let grid = grid_m
        .checked_mul(grid_n)
        .ok_or("Fixed launch grid exceeds u32")?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let bias = operands.bias_ptr.unwrap_or(0);
    let mut builder = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_f32_f32);
    builder.arg(&operands.c.ptr);
    builder.arg(&operands.x.ptr);
    builder.arg(&operands.w.ptr);
    builder.arg(&bias);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&m);
    builder.arg(&n);
    builder.arg(&k);
    builder.arg(&k);
    builder.arg(&n);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi exact-f32 legacy baseline launch: {error:?}"))
}

/// Underfill threshold: below this many 128-tiles the 64-tile grid wins
/// (four times the CTAs at the same total FLOPs keep a wide GPU busy).
/// The rungs are bit-identical, so the pick can never change bits.
const TC64_PREFER_MAX_TILES128: u32 = 72;

fn wave_eff(tiles: u32, multiprocessors: u32) -> f64 {
    let waves = tiles.div_ceil(multiprocessors);
    f64::from(tiles) / f64::from(waves * multiprocessors)
}

#[derive(Clone, Copy)]
struct FixedTileDevice {
    multiprocessors: u32,
    compute_capability: (u32, u32),
}

/// Shape-keyed pick for the bf16/f16 inference ladder.
/// The reduction depth `k` keys only the wide fragment-reuse rung - a
/// legal key like every other, since the rungs are bit-identical.
fn fixed_pick_tile(
    rows: usize,
    cols: usize,
    k: usize,
    device: FixedTileDevice,
) -> Option<FixedTile> {
    if cols < 32 {
        return Some(FixedTile::Tc16);
    }
    if rows <= 64 || cols < 64 {
        return Some(FixedTile::Tc16);
    }
    let thin_tiles = (rows as u32).div_ceil(16) * (cols as u32).div_ceil(32);
    // The compact SM89 layout keeps four Tc16 CTAs resident. Paired
    // measurements admit one full residency wave, except for the
    // 1537..2047-column band where the square tile crosses earlier.
    let thin_tile_limit = if device.compute_capability == (8, 9) {
        let wave_numerator = if (1537..2048).contains(&cols) { 11 } else { 16 };
        device
            .multiprocessors
            .saturating_mul(wave_numerator)
            .div_ceil(4)
    } else {
        device.multiprocessors.saturating_mul(2)
    };
    if thin_tiles <= thin_tile_limit {
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
                if wave_eff(tiles_wn, device.multiprocessors)
                    >= wave_eff(tiles128, device.multiprocessors)
                {
                    return Some(FixedTile::TcWn64);
                }
            }
            return Some(FixedTile::Tc128);
        }
    }
    Some(FixedTile::Tc64)
}

fn fixed_pick_f32out_tile(
    rows: usize,
    k: usize,
    cols: usize,
    device: FixedTileDevice,
) -> FixedTile {
    if cols < 32 {
        return FixedTile::Tc16;
    }
    let multiprocessors = device.multiprocessors.max(1);
    let thin_tiles = (rows as u32).div_ceil(16) * (cols as u32).div_ceil(32);
    let thin_residency = if device.compute_capability == (8, 9)
        || super::device::is_sm120_family(device.compute_capability)
    {
        4
    } else {
        2
    };
    if thin_tiles <= multiprocessors.saturating_mul(thin_residency) {
        return FixedTile::Tc16;
    }

    if device.compute_capability == (12, 0)
        && device.multiprocessors == 170
        && (rows, k, cols) == (4621, 1928, 384)
    {
        return FixedTile::Tc64;
    }

    let tiles64 = (rows as u32).div_ceil(64) * (cols as u32).div_ceil(64);
    let waves64 = tiles64.div_ceil(multiprocessors.saturating_mul(2));
    let tiles128 = (rows as u32).div_ceil(128) * (cols as u32).div_ceil(128);
    let waves128 = tiles128.div_ceil(multiprocessors);
    if rows >= 128 && cols >= 128 && waves64 >= waves128.saturating_mul(2) {
        FixedTile::Tc128
    } else {
        FixedTile::Tc64
    }
}

fn fixed_adjust_arch_tile(tile: FixedTile, k: usize, compute_capability: (u32, u32)) -> FixedTile {
    if super::device::is_sm120_family(compute_capability) && tile == FixedTile::Tc128 && k >= 1024 {
        FixedTile::TcW64
    } else {
        tile
    }
}

fn ladder_cfg(tile: FixedTile, rows: usize, cols: usize) -> cudarc::driver::LaunchConfig {
    let (bm, bn, threads, dyn_bytes) = match tile {
        FixedTile::Tc128 => (128u32, 128u32, 256u32, 71_680u32),
        FixedTile::TcW64 => (128, 128, 128, 65_536),
        FixedTile::TcWn64 => (128, 256, 256, 98_304),
        FixedTile::Tc64 => (64, 64, 128, 0),
        FixedTile::Tc16 => (16, 32, 128, 0),
        FixedTile::Sm90Wgmma => (64, 128, 128, 49_152),
        FixedTile::Sm100Tcgen => (128, 128, 128, 65_536),
        FixedTile::Legacy
        | FixedTile::F32N128S2
        | FixedTile::Tf32M128S2
        | FixedTile::Tf32M128S3
        | FixedTile::Tf32M128N128S3
        | FixedTile::Tf32M64S2
        | FixedTile::Tf32M64S3
        | FixedTile::Tf32M16S4
        | FixedTile::Tf32Sm120M128S2
        | FixedTile::Tf32Sm120M128S3
        | FixedTile::Tf32Sm120M64N128S2
        | FixedTile::Tf32Sm120M64N128S3
        | FixedTile::Tf32Sm120M64S2ProducerWarp
        | FixedTile::Tf32Sm120M64S2
        | FixedTile::Sm120Half(_) => unreachable!("tile has its own launcher"),
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

impl FixedArgs {
    fn try_new(operands: FixedFwdOperands, shape: FixedShape) -> Result<Self, String> {
        Ok(Self {
            c: operands.c.ptr,
            a: operands.x.ptr,
            b: operands.w.ptr,
            bias: operands.bias_ptr.unwrap_or(0),
            m: i32::try_from(shape.m).map_err(|_| "Fixed M exceeds i32")?,
            n: i32::try_from(shape.n).map_err(|_| "Fixed N exceeds i32")?,
            k: i32::try_from(shape.k).map_err(|_| "Fixed K exceeds i32")?,
        })
    }
}

fn launch_f32_n128_s2(ctx: &GpuCtx, args: &FixedArgs) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    let rows = u32::try_from(args.m).map_err(|_| "Fixed F32 N128 M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed F32 N128 N is negative")?;
    let grid = rows
        .div_ceil(64)
        .checked_mul(cols.div_ceil(128))
        .ok_or("Fixed F32 N128 launch grid exceeds u32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let lda = args.k;
    let ldb = args.n;
    let ldc = args.n;
    let mut builder = ctx
        .stream
        .launch_builder(&ctx.kernels.gemm_bi_f32_f32_n128_s2);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&args.m);
    builder.arg(&args.n);
    builder.arg(&args.k);
    builder.arg(&lda);
    builder.arg(&ldb);
    builder.arg(&ldc);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi forced exact-f32 N128 S2: {error:?}"))
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FixedTf32Params {
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for FixedTf32Params {}

pub(crate) const FIXED_TF32_PARAMS_SIZE: usize = std::mem::size_of::<FixedTf32Params>();

/// The borrowed Triad symbol takes alpha/beta before the six dimensions;
/// its 32-byte parameter block is not the 24-byte Fixed TF32 ABI above.
#[derive(Clone, Copy)]
#[repr(C)]
struct FixedTf32WideParams {
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for FixedTf32WideParams {}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct FixedTensorMap(sys::CUtensorMap);

unsafe impl DeviceRepr for FixedTensorMap {}

pub(crate) const FIXED_TENSOR_MAP_SIZE: usize = std::mem::size_of::<sys::CUtensorMap>();
pub(crate) const FIXED_TENSOR_MAP_ALIGN: usize = std::mem::align_of::<sys::CUtensorMap>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedTf32MapKey {
    a: CUptr,
    b: CUptr,
    m: usize,
    k: usize,
    n: usize,
    tile_rows: u32,
}

struct FixedTf32MapEntry {
    key: FixedTf32MapKey,
    maps: [FixedTensorMap; 2],
    epoch: ManagedAllocationEpochStamp,
}

#[derive(Default)]
pub(crate) struct FixedTf32MapCache {
    entries: VecDeque<FixedTf32MapEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedHalfMapKey {
    a: CUptr,
    b: CUptr,
    m: usize,
    k: usize,
    n: usize,
    tile: FixedSm120HalfTile,
}

struct FixedHalfMapEntry {
    key: FixedHalfMapKey,
    maps: [FixedTensorMap; 2],
    epoch: ManagedAllocationEpochStamp,
}

#[derive(Default)]
pub(crate) struct FixedHalfMapCache {
    entries: VecDeque<FixedHalfMapEntry>,
}

impl FixedHalfMapCache {
    fn get(&mut self, key: FixedHalfMapKey) -> Option<[FixedTensorMap; 2]> {
        self.entries.retain(|entry| entry.epoch.is_current());
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        let entry = self.entries.remove(index)?;
        let maps = entry.maps;
        self.entries.push_front(entry);
        Some(maps)
    }

    fn insert(
        &mut self,
        key: FixedHalfMapKey,
        maps: [FixedTensorMap; 2],
        epoch: ManagedAllocationEpochStamp,
    ) {
        self.entries.retain(|entry| entry.key != key);
        self.entries
            .push_front(FixedHalfMapEntry { key, maps, epoch });
    }
}

impl FixedTf32MapCache {
    fn get(&mut self, key: FixedTf32MapKey) -> Option<[FixedTensorMap; 2]> {
        self.entries.retain(|entry| entry.epoch.is_current());
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        let entry = self.entries.remove(index)?;
        let maps = entry.maps;
        self.entries.push_front(entry);
        Some(maps)
    }

    fn insert(
        &mut self,
        key: FixedTf32MapKey,
        maps: [FixedTensorMap; 2],
        epoch: ManagedAllocationEpochStamp,
    ) {
        self.entries.retain(|entry| entry.key != key);
        self.entries
            .push_front(FixedTf32MapEntry { key, maps, epoch });
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FixedSm120Tf32Params {
    m: i32,
    k: i32,
    n: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for FixedSm120Tf32Params {}

pub(crate) const FIXED_SM120_TF32_PARAMS_SIZE: usize = std::mem::size_of::<FixedSm120Tf32Params>();

#[derive(Clone, Copy)]
#[repr(C)]
struct FixedSm120HalfParams {
    a_x: i32,
    a_y: i32,
    b_x: i32,
    b_y: i32,
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for FixedSm120HalfParams {}

pub(crate) const FIXED_SM120_HALF_PARAMS_SIZE: usize = std::mem::size_of::<FixedSm120HalfParams>();

fn encode_fixed_tf32_map(
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
) -> Result<FixedTensorMap, String> {
    if base == 0 || !base.is_multiple_of(16) {
        return Err("Fixed SM120 TF32 map base must be non-null and 16-byte aligned".into());
    }
    if global_dimensions.contains(&0)
        || outer_byte_stride == 0
        || !outer_byte_stride.is_multiple_of(16)
        || outer_byte_stride >= (1_u64 << 40)
    {
        return Err("Fixed SM120 TF32 map dimensions or stride are unsupported".into());
    }
    let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
    let global_strides = [outer_byte_stride];
    let element_strides = [1_u32, 1_u32];
    unsafe {
        sys::cuTensorMapEncodeTiled(
            raw.as_mut_ptr(),
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
            2,
            base as usize as *mut std::ffi::c_void,
            global_dimensions.as_ptr(),
            global_strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()
        .map_err(|error| format!("Fixed SM120 TF32 tensor-map encode: {error:?}"))?;
        Ok(FixedTensorMap(raw.assume_init()))
    }
}

fn fixed_sm120_tf32_maps(
    ctx: &GpuCtx,
    args: &FixedArgs,
    tile_rows: u32,
) -> Result<[FixedTensorMap; 2], String> {
    let key = FixedTf32MapKey {
        a: args.a,
        b: args.b,
        m: args.m as usize,
        k: args.k as usize,
        n: args.n as usize,
        tile_rows,
    };
    if let Some(maps) = ctx.fixed_tf32_maps.borrow_mut().get(key) {
        return Ok(maps);
    }
    if ctx
        .stream
        .capture_status()
        .map_err(|error| format!("query Fixed TF32 capture status: {error:?}"))?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("Fixed SM120 TF32 tensor maps must be prepared before graph capture".into());
    }
    let m = u64::try_from(args.m).map_err(|_| "Fixed TF32 M is negative")?;
    let k = u64::try_from(args.k).map_err(|_| "Fixed TF32 K is negative")?;
    let n = u64::try_from(args.n).map_err(|_| "Fixed TF32 N is negative")?;
    let a_stride = k.checked_mul(4).ok_or("Fixed TF32 A stride overflows")?;
    let b_stride = n.checked_mul(4).ok_or("Fixed TF32 B stride overflows")?;
    let maps = [
        encode_fixed_tf32_map(args.a, [k, m], a_stride, [32, tile_rows])?,
        encode_fixed_tf32_map(args.b, [n, k], b_stride, [32, 32])?,
    ];
    let a_bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or("Fixed TF32 A span overflows")?;
    let b_bytes = k
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or("Fixed TF32 B span overflows")?;
    let context_handle = ctx.stream.context().cu_ctx() as usize;
    if let Some(epoch) =
        managed_allocation_epoch_for_ranges(context_handle, &[(args.a, a_bytes), (args.b, b_bytes)])
    {
        ctx.fixed_tf32_maps.borrow_mut().insert(key, maps, epoch);
    }
    Ok(maps)
}

fn encode_fixed_half_map(
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
    swizzle: sys::CUtensorMapSwizzle,
) -> Result<FixedTensorMap, String> {
    if base == 0 || !base.is_multiple_of(16) {
        return Err("Fixed SM120 half map base must be non-null and 16-byte aligned".into());
    }
    let row_bytes = global_dimensions[0]
        .checked_mul(2)
        .ok_or("Fixed SM120 half map row width overflows")?;
    if global_dimensions.contains(&0)
        || box_dimensions.contains(&0)
        || outer_byte_stride < row_bytes
        || !outer_byte_stride.is_multiple_of(16)
        || outer_byte_stride >= (1_u64 << 40)
    {
        return Err("Fixed SM120 half map dimensions or stride are unsupported".into());
    }
    let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
    let global_strides = [outer_byte_stride];
    let element_strides = [1_u32, 1_u32];
    unsafe {
        sys::cuTensorMapEncodeTiled(
            raw.as_mut_ptr(),
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
            2,
            base as usize as *mut std::ffi::c_void,
            global_dimensions.as_ptr(),
            global_strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()
        .map_err(|error| format!("Fixed SM120 half tensor-map encode: {error:?}"))?;
        Ok(FixedTensorMap(raw.assume_init()))
    }
}

fn fixed_sm120_half_maps(
    ctx: &GpuCtx,
    args: &FixedArgs,
    tile: FixedSm120HalfTile,
) -> Result<[FixedTensorMap; 2], String> {
    let key = FixedHalfMapKey {
        a: args.a,
        b: args.b,
        m: args.m as usize,
        k: args.k as usize,
        n: args.n as usize,
        tile,
    };
    if let Some(maps) = ctx.fixed_half_maps.borrow_mut().get(key) {
        return Ok(maps);
    }
    if ctx
        .stream
        .capture_status()
        .map_err(|error| format!("query Fixed half capture status: {error:?}"))?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("Fixed SM120 half tensor maps must be prepared before graph capture".into());
    }
    let m = u64::try_from(args.m).map_err(|_| "Fixed half M is negative")?;
    let k = u64::try_from(args.k).map_err(|_| "Fixed half K is negative")?;
    let n = u64::try_from(args.n).map_err(|_| "Fixed half N is negative")?;
    let (tile_m, tile_n, bk, _) = tile.geometry();
    let a_stride = k.checked_mul(2).ok_or("Fixed half A stride overflows")?;
    let b_stride = n.checked_mul(2).ok_or("Fixed half B stride overflows")?;
    let a_swizzle = if bk == 32 {
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
    } else {
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    };
    let special_wide_b = tile_m == 128 && tile_n == 64 && bk == 32;
    let b_box = if special_wide_b { [64, 32] } else { [bk, bk] };
    let b_swizzle = if special_wide_b || bk == 64 {
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    } else {
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
    };
    let maps = [
        encode_fixed_half_map(args.a, [k, m], a_stride, [bk, tile_m], a_swizzle)?,
        encode_fixed_half_map(args.b, [n, k], b_stride, b_box, b_swizzle)?,
    ];
    let a_bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or("Fixed half A span overflows")?;
    let b_bytes = k
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or("Fixed half B span overflows")?;
    let context_handle = ctx.stream.context().cu_ctx() as usize;
    if let Some(epoch) =
        managed_allocation_epoch_for_ranges(context_handle, &[(args.a, a_bytes), (args.b, b_bytes)])
    {
        ctx.fixed_half_maps.borrow_mut().insert(key, maps, epoch);
    }
    Ok(maps)
}

fn fixed_pick_tf32(
    rows: usize,
    inner: usize,
    cols: usize,
    multiprocessors: u32,
    compute_capability: (u32, u32),
    nvrtc_version: (i32, i32),
    sm120_tma: bool,
) -> FixedTile {
    if rows <= 16 || cols <= 32 {
        return FixedTile::Tf32M16S4;
    }
    if sm120_tma {
        if compute_capability == (12, 0) && multiprocessors == 170 && nvrtc_version == (13, 2) {
            match (rows, inner, cols) {
                (4621, 768, 2304) => return FixedTile::Tf32Sm120M128S2,
                (2048, 768, 2304) => return FixedTile::Tf32Sm120M64S2ProducerWarp,
                _ => {}
            }
        }
        return FixedTile::Tf32Sm120M64S2;
    }
    if super::device::is_sm120_family(compute_capability) {
        return FixedTile::Tf32M64S2;
    }
    if cols >= 768 {
        return FixedTile::Tf32M64S2;
    }
    let tiles128 = rows.div_ceil(128).saturating_mul(cols.div_ceil(64));
    if tiles128 < multiprocessors as usize {
        FixedTile::Tf32M64S2
    } else {
        FixedTile::Tf32M128S2
    }
}

fn fixed_pick_f32_exact(rows: usize, k: usize, cols: usize, device: FixedTileDevice) -> FixedTile {
    if device.compute_capability == (12, 0)
        && device.multiprocessors == 170
        && matches!((rows, k, cols), (4621, 384, 1928) | (4621, 768, 2304))
    {
        FixedTile::F32N128S2
    } else {
        FixedTile::Legacy
    }
}

fn launch_tf32(
    ctx: &GpuCtx,
    tile: FixedTile,
    args: &FixedArgs,
    allow_schedule_select: bool,
) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    if tile == FixedTile::Tf32M128N128S3 {
        return launch_tf32_wide(ctx, args);
    }
    if matches!(
        tile,
        FixedTile::Tf32Sm120M128S2
            | FixedTile::Tf32Sm120M128S3
            | FixedTile::Tf32Sm120M64N128S2
            | FixedTile::Tf32Sm120M64N128S3
            | FixedTile::Tf32Sm120M64S2ProducerWarp
            | FixedTile::Tf32Sm120M64S2
    ) {
        return launch_sm120_tf32(ctx, tile, args, allow_schedule_select);
    }
    let (function, bm, bn, threads, shared_mem_bytes) = match tile {
        FixedTile::Tf32M128S2 => (
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s2,
            128,
            64,
            256,
            55_296,
        ),
        FixedTile::Tf32M128S3 => (
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s3,
            128,
            64,
            256,
            82_944,
        ),
        FixedTile::Tf32M64S2 => (&ctx.kernels.gemm_bi_nn_tf32.m64n64_s2, 64, 64, 128, 32_768),
        FixedTile::Tf32M64S3 => (&ctx.kernels.gemm_bi_nn_tf32.m64n64_s3, 64, 64, 128, 55_296),
        FixedTile::Tf32M16S4 => (&ctx.kernels.gemm_bi_nn_tf32.m16n32_s4, 16, 32, 128, 29_696),
        _ => return Err(format!("{tile:?} is not a Fixed TF32 tile")),
    };
    let rows = u32::try_from(args.m).map_err(|_| "Fixed TF32 M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed TF32 N is negative")?;
    let grid = rows
        .div_ceil(bm)
        .checked_mul(cols.div_ceil(bn))
        .ok_or("Fixed TF32 launch grid exceeds u32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let params = FixedTf32Params {
        m: args.m,
        k: args.k,
        n: args.n,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed TF32 ({tile:?}): {error:?}"))
}

fn launch_tf32_wide(ctx: &GpuCtx, args: &FixedArgs) -> Result<(), String> {
    if args.k % 4 != 0 || args.n % 4 != 0 {
        return Err("Fixed TF32 wide requires K and N divisible by four".into());
    }
    // The wide body has vector-only input staging. K=0 does not issue any
    // input copies and may use null A/B; C and bias use scalar-safe stores/loads.
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
    {
        return Err("Fixed TF32 wide requires non-null, 16-byte-aligned A and B".into());
    }
    if args.c == 0 || !args.c.is_multiple_of(4) || !args.bias.is_multiple_of(4) {
        return Err("Fixed TF32 wide requires non-null f32-aligned C and f32-aligned bias".into());
    }
    let rows = u32::try_from(args.m).map_err(|_| "Fixed TF32 wide M is negative")?;
    // The CUDA body computes (N + 127) in signed 32-bit arithmetic.
    let padded_columns = args
        .n
        .checked_add(127)
        .ok_or("Fixed TF32 wide padded N exceeds i32")?;
    let column_tiles =
        u32::try_from(padded_columns).map_err(|_| "Fixed TF32 wide N is negative")? / 128;
    let grid = rows
        .div_ceil(128)
        .checked_mul(column_tiles)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed TF32 wide launch grid exceeds i32")?;
    // The loader already admits this exact function's driver ABI, registers,
    // local memory and occupancy. No module is loaded or reconfigured here.
    let function = ctx
        .kernels
        .tf32_function("gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3")
        .ok_or("Fixed TF32 wide Triad symbol is not bound")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 98_304,
    };
    let params = FixedTf32WideParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        k: args.k,
        n: args.n,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi forced Fixed TF32 wide: {error:?}"))
}

fn fixed_sm120_pair_store_schedule_cell(
    dims: (usize, usize, usize),
    compute_capability: (u32, u32),
    multiprocessors: u32,
    nvrtc_version: (i32, i32),
    output_aligned: bool,
) -> bool {
    if compute_capability != (12, 0) || multiprocessors != 170 || !output_aligned {
        return false;
    }
    match nvrtc_version {
        (13, 0) => matches!(
            dims,
            (4621, 384, 1928)
                | (4621, 768, 2304)
                | (4621, 1928, 384)
                | (2048, 768, 2304)
                | (2048, 2304, 768)
        ),
        (12, 8) => matches!(dims, (2048, 768, 2304) | (2048, 2304, 768)),
        _ => false,
    }
}

fn launch_sm120_tf32(
    ctx: &GpuCtx,
    tile: FixedTile,
    args: &FixedArgs,
    allow_schedule_select: bool,
) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    let kernels = ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .ok_or("Fixed SM120 TF32 kernels are not loaded")?;
    let (function, bm, bn, threads, shared_mem_bytes) = match tile {
        FixedTile::Tf32Sm120M128S2 => (&kernels.m128n64_s2, 128, 64, 128, 49_280),
        FixedTile::Tf32Sm120M128S3 => (&kernels.m128n64_s3, 128, 64, 256, 73_856),
        FixedTile::Tf32Sm120M64N128S2 => (&kernels.m64n128_s2, 64, 128, 128, 49_280),
        FixedTile::Tf32Sm120M64N128S3 => (&kernels.m64n128_s3, 64, 128, 256, 73_856),
        FixedTile::Tf32Sm120M64S2ProducerWarp => {
            (&kernels.m64n64_s2_producer_warp, 64, 64, 160, 32_896)
        }
        FixedTile::Tf32Sm120M64S2 => {
            let pair_store = allow_schedule_select
                && fixed_sm120_pair_store_schedule_cell(
                    (args.m as usize, args.k as usize, args.n as usize),
                    ctx.compute_capability(),
                    ctx.kernels.multiprocessor_count(),
                    ctx.kernels.compiler_identity().nvrtc_version,
                    args.c.is_multiple_of(8),
                );
            let function = if pair_store {
                &kernels.m64n64_s2_pair_store
            } else {
                &kernels.m64n64_s2
            };
            (function, 64, 64, 128, 32_896)
        }
        _ => return Err(format!("{tile:?} is not a Fixed SM120 TF32 tile")),
    };
    let rows = u32::try_from(args.m).map_err(|_| "Fixed SM120 TF32 M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed SM120 TF32 N is negative")?;
    let grid = rows
        .div_ceil(bm)
        .checked_mul(cols.div_ceil(bn))
        .ok_or("Fixed SM120 TF32 launch grid exceeds u32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let maps = if args.k == 0 {
        let zero = unsafe {
            FixedTensorMap(std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed().assume_init())
        };
        [zero, zero]
    } else {
        fixed_sm120_tf32_maps(ctx, args, bm)?
    };
    let params = FixedSm120Tf32Params {
        m: args.m,
        k: args.k,
        n: args.n,
        ldc: args.n,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&maps[0]);
    builder.arg(&maps[1]);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 TF32 ({tile:?}): {error:?}"))
}

fn fixed_sm120_half_eligible(ctx: &GpuCtx, args: &FixedArgs) -> bool {
    ctx.kernels.gemm_bi_nn_half_sm120.is_some()
        && super::device::is_sm120_family(ctx.compute_capability())
        && args.k > 0
        && args.a.is_multiple_of(16)
        && args.b.is_multiple_of(16)
        && args.k % 8 == 0
        && args.n % 8 == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedSm120HalfExactRequest {
    dims: (usize, usize, usize),
    dtype: WeightDtype,
    output_aligned: bool,
    has_bias: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedSm120HalfExactDevice {
    compute_capability: (u32, u32),
    multiprocessors: u32,
    nvrtc_version: (i32, i32),
}

fn fixed_pick_sm120_half_exact(
    request: FixedSm120HalfExactRequest,
    device: FixedSm120HalfExactDevice,
) -> Option<FixedSm120HalfTile> {
    if device.compute_capability != (12, 0)
        || device.multiprocessors != 170
        || device.nvrtc_version != (13, 2)
        || !request.output_aligned
    {
        return None;
    }
    match (request.dims, request.dtype, request.has_bias) {
        ((4621, 768, 2304), WeightDtype::Bf16 | WeightDtype::F16, _) => {
            Some(FixedSm120HalfTile::M128N128Bk32S3)
        }
        ((2048, 1928, 2304) | (1536, 1032, 1536), WeightDtype::Bf16 | WeightDtype::F16, false) => {
            Some(FixedSm120HalfTile::M128N128Bk32S3)
        }
        (
            (1024, 1928, 1928)
            | (1024, 1928, 2304)
            | (1536, 1928, 1536)
            | (2048, 768, 2304)
            | (3072, 768, 2304)
            | (4621, 768, 1536),
            WeightDtype::F16,
            false,
        ) => Some(FixedSm120HalfTile::M128N128Bk32S3),
        (
            (512, 1928, 2304)
            | (1536, 1928, 2304)
            | (1536, 1928, 1928)
            | (3072, 1928, 1928)
            | (4096, 520, 1536)
            | (2048, 1928, 1536)
            | (1024, 1928, 1928)
            | (1024, 1928, 2304)
            | (1536, 1032, 1536)
            | (1536, 1928, 1536)
            | (4621, 1928, 1928)
            | (1536, 768, 1536),
            WeightDtype::Bf16 | WeightDtype::F16,
            _,
        ) => Some(FixedSm120HalfTile::M64N128Bk64S2),
        _ => None,
    }
}

fn fixed_pick_sm120_half(
    rows: usize,
    cols: usize,
    k: usize,
    multiprocessors: u32,
) -> Option<FixedSm120HalfTile> {
    let row_tiles = rows.div_ceil(64);
    let column_tiles = cols.div_ceil(64);
    let grid = row_tiles.saturating_mul(column_tiles);
    let sms = multiprocessors as usize;
    if cols < 32 || k < 64 || sms == 0 || grid.saturating_mul(4) < sms {
        return None;
    }

    if cols >= 1536 && k > 1024 {
        let b_columns = cols.div_ceil(128);
        let b_grid = rows.div_ceil(128).saturating_mul(b_columns);
        let b_waves = b_grid.div_ceil(sms);
        let c_waves = grid.div_ceil(sms);
        if b_columns == 16 && b_waves == 3 {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        }
        let Some(b_product) = 4_u128.checked_mul(b_grid as u128) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let Some(b_efficiency) = b_product.checked_mul(c_waves as u128) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let Some(c_product) = 3_u128.checked_mul(grid as u128) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let Some(c_efficiency) = c_product.checked_mul(b_waves as u128) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        return Some(if b_efficiency > c_efficiency {
            FixedSm120HalfTile::M128N128Bk32S2
        } else {
            FixedSm120HalfTile::M64N64Bk64S2
        });
    }

    if multiprocessors == 170 && (64..=1024).contains(&k) && matches!(cols, 1536 | 1928 | 2304) {
        let Some(c_grid) = rows.div_ceil(64).checked_mul(cols.div_ceil(64)) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let Some(a_grid) = rows.div_ceil(128).checked_mul(cols.div_ceil(64)) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let Some(b_grid) = rows.div_ceil(128).checked_mul(cols.div_ceil(128)) else {
            return Some(FixedSm120HalfTile::M64N64Bk64S2);
        };
        let sms = multiprocessors as usize;
        let c_waves = c_grid.div_ceil(sms);
        let a_waves = a_grid.div_ceil(sms);
        let b_waves = b_grid.div_ceil(sms);
        // These paired 170-SM wave keys promote only measured A wins; every
        // unqualified topology falls through to the existing shallow policy.
        let paired_a_win = match cols {
            1536 if k <= 512 => matches!((a_waves, c_waves), (2, 4)),
            1536 => matches!((a_waves, c_waves), (2, 4) | (5, 10)),
            1928 if k <= 512 => matches!((a_waves, c_waves), (3, 5) | (3, 6) | (5, 9) | (6, 12)),
            1928 => matches!((a_waves, c_waves), (3, 6) | (6, 12)),
            2304 if k > 512 => matches!((a_waves, c_waves, b_waves), (7, 14, 4)),
            _ => false,
        };
        if paired_a_win {
            return Some(FixedSm120HalfTile::M128N64Bk32S3);
        }
    }

    let medium_grid = grid.saturating_mul(4) >= sms.saturating_mul(5)
        && grid.saturating_mul(2) <= sms.saturating_mul(11);
    let large_grid = grid >= sms.saturating_mul(12);
    if cols >= 2048 && k <= 1024 {
        if large_grid && k > 512 {
            return Some(FixedSm120HalfTile::M128N128Bk32S2);
        }
        if medium_grid || large_grid {
            return Some(FixedSm120HalfTile::M128N64Bk32S3);
        }
    }
    if (1536..2048).contains(&cols)
        && k <= 1024
        && ((grid >= sms && grid.saturating_mul(4) <= sms.saturating_mul(7)) || large_grid)
    {
        return Some(FixedSm120HalfTile::M128N64Bk32S3);
    }
    Some(FixedSm120HalfTile::M64N64Bk64S2)
}

fn fixed_pick_sm120_f32out(
    rows: usize,
    cols: usize,
    k: usize,
    device: FixedSm120HalfExactDevice,
) -> Option<FixedSm120HalfTile> {
    if !super::device::is_sm120_family(device.compute_capability) {
        return None;
    }
    let fallback = fixed_pick_sm120_half(rows, cols, k, device.multiprocessors);
    if device.compute_capability != (12, 0)
        || device.multiprocessors != 170
        || device.nvrtc_version != (13, 2)
    {
        return fallback;
    }
    match (rows, k, cols) {
        (4621, 384, 1928) => Some(FixedSm120HalfTile::M128N64Bk32S3),
        (4621, 768, 2304) => Some(FixedSm120HalfTile::M128N128Bk32S3),
        (4621, 1928, 384) | (2048, 768, 2304) | (2048, 2304, 768) => {
            Some(FixedSm120HalfTile::M64N64Bk64S2)
        }
        // The deep wide projection sits past the band the generic comparator
        // was fitted on (k up to 2304); there it extrapolates to the 128x128
        // tile at 228 us where this tile measures 178 us
        // (internal/perf/sm120-half-tiles-20260903).
        (4096, 3072, 1536) => Some(FixedSm120HalfTile::M64N128Bk64S2),
        _ => fallback,
    }
}

fn launch_sm120_half(
    ctx: &GpuCtx,
    tile: FixedSm120HalfTile,
    input_dtype: WeightDtype,
    output_dtype: WeightDtype,
    args: &FixedArgs,
) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    if input_dtype == WeightDtype::F32
        || !matches!(output_dtype, WeightDtype::F32) && output_dtype != input_dtype
        || !fixed_sm120_half_eligible(ctx, args)
    {
        return Err("Fixed SM120 half route is unavailable for these operands".into());
    }
    let kernels = if output_dtype == WeightDtype::F32 {
        ctx.kernels.gemm_bi_nn_half_sm120_f32out.as_ref()
    } else {
        ctx.kernels.gemm_bi_nn_half_sm120.as_ref()
    }
    .ok_or("Fixed SM120 half kernels are not loaded")?;
    let (bm, bn, _, _) = tile.geometry();
    let rows = u32::try_from(args.m).map_err(|_| "Fixed SM120 half M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed SM120 half N is negative")?;
    let grid = rows
        .div_ceil(bm)
        .checked_mul(cols.div_ceil(bn))
        .ok_or("Fixed SM120 half launch grid exceeds u32")?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (tile.threads(), 1, 1),
        shared_mem_bytes: tile.shared_mem_bytes(),
    };
    let maps = fixed_sm120_half_maps(ctx, args, tile)?;
    let params = FixedSm120HalfParams {
        a_x: 0,
        a_y: 0,
        b_x: 0,
        b_y: 0,
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        k: args.k,
        n: args.n,
        ldc: args.n,
    };
    let function = match tile {
        FixedSm120HalfTile::M64N64Bk64S2 => &kernels.m64n64_bk64_s2,
        FixedSm120HalfTile::M64N128Bk64S2 => &kernels.m64n128_bk64_s2,
        FixedSm120HalfTile::M128N64Bk32S3 => &kernels.m128n64_bk32_s3,
        FixedSm120HalfTile::M128N128Bk32S2 => &kernels.m128n128_bk32_s2,
        FixedSm120HalfTile::M128N128Bk32S3 => &kernels.m128n128_bk32_s3,
    }
    .get(input_dtype);
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&maps[0]);
    builder.arg(&maps[1]);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 half ({tile:?}): {error:?}"))
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
        FixedTile::TcW64 => ctx.kernels.gemm_bi_nn_tcw64_typed.get(dt),
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
        FixedTile::Legacy
        | FixedTile::F32N128S2
        | FixedTile::Tf32M128S2
        | FixedTile::Tf32M128S3
        | FixedTile::Tf32M128N128S3
        | FixedTile::Tf32M64S2
        | FixedTile::Tf32M64S3
        | FixedTile::Tf32M16S4
        | FixedTile::Tf32Sm120M128S2
        | FixedTile::Tf32Sm120M128S3
        | FixedTile::Tf32Sm120M64N128S2
        | FixedTile::Tf32Sm120M64N128S3
        | FixedTile::Tf32Sm120M64S2ProducerWarp
        | FixedTile::Tf32Sm120M64S2
        | FixedTile::Sm120Half(_) => unreachable!("tile has its own launcher"),
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

fn launch_f32out_ladder(
    ctx: &GpuCtx,
    tile: FixedTile,
    input_dtype: WeightDtype,
    args: &FixedArgs,
) -> Result<(), String> {
    let func = match tile {
        FixedTile::Tc16 => ctx.kernels.gemm_bi_nn_tc16_f32out.get(input_dtype),
        FixedTile::Tc64 => ctx.kernels.gemm_bi_nn_tc64_f32out.get(input_dtype),
        FixedTile::Tc128 => ctx.kernels.gemm_bi_nn_tc128_f32out.get(input_dtype),
        _ => unreachable!("mixed-output tile is validated before launch"),
    };
    let cfg = ladder_cfg(tile, args.m as usize, args.n as usize);
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let mut builder = ctx.stream.launch_builder(func);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&args.m);
    builder.arg(&args.n);
    builder.arg(&args.k);
    builder.arg(&args.k);
    builder.arg(&args.n);
    builder.arg(&args.n);
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi fixed {tile:?} f32 output: {error:?}"))
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
        match std::env::var("MAMBA_RS_ARCH_RUNG") {
            Ok(value) if value.trim().eq_ignore_ascii_case("off") => {
                eprintln!("gemm_bi: architecture rung disabled by MAMBA_RS_ARCH_RUNG=off");
                return false;
            }
            Ok(value) if !value.trim().is_empty() => {
                // A context built without the env route reaches here with the
                // flag unchecked; say so rather than run as if it were unset.
                eprintln!(
                    "gemm_bi WARNING: MAMBA_RS_ARCH_RUNG={value:?} is not a recognized \
                     value (only off is); the rung runs its self-check as if unset"
                );
            }
            _ => {}
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
    let operands = FixedFwdOperands { c, x, w, bias_ptr };
    let shape = FixedShape {
        m: batch,
        k: n_in,
        n: n_out,
    };
    let args = FixedArgs::try_new(operands, shape)?;
    let homogeneous_f32 =
        c.dtype == WeightDtype::F32 && x.dtype == WeightDtype::F32 && w.dtype == WeightDtype::F32;
    if homogeneous_f32
        && ctx.f32_triad_policy() == super::context::F32TriadPolicy::AllowDeterministicTf32V1
    {
        let sm120_tma = ctx.kernels.gemm_bi_nn_tf32_sm120.is_some()
            && super::device::is_sm120_family(ctx.compute_capability())
            && n_in > 0
            && x.ptr.is_multiple_of(16)
            && w.ptr.is_multiple_of(16)
            && n_in.is_multiple_of(4)
            && n_out.is_multiple_of(4);
        let tile = fixed_pick_tf32(
            batch,
            n_in,
            n_out,
            ctx.kernels.multiprocessor_count(),
            ctx.compute_capability(),
            ctx.kernels.compiler_identity().nvrtc_version,
            sm120_tma,
        );
        launch_tf32(ctx, tile, &args, true)?;
        return Ok(tile);
    }
    if homogeneous_f32 {
        let tile = fixed_pick_f32_exact(
            batch,
            n_in,
            n_out,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
        );
        if tile == FixedTile::F32N128S2 {
            launch_f32_n128_s2(ctx, &args)?;
        } else {
            super::blas::fixed_legacy_forward(ctx, c, x, w, bias_ptr, dims)?;
        }
        return Ok(tile);
    }
    let homogeneous_half = c.dtype != WeightDtype::F32 && c.dtype == x.dtype && x.dtype == w.dtype;
    let mixed_half_f32 =
        c.dtype == WeightDtype::F32 && x.dtype != WeightDtype::F32 && x.dtype == w.dtype;
    // Architecture rungs: on Hopper and datacenter Blackwell the arch's
    // own tensor path is the numeric family for every eligible shape
    // (per-arch families are the documented law; batch invariance holds
    // inside each). Unaligned operands fall back to the portable
    // ladder, whose kernels carry their own scalar staging.
    if homogeneous_half
        && fixed_sm120_half_eligible(ctx, &args)
        && let Some(tile) = fixed_pick_sm120_half_exact(
            FixedSm120HalfExactRequest {
                dims,
                dtype: c.dtype,
                output_aligned: c.ptr.is_multiple_of(4),
                has_bias: bias_ptr.is_some(),
            },
            FixedSm120HalfExactDevice {
                compute_capability: ctx.compute_capability(),
                multiprocessors: ctx.kernels.multiprocessor_count(),
                nvrtc_version: ctx.kernels.compiler_identity().nvrtc_version,
            },
        )
        .or_else(|| fixed_pick_sm120_half(batch, n_out, n_in, ctx.kernels.multiprocessor_count()))
    {
        launch_sm120_half(ctx, tile, x.dtype, c.dtype, &args)?;
        return Ok(FixedTile::Sm120Half(tile));
    }
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
    if mixed_half_f32 {
        if fixed_sm120_half_eligible(ctx, &args)
            && let Some(tile) = fixed_pick_sm120_f32out(
                batch,
                n_out,
                n_in,
                FixedSm120HalfExactDevice {
                    compute_capability: ctx.compute_capability(),
                    multiprocessors: ctx.kernels.multiprocessor_count(),
                    nvrtc_version: ctx.kernels.compiler_identity().nvrtc_version,
                },
            )
        {
            launch_sm120_half(ctx, tile, x.dtype, c.dtype, &args)?;
            return Ok(FixedTile::Sm120Half(tile));
        }
        let tile = fixed_pick_f32out_tile(
            batch,
            n_in,
            n_out,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
        );
        launch_f32out_ladder(ctx, tile, x.dtype, &args)?;
        return Ok(tile);
    }
    if homogeneous_half
        && let Some(tile) = fixed_pick_tile(
            batch,
            n_out,
            n_in,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
        )
    {
        let tile = fixed_adjust_arch_tile(tile, n_in, ctx.compute_capability());
        launch_ladder(ctx, tile, c.dtype, &args)?;
        return Ok(tile);
    }
    // Unsupported dtype triples retain the legacy implementation.
    let _ = args;
    super::blas::fixed_legacy_forward(ctx, c, x, w, bias_ptr, dims)?;
    Ok(FixedTile::Legacy)
}

#[cfg(test)]
mod pair_store_schedule_tests {
    use super::{
        FixedFwdOperands, FixedShape, fixed_sm120_pair_store_schedule_cell,
        launch_sm120_tf32_custom_ldc_for_test,
    };
    use crate::mamba_ssm::gpu::{
        blas::TypedPtr, buffers::DtypedBuf, context::GpuCtx, device::GpuDevice, dtype::WeightDtype,
    };

    #[test]
    fn pair_store_schedule_is_exactly_the_qualified_cc120_170sm_cells() {
        let cells = [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ];
        for cell in cells {
            assert!(fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0 + 1, cell.1, cell.2),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0 - 1, cell.1, cell.2),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0, cell.1 + 1, cell.2),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0, cell.1 - 1, cell.2),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0, cell.1, cell.2 + 1),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                (cell.0, cell.1, cell.2 - 1),
                (12, 0),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 1),
                170,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 0),
                171,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 0),
                169,
                (13, 0),
                true,
            ));
            assert!(!fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 0),
                170,
                (13, 0),
                false,
            ));
        }
        assert!(!fixed_sm120_pair_store_schedule_cell(
            (4621, 768, 2304),
            (12, 0),
            170,
            (12, 8),
            true,
        ));
        assert!(fixed_sm120_pair_store_schedule_cell(
            (2048, 768, 2304),
            (12, 0),
            170,
            (12, 8),
            true,
        ));
        assert!(fixed_sm120_pair_store_schedule_cell(
            (2048, 2304, 768),
            (12, 0),
            170,
            (12, 8),
            true,
        ));
        for cell in [(4621, 384, 1928), (4621, 1928, 384)] {
            assert!(!fixed_sm120_pair_store_schedule_cell(
                cell,
                (12, 0),
                170,
                (12, 8),
                true,
            ));
        }
        assert!(!fixed_sm120_pair_store_schedule_cell(
            (2048, 768, 2304),
            (12, 0),
            170,
            (12, 7),
            true,
        ));
        assert!(!fixed_sm120_pair_store_schedule_cell(
            (2048, 768, 2304),
            (12, 0),
            170,
            (13, 2),
            true,
        ));
        assert!(!fixed_sm120_pair_store_schedule_cell(
            (2048, 768, 2304),
            (12, 0),
            170,
            (13, 1),
            true,
        ));
    }

    #[test]
    #[ignore = "requires a CC12.0 CUDA device"]
    fn pair_store_loaded_resources_preserve_three_cta_residency() {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        let kernels = ctx
            .kernels
            .gemm_bi_nn_tf32_sm120
            .as_ref()
            .expect("SM120 TF32 holder");
        let resources = |name: &str, function: &cudarc::driver::CudaFunction| {
            let registers = function.num_regs().expect("TF32 registers");
            let local_bytes = function.local_size_bytes().expect("TF32 local bytes");
            let static_shared = function
                .shared_size_bytes()
                .expect("TF32 static shared bytes");
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(128, 32_896, None)
                .expect("TF32 occupancy");
            println!(
                "{name} registers={registers} local_bytes={local_bytes} static_shared={static_shared} dynamic_shared=32896 occupancy={occupancy}"
            );
            (registers, local_bytes, static_shared, occupancy)
        };
        let incumbent = resources("incumbent", &kernels.m64n64_s2);
        let pair_store = resources("pair_store", &kernels.m64n64_s2_pair_store);
        let (registers, local_bytes, static_shared, occupancy) = pair_store;
        assert_eq!(incumbent.1, 0);
        assert_eq!(incumbent.2, 0);
        assert_eq!(local_bytes, 0);
        assert_eq!(static_shared, 0);
        assert!(registers <= 128, "pair-store uses {registers} registers");
        assert!(
            occupancy >= 3,
            "pair-store occupancy {occupancy} is below 3"
        );
        match ctx.kernels.compiler_identity().nvrtc_version {
            (12, 8) => {
                assert!(
                    registers <= incumbent.0 + 4,
                    "pair-store registers {registers} exceed incumbent {} + 4",
                    incumbent.0
                );
                assert!(
                    occupancy >= incumbent.3,
                    "pair-store occupancy {occupancy} is below incumbent {}",
                    incumbent.3
                );
            }
            (13, 0) => assert!(registers <= 96, "pair-store uses {registers} registers"),
            _ => {}
        }
    }

    #[test]
    #[ignore = "requires a CC12.0 CUDA device"]
    fn pair_store_odd_ldc_and_misaligned_output_match_incumbent_bits() {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        let shape = FixedShape {
            m: 65,
            k: 36,
            n: 68,
        };
        let a =
            DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32).expect("odd-ldc A");
        let b =
            DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32).expect("odd-ldc B");
        let values = |len: usize, mut state: u64| {
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state & 0xffff) as f32 / 32768.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        a.upload_f32(&ctx.stream, &values(shape.m * shape.k, 0xa11d_1dc0))
            .expect("odd-ldc A upload");
        b.upload_f32(&ctx.stream, &values(shape.k * shape.n, 0xb11d_1dc0))
            .expect("odd-ldc B upload");
        for (ldc, output_offset) in [(68, 0), (69, 0), (68, 1), (69, 1)] {
            let span = output_offset + shape.m * ldc + 16;
            let incumbent = DtypedBuf::zeros(&ctx.stream, span, WeightDtype::F32)
                .expect("custom-ldc incumbent output");
            let pair_store = DtypedBuf::zeros(&ctx.stream, span, WeightDtype::F32)
                .expect("custom-ldc pair-store output");
            let sentinel = vec![-119.25; span];
            incumbent
                .upload_f32(&ctx.stream, &sentinel)
                .expect("custom-ldc incumbent poison");
            pair_store
                .upload_f32(&ctx.stream, &sentinel)
                .expect("custom-ldc pair-store poison");
            let operands = |output: &DtypedBuf| FixedFwdOperands {
                c: TypedPtr {
                    ptr: output.cached_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64,
                    dtype: WeightDtype::F32,
                },
                x: TypedPtr {
                    ptr: a.cached_ptr(),
                    dtype: WeightDtype::F32,
                },
                w: TypedPtr {
                    ptr: b.cached_ptr(),
                    dtype: WeightDtype::F32,
                },
                bias_ptr: None,
            };
            launch_sm120_tf32_custom_ldc_for_test(&ctx, operands(&incumbent), shape, ldc, false)
                .expect("custom-ldc incumbent launch");
            launch_sm120_tf32_custom_ldc_for_test(&ctx, operands(&pair_store), shape, ldc, true)
                .expect("custom-ldc pair-store launch");
            ctx.stream.synchronize().expect("custom-ldc sync");
            let download = |buffer: &DtypedBuf| {
                let mut host = vec![0.0; span];
                buffer
                    .download_f32(&ctx.stream, &mut host)
                    .expect("custom-ldc download");
                host.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            };
            assert_eq!(
                download(&pair_store),
                download(&incumbent),
                "ldc={ldc} output_offset={output_offset}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FixedHalfMapCache, FixedHalfMapKey, FixedSm120HalfExactDevice, FixedSm120HalfExactRequest,
        FixedSm120HalfTile, FixedTensorMap, FixedTf32MapCache, FixedTf32MapKey, FixedTile,
        FixedTileDevice, fixed_adjust_arch_tile, fixed_pick_f32_exact, fixed_pick_f32out_tile,
        fixed_pick_sm120_f32out, fixed_pick_sm120_half, fixed_pick_sm120_half_exact,
        fixed_pick_tf32, fixed_pick_tile,
    };
    use crate::mamba_ssm::gpu::buffers::{
        managed_allocation_epoch_for_ranges, register_managed_allocation_range,
    };
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::kernel_identity::{
        NUMERIC_ABI_REVISION, SCHEDULE_REVISION, TUNING_TABLE_REVISION,
    };

    fn zero_tensor_maps() -> [FixedTensorMap; 2] {
        let map = unsafe { FixedTensorMap(std::mem::MaybeUninit::zeroed().assume_init()) };
        [map, map]
    }

    #[test]
    fn sm120_half_forced_tile_inventory_is_complete() {
        assert_eq!(FixedSm120HalfTile::ALL.len(), 5);
        assert_eq!(
            FixedSm120HalfTile::M128N128Bk32S3.geometry(),
            (128, 128, 32, 3)
        );
    }

    #[test]
    fn sm120_map_caches_retain_all_live_graph_entries() {
        let context_handle = 0x12_0120;
        let mut half = FixedHalfMapCache::default();
        let mut tf32 = FixedTf32MapCache::default();
        let mut registrations = Vec::new();
        let mut first_half_key = None;
        let mut first_tf32_key = None;

        for index in 0..33_u64 {
            let base = 0x1000_0000 + index * 0x1_0000;
            registrations.push(
                register_managed_allocation_range(context_handle, base, 0x8000)
                    .expect("register live map allocation"),
            );
            let epoch = managed_allocation_epoch_for_ranges(
                context_handle,
                &[(base, 0x1000), (base + 0x4000, 0x1000)],
            )
            .expect("live allocation epoch");
            let half_key = FixedHalfMapKey {
                a: base,
                b: base + 0x4000,
                m: 128,
                k: 96,
                n: 1536,
                tile: FixedSm120HalfTile::M64N64Bk64S2,
            };
            let tf32_key = FixedTf32MapKey {
                a: base,
                b: base + 0x4000,
                m: 128,
                k: 96,
                n: 1536,
                tile_rows: 64,
            };
            first_half_key.get_or_insert(half_key);
            first_tf32_key.get_or_insert(tf32_key);
            half.insert(half_key, zero_tensor_maps(), epoch.clone());
            tf32.insert(tf32_key, zero_tensor_maps(), epoch);
        }

        assert!(
            half.get(first_half_key.expect("first half key")).is_some(),
            "live half tensor maps must survive eager warmup beyond 32 entries"
        );
        assert!(
            tf32.get(first_tf32_key.expect("first TF32 key")).is_some(),
            "live TF32 tensor maps must survive eager warmup beyond 32 entries"
        );
        drop(registrations);
    }

    #[test]
    fn sm120_half_selector_matches_measured_regions() {
        let sms = 170;
        for cols in [8, 16, 24] {
            assert_eq!(
                fixed_pick_sm120_half(8192, cols, 768, sms),
                None,
                "N={cols} must stay in the all-M narrow numeric family"
            );
        }
        assert_eq!(fixed_pick_sm120_half(64, 1536, 768, sms), None);
        assert_eq!(
            fixed_pick_sm120_half(128, 1536, 768, sms),
            Some(FixedSm120HalfTile::M64N64Bk64S2)
        );
        assert_eq!(
            fixed_pick_sm120_half(512, 1536, 384, sms),
            Some(FixedSm120HalfTile::M128N64Bk32S3)
        );
        assert_eq!(
            fixed_pick_sm120_half(1024, 1536, 384, sms),
            Some(FixedSm120HalfTile::M64N64Bk64S2)
        );
        assert_eq!(
            fixed_pick_sm120_half(1024, 2304, 1928, sms),
            Some(FixedSm120HalfTile::M128N128Bk32S2)
        );
        assert_eq!(
            fixed_pick_sm120_half(2048, 2304, 768, sms),
            Some(FixedSm120HalfTile::M64N64Bk64S2)
        );
        assert_eq!(
            fixed_pick_sm120_half(4621, 1928, 384, sms),
            Some(FixedSm120HalfTile::M128N64Bk32S3)
        );
        assert_eq!(
            fixed_pick_sm120_half(4621, 2304, 768, sms),
            Some(FixedSm120HalfTile::M128N128Bk32S2)
        );
        assert_eq!(
            fixed_pick_sm120_half(2048, 2304, 1928, sms),
            Some(FixedSm120HalfTile::M128N128Bk32S2)
        );
    }

    #[test]
    fn sm120_half_exact_overlay_promotes_only_qualified_full_context_cells() {
        use FixedSm120HalfTile::{M64N64Bk64S2 as C, M64N128Bk64S2 as D, M128N64Bk32S3 as A};

        let qualified_device = FixedSm120HalfExactDevice {
            compute_capability: (12, 0),
            multiprocessors: 170,
            nvrtc_version: (13, 2),
        };
        assert_eq!(
            (
                NUMERIC_ABI_REVISION,
                TUNING_TABLE_REVISION,
                SCHEDULE_REVISION,
            ),
            (5, 38, 8),
            "the release compiler identity must remain explicitly pinned"
        );
        let mut promoted = Vec::new();
        for (dims, incumbent) in [
            ((512, 1928, 2304), C),
            ((1536, 1928, 2304), C),
            ((1536, 1928, 1928), C),
            ((3072, 1928, 1928), C),
            ((4096, 520, 1536), A),
            ((2048, 1928, 1536), C),
            ((1024, 1928, 1928), FixedSm120HalfTile::M128N128Bk32S2),
            ((1024, 1928, 2304), FixedSm120HalfTile::M128N128Bk32S2),
            ((1536, 1032, 1536), FixedSm120HalfTile::M128N128Bk32S2),
            ((1536, 1928, 1536), FixedSm120HalfTile::M128N128Bk32S2),
            ((4621, 1928, 1928), FixedSm120HalfTile::M128N128Bk32S2),
            ((1536, 768, 1536), A),
        ] {
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for has_bias in [false, true] {
                    let expected = if !has_bias
                        && (dims == (1536, 1032, 1536)
                            || (dtype == WeightDtype::F16
                                && matches!(
                                    dims,
                                    (1024, 1928, 1928) | (1024, 1928, 2304) | (1536, 1928, 1536)
                                ))) {
                        FixedSm120HalfTile::M128N128Bk32S3
                    } else {
                        D
                    };
                    promoted.push((dims, dtype, has_bias, expected, incumbent));
                }
            }
        }

        let generic = |dims: (usize, usize, usize)| {
            fixed_pick_sm120_half(dims.0, dims.2, dims.1, 170)
                .expect("qualified dimensions have a generic half tile")
        };
        let overlay = |dims, dtype, output_aligned, has_bias, device| {
            fixed_pick_sm120_half_exact(
                FixedSm120HalfExactRequest {
                    dims,
                    dtype,
                    output_aligned,
                    has_bias,
                },
                device,
            )
        };

        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for has_bias in [false, true] {
                assert_eq!(
                    overlay((4621, 768, 2304), dtype, true, has_bias, qualified_device),
                    Some(FixedSm120HalfTile::M128N128Bk32S3),
                    "qualified hot-B S3 context {dtype:?} bias={has_bias}",
                );
            }
            assert_eq!(
                overlay((2048, 1928, 2304), dtype, true, false, qualified_device),
                Some(FixedSm120HalfTile::M128N128Bk32S3),
                "qualified deep S3 context {dtype:?}",
            );
            assert_eq!(
                overlay((2048, 1928, 2304), dtype, true, true, qualified_device),
                None,
                "unmeasured deep S3 bias context {dtype:?}",
            );
        }
        for dims in [(2048, 768, 2304), (3072, 768, 2304), (4621, 768, 1536)] {
            assert_eq!(
                overlay(dims, WeightDtype::F16, true, false, qualified_device),
                Some(FixedSm120HalfTile::M128N128Bk32S3),
                "qualified F16-only S3 context {dims:?}",
            );
        }
        let select = |dims, dtype, output_aligned, has_bias, device| {
            overlay(dims, dtype, output_aligned, has_bias, device).unwrap_or_else(|| generic(dims))
        };

        for (dims, dtype, has_bias, expected, incumbent) in promoted {
            assert_eq!(
                generic(dims),
                incumbent,
                "pre-overlay incumbent for {dims:?}"
            );
            assert_eq!(
                overlay(dims, dtype, true, has_bias, qualified_device),
                Some(expected),
                "qualified exact overlay for {dtype:?} {dims:?}"
            );
            assert_eq!(
                select(dims, dtype, true, has_bias, qualified_device),
                expected,
                "qualified {dtype:?} cell {dims:?}"
            );

            for negative_dims in [
                (dims.0 - 1, dims.1, dims.2),
                (dims.0 + 1, dims.1, dims.2),
                (dims.0, dims.1 - 1, dims.2),
                (dims.0, dims.1 + 1, dims.2),
                (dims.0, dims.1, dims.2 - 1),
                (dims.0, dims.1, dims.2 + 1),
            ] {
                assert_eq!(
                    overlay(negative_dims, dtype, true, has_bias, qualified_device),
                    None,
                    "independent +/-1 exact exclusion for {dtype:?} {negative_dims:?}"
                );
                assert_eq!(
                    select(negative_dims, dtype, true, has_bias, qualified_device),
                    generic(negative_dims),
                    "independent +/-1 dimension fallback for {dtype:?} {negative_dims:?}"
                );
            }

            for device in [
                FixedSm120HalfExactDevice {
                    compute_capability: (12, 1),
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    multiprocessors: 169,
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    multiprocessors: 171,
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    nvrtc_version: (13, 1),
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    nvrtc_version: (13, 0),
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    nvrtc_version: (12, 8),
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    nvrtc_version: (13, 3),
                    ..qualified_device
                },
                FixedSm120HalfExactDevice {
                    nvrtc_version: (0, 0),
                    ..qualified_device
                },
            ] {
                assert_eq!(
                    overlay(dims, dtype, true, has_bias, device),
                    None,
                    "device/compiler exact exclusion for {dtype:?} {dims:?} {device:?}"
                );
                assert_eq!(
                    select(dims, dtype, true, has_bias, device),
                    incumbent,
                    "device/compiler fallback for {dtype:?} {dims:?} {device:?}"
                );
            }

            assert_eq!(
                overlay(dims, dtype, false, has_bias, qualified_device),
                None,
                "C+2 exact exclusion for {dtype:?} {dims:?}"
            );
            assert_eq!(
                select(dims, dtype, false, has_bias, qualified_device),
                incumbent,
                "C+2-byte fallback for {dtype:?} {dims:?}"
            );
            assert_eq!(
                overlay(dims, WeightDtype::F32, true, has_bias, qualified_device,),
                None,
                "F32 exact exclusion for {dims:?}"
            );
            assert_eq!(
                select(dims, WeightDtype::F32, true, has_bias, qualified_device,),
                incumbent,
                "F32 fallback for {dims:?}"
            );
        }

        for (dims, dtype, has_bias, reason) in [
            (
                (3072, 768, 1928),
                WeightDtype::Bf16,
                false,
                "A1 BF16 no bias",
            ),
            ((3072, 768, 1928), WeightDtype::Bf16, true, "A1 BF16 bias"),
            ((3072, 768, 1928), WeightDtype::F16, false, "A1 F16 no bias"),
            ((3072, 768, 1928), WeightDtype::F16, true, "A1 F16 bias"),
            (
                (4096, 384, 1536),
                WeightDtype::Bf16,
                false,
                "A2 BF16 no bias",
            ),
            ((4096, 384, 1536), WeightDtype::Bf16, true, "A2 BF16 bias"),
            ((4096, 384, 1536), WeightDtype::F16, true, "A2 F16 bias"),
            ((4096, 384, 1536), WeightDtype::F16, false, "A2 F16 no bias"),
            (
                (1536, 768, 1928),
                WeightDtype::Bf16,
                false,
                "A3 BF16 no bias",
            ),
            ((1536, 768, 1928), WeightDtype::Bf16, true, "A3 BF16 bias"),
            ((1536, 768, 1928), WeightDtype::F16, false, "A3 F16 no bias"),
            ((1536, 768, 1928), WeightDtype::F16, true, "A3 F16 bias"),
        ] {
            assert_eq!(generic(dims), C, "{reason} generic incumbent");
            assert_eq!(
                overlay(dims, dtype, true, has_bias, qualified_device),
                None,
                "{reason} exact exclusion"
            );
            assert_eq!(
                select(dims, dtype, true, has_bias, qualified_device),
                C,
                "{reason} generic fallback"
            );
        }
    }

    #[test]
    fn sm120_half_deep_selector_matches_measured_170_sm_anchors() {
        let sms = 170;
        let cases = [
            (512, 2304, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (1024, 1536, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (1024, 1928, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (1024, 2304, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (1536, 1536, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (1536, 1928, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (1536, 2304, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (2048, 1536, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (2048, 1928, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (2048, 2304, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (3072, 1536, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (3072, 1928, 1928, FixedSm120HalfTile::M64N64Bk64S2),
            (3072, 2304, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (4621, 1928, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (4621, 2304, 1928, FixedSm120HalfTile::M128N128Bk32S2),
            (1536, 1536, 1032, FixedSm120HalfTile::M128N128Bk32S2),
            (2048, 1536, 1032, FixedSm120HalfTile::M64N64Bk64S2),
        ];

        for (rows, cols, k, expected) in cases {
            assert_eq!(
                fixed_pick_sm120_half(rows, cols, k, sms),
                Some(expected),
                "M={rows}, N={cols}, K={k}"
            );
        }
    }

    #[test]
    fn sm120_half_shallow_selector_matches_measured_170_sm_wave_keys() {
        let a = FixedSm120HalfTile::M128N64Bk32S3;
        let b = FixedSm120HalfTile::M128N128Bk32S2;
        let c = FixedSm120HalfTile::M64N64Bk64S2;
        let cases = [
            (1536, 1536, 384, 170, a),
            (1536, 1536, 768, 170, a),
            (1536, 1928, 384, 170, a),
            (2048, 1928, 384, 170, a),
            (2048, 1928, 768, 170, a),
            (3072, 1928, 384, 170, a),
            (4096, 1536, 768, 170, a),
            (4096, 1928, 384, 170, a),
            (4096, 1928, 768, 170, a),
            (4096, 2304, 768, 170, a),
            (1536, 1928, 768, 170, c),
            (2048, 1536, 384, 170, c),
            (2048, 1536, 768, 170, c),
            (3072, 1928, 768, 170, c),
            (4096, 1536, 384, 170, c),
            (4621, 1536, 384, 170, c),
            (4621, 1536, 768, 170, c),
            (3072, 2304, 768, 170, c),
            (4621, 2304, 768, 170, b),
            (1536, 1928, 512, 170, a),
            (1536, 1928, 520, 170, c),
            (4096, 1536, 512, 170, c),
            (4096, 1536, 520, 170, a),
            (1536, 1536, 768, 169, c),
            (1536, 1536, 768, 171, c),
        ];

        for (rows, cols, k, sms, expected) in cases {
            assert_eq!(
                fixed_pick_sm120_half(rows, cols, k, sms),
                Some(expected),
                "M={rows}, N={cols}, K={k}, SMs={sms}"
            );
        }
    }

    #[test]
    fn exact_f32_selector_promotes_only_measured_cc120_170_sm_points() {
        let measured_device = FixedTileDevice {
            multiprocessors: 170,
            compute_capability: (12, 0),
        };
        for (rows, k, cols) in [(4621, 384, 1928), (4621, 768, 2304)] {
            assert_eq!(
                fixed_pick_f32_exact(rows, k, cols, measured_device),
                FixedTile::F32N128S2,
                "measured point M={rows}, K={k}, N={cols}"
            );
        }

        for (rows, k, cols) in [
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
            (4620, 384, 1928),
            (4622, 384, 1928),
            (4621, 383, 1928),
            (4621, 385, 1928),
            (4621, 384, 1927),
            (4621, 384, 1929),
            (4620, 768, 2304),
            (4622, 768, 2304),
            (4621, 767, 2304),
            (4621, 769, 2304),
            (4621, 768, 2303),
            (4621, 768, 2305),
        ] {
            assert_eq!(
                fixed_pick_f32_exact(rows, k, cols, measured_device),
                FixedTile::Legacy,
                "unmeasured point M={rows}, K={k}, N={cols}"
            );
        }

        for device in [
            FixedTileDevice {
                multiprocessors: 169,
                compute_capability: (12, 0),
            },
            FixedTileDevice {
                multiprocessors: 171,
                compute_capability: (12, 0),
            },
            FixedTileDevice {
                multiprocessors: 170,
                compute_capability: (12, 1),
            },
            FixedTileDevice {
                multiprocessors: 170,
                compute_capability: (8, 9),
            },
        ] {
            assert_eq!(
                fixed_pick_f32_exact(4621, 384, 1928, device),
                FixedTile::Legacy,
                "A must remain Legacy off the measured device"
            );
            assert_eq!(
                fixed_pick_f32_exact(4621, 768, 2304, device),
                FixedTile::Legacy,
                "B must remain Legacy off the measured device"
            );
        }
    }

    #[test]
    fn exact_f32_selector_keeps_cc121_a_and_b_on_legacy() {
        let cc121 = FixedTileDevice {
            multiprocessors: 170,
            compute_capability: (12, 1),
        };
        assert_eq!(
            fixed_pick_f32_exact(4621, 384, 1928, cc121),
            FixedTile::Legacy
        );
        assert_eq!(
            fixed_pick_f32_exact(4621, 768, 2304, cc121),
            FixedTile::Legacy
        );
    }

    #[test]
    fn sm120_tf32_selector_promotes_only_the_qualified_cuda_132_b_and_d_cells() {
        assert_eq!(
            fixed_pick_tf32(4621, 768, 2304, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M128S2
        );
        assert_eq!(
            fixed_pick_tf32(2048, 768, 2304, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2ProducerWarp
        );
        for nvrtc_version in [(12, 8), (13, 0), (13, 1), (13, 3)] {
            assert_eq!(
                fixed_pick_tf32(4621, 768, 2304, 170, (12, 0), nvrtc_version, true,),
                FixedTile::Tf32Sm120M64S2,
                "unqualified compiler {nvrtc_version:?} must use the portable SM120 schedule"
            );
            assert_eq!(
                fixed_pick_tf32(2048, 768, 2304, 170, (12, 0), nvrtc_version, true,),
                FixedTile::Tf32Sm120M64S2,
                "unqualified compiler {nvrtc_version:?} must not use the producer-warp schedule"
            );
        }
        assert_eq!(
            fixed_pick_tf32(4621, 768, 2304, 169, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4621, 768, 2304, 170, (12, 1), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4622, 768, 2304, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4621, 769, 2304, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4621, 768, 2305, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
    }

    #[test]
    fn tf32_selector_keeps_portable_and_thin_fallbacks() {
        assert_eq!(
            fixed_pick_tf32(4621, 768, 1928, 170, (12, 0), (13, 2), false),
            FixedTile::Tf32M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4621, 768, 1928, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(1, 768, 1928, 170, (12, 0), (13, 2), true),
            FixedTile::Tf32M16S4
        );
        assert_eq!(
            fixed_pick_tf32(4621, 768, 2304, 142, (8, 9), (13, 2), false),
            FixedTile::Tf32M64S2
        );
        assert_eq!(
            fixed_pick_tf32(4621, 768, 384, 142, (8, 9), (13, 2), false),
            FixedTile::Tf32M128S2
        );
    }

    #[test]
    fn selector_uses_measured_sm89_thin_wave_boundary() {
        let ada = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        for (m, k, n) in [
            (512, 768, 512),
            (128, 2560, 1024),
            (256, 768, 1024),
            (176, 768, 1536),
            (96, 768, 1928),
            (112, 768, 2304),
        ] {
            assert_eq!(fixed_pick_tile(m, n, k, ada), Some(FixedTile::Tc16));
        }
        for (m, k, n) in [
            (576, 768, 512),
            (288, 768, 1024),
            (192, 768, 1536),
            (112, 768, 1928),
            (128, 768, 2304),
        ] {
            assert_eq!(fixed_pick_tile(m, n, k, ada), Some(FixedTile::Tc64));
        }
    }

    #[test]
    fn mixed_f32_output_selector_follows_sm120_wave_balance() {
        let blackwell = FixedTileDevice {
            multiprocessors: 170,
            compute_capability: (12, 0),
        };
        for (m, n) in [(1, 17), (128, 2304), (512, 512)] {
            assert_eq!(
                fixed_pick_f32out_tile(m, 768, n, blackwell),
                FixedTile::Tc16
            );
        }
        for (m, n) in [(512, 768), (1536, 2304), (4621, 768)] {
            assert_eq!(
                fixed_pick_f32out_tile(m, 768, n, blackwell),
                FixedTile::Tc64
            );
        }
        for (m, n) in [(768, 2304), (2048, 2304), (4621, 2304)] {
            assert_eq!(
                fixed_pick_f32out_tile(m, 768, n, blackwell),
                FixedTile::Tc128
            );
        }
        assert_eq!(
            fixed_pick_f32out_tile(4621, 1928, 384, blackwell),
            FixedTile::Tc64
        );
        assert_eq!(
            fixed_pick_f32out_tile(4621, 2304, 384, blackwell),
            FixedTile::Tc128
        );
    }

    #[test]
    fn sm120_mixed_f32_output_selector_retains_measured_winners() {
        use FixedSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A, M128N128Bk32S3 as E};

        let device = FixedSm120HalfExactDevice {
            compute_capability: (12, 0),
            multiprocessors: 170,
            nvrtc_version: (13, 2),
        };
        for (dims, expected) in [
            ((4621, 384, 1928), A),
            ((4621, 768, 2304), E),
            ((4621, 1928, 384), C),
            ((2048, 768, 2304), C),
            ((2048, 2304, 768), C),
            ((4096, 3072, 1536), FixedSm120HalfTile::M64N128Bk64S2),
        ] {
            assert_eq!(
                fixed_pick_sm120_f32out(dims.0, dims.2, dims.1, device),
                Some(expected),
                "mixed-output selector changed at {dims:?}",
            );
        }
        assert_eq!(
            fixed_pick_sm120_half(4096, 1536, 3072, 170),
            Some(FixedSm120HalfTile::M128N128Bk32S2),
            "the generic comparator still extrapolates on the deep wide projection"
        );
        assert_eq!(
            fixed_pick_sm120_f32out(
                4621,
                2304,
                768,
                FixedSm120HalfExactDevice {
                    nvrtc_version: (12, 8),
                    ..device
                },
            ),
            fixed_pick_sm120_half(4621, 2304, 768, 170),
        );
    }

    #[test]
    fn tc16_compact_xor_layout_is_bijective_and_bank_distinct() {
        for (rows, chunks) in [(16usize, 8usize), (64, 4)] {
            let mut slots = vec![false; rows * chunks];
            for row in 0..rows {
                for chunk in 0..chunks {
                    let slot = (row * chunks + chunk) ^ (row & 7);
                    assert!(slot < slots.len());
                    assert!(!slots[slot], "layout alias at row={row} chunk={chunk}");
                    slots[slot] = true;
                }
            }
            assert!(slots.into_iter().all(|seen| seen));
            for row_group in (0..rows).step_by(8) {
                for chunk in 0..chunks {
                    let mut bank_groups = [false; 8];
                    for row in row_group..row_group + 8 {
                        let slot = (row * chunks + chunk) ^ (row & 7);
                        let bank_group = slot & 7;
                        assert!(!bank_groups[bank_group]);
                        bank_groups[bank_group] = true;
                    }
                    assert!(bank_groups.into_iter().all(|seen| seen));
                }
            }
        }
    }

    #[test]
    fn selector_keeps_unmeasured_architectures_on_two_thin_waves() {
        let ampere = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 0),
        };
        assert_eq!(
            fixed_pick_tile(320, 512, 768, ampere),
            Some(FixedTile::Tc64)
        );
    }

    #[test]
    fn selector_uses_live_sm_count_for_wide_wave_efficiency() {
        let ada = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        let blackwell = FixedTileDevice {
            multiprocessors: 170,
            compute_capability: (12, 0),
        };
        assert_eq!(
            fixed_pick_tile(384, 6144, 768, ada),
            Some(FixedTile::TcWn64)
        );
        assert_eq!(
            fixed_pick_tile(384, 6144, 768, blackwell),
            Some(FixedTile::Tc128)
        );
    }

    #[test]
    fn sm120_uses_fragment_reuse_for_deep_square_tiles() {
        assert_eq!(
            fixed_adjust_arch_tile(FixedTile::Tc128, 1024, (12, 0)),
            FixedTile::TcW64
        );
        assert_eq!(
            fixed_adjust_arch_tile(FixedTile::Tc128, 768, (12, 0)),
            FixedTile::Tc128
        );
        assert_eq!(
            fixed_adjust_arch_tile(FixedTile::Tc128, 2304, (8, 9)),
            FixedTile::Tc128
        );
        assert_eq!(
            fixed_adjust_arch_tile(FixedTile::TcWn64, 2304, (12, 0)),
            FixedTile::TcWn64
        );
    }
}
