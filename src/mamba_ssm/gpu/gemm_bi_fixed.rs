//! The FIXED family - the standalone deterministic INFERENCE kernel.
//!
//! Owns the fast NN-forward tile ladder in `kernels/gemm_bi_fixed.cu`
//! (the GBF tiles, byte-identical to each other per output element),
//! plus the legacy 64x64 WMMA tile as the narrow-N fallback and the f32
//! FFMA tile. One dispatcher, shape-keyed:
//!
//!   exact f32            -> CC12.0 exact-TMA at qualified A0/B0, Ada/CC12.0
//!                           N64 copy-plans at qualified rows (Ada A/B/D/E);
//!                           FFMA 64x128 at other measured CC12.0/
//!                           170-SM points; portable FFMA 64x64 elsewhere
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
    /// Fixed-owned 128x128/S3 twin with explicit RNA conversion; narrow Ada AUTO.
    Tf32RnaM128N128S3,
    /// Ada explicit-RNA 128x96/S3, AUTO at measured E0 rows in revision 45.
    Tf32RnaM128N96S3,
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
    /// Force-only SM120 TMA TF32, 64x64 CTA, two-stage pair-store epilogue.
    Tf32Sm120M64S2PairStore,
    /// SM120 TMA BF16/F16 route selected from the qualified tile matrix.
    Sm120Half(FixedSm120HalfTile),
    /// 128x128 CTA, 256 threads, 2-stage cp.async, dynamic smem 71 680 B.
    Tc128,
    /// Ada-only pipelined/vector-store Tc128, AUTO in qualified hot cells.
    Tc128Sm89Pipeline,
    /// Ada-only packed/XOR Tc128 twin, AUTO in qualified hot cells.
    Tc128Sm89Swizzle,
    /// Ada-only three-stage packed/XOR Tc128, independently qualified force route.
    Tc128Sm89S3,
    /// Ada F16 M64xN64/BK64/S3, AUTO at measured D0 rows in revision 45.
    TcM64N64Sm89S3,
    /// Ada F16 M128xN64/BK64/S2, AUTO at measured E0 rows in revision 45.
    TcM128N64Sm89S2,
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
    /// Ada-only exact-F32 N64 copy-plan, AUTO at qualified A/B/D/E bias rows.
    F32Sm89N64CopyPlan,
    /// CC12.0 exact-F32 N64 copy-plan, AUTO at qualified E0/E1/B1 rows.
    F32Sm120N64CopyPlan,
    /// Force-only exact-F32 CopyPlan twin with 256 threads and 4x4 microtiles.
    F32Sm120N64CopyPlanT256,
    /// Force-only M128N64/BK32 CopyPlan with 256 threads and 8x4 microtiles.
    F32Sm120M128N64CopyPlanT256,
    /// Force-only CC12.0 sliced-copy N64; same ascending scalar FMA.
    F32Sm120N64Sliced,
    /// Fixed bridge to exact-TMA M128xN64, AUTO at qualified SM120 B0.
    F32Sm120TmaFmaM128N64,
    /// Fixed bridge to exact-TMA M64xN128, AUTO at qualified SM120 A0.
    F32Sm120TmaFmaM64N128,
    /// Fixed-local exact-TMA M128xN64 post-dot-bias fallback at qualified SM120 A1.
    F32Sm120TmaFmaFixedPostBiasM128N64,
    /// Force-only exact-TMA M64xN128 with Fixed-compatible post-dot bias.
    F32Sm120TmaFmaFixedPostBiasM64N128,
    /// Force-only exact-TMA M128xN96 screening tile with 256 threads.
    F32Sm120TmaFmaFixedPostBiasM128N96,
    /// Force-only exact-TMA M128xN64 post-dot bias with four-value K chunks.
    F32Sm120TmaFmaFixedPostBiasM128N64K4,
    /// Exact-TMA M128xN64 post-dot bias with eight compute warps, AUTO at qualified SM120 A1.
    F32Sm120TmaFmaFixedPostBiasM128N64T256,
    /// Force-only exact-TMA M128xN64 with eight compute warps and no bias epilogue.
    F32Sm120TmaFmaFixedNoBiasM128N64T256,
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
    if tile == FixedTile::F32Sm89N64CopyPlan {
        return launch_sm89_exact_n64(ctx, operands, shape);
    }
    if tile == FixedTile::F32Sm120N64CopyPlan {
        return launch_sm120_exact_n64(ctx, operands, shape);
    }
    if matches!(
        tile,
        FixedTile::F32Sm120N64CopyPlanT256 | FixedTile::F32Sm120M128N64CopyPlanT256
    ) {
        return launch_sm120_copyplan_t256(ctx, operands, shape, tile);
    }
    if tile == FixedTile::F32Sm120N64Sliced {
        return launch_sm120_sliced(ctx, operands, shape);
    }
    if matches!(
        tile,
        FixedTile::F32Sm120TmaFmaM128N64 | FixedTile::F32Sm120TmaFmaM64N128
    ) {
        return launch_sm120_tma_fma(ctx, operands, shape, tile);
    }
    if matches!(
        tile,
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64
            | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
            | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
            | FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
            | FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
            | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96
    ) {
        return launch_sm120_tma_postbias(ctx, operands, shape, tile);
    }
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
    if tile == FixedTile::Tc128Sm89Pipeline {
        if operands.x.dtype == WeightDtype::F32
            || operands.x.dtype != operands.w.dtype
            || operands.x.dtype != operands.c.dtype
        {
            return Err("Fixed Ada half pipeline requires matching bf16/f16 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_sm89_half_pipeline(ctx, operands.x.dtype, &args);
    }
    if tile == FixedTile::Tc128Sm89Swizzle {
        if operands.x.dtype == WeightDtype::F32
            || operands.x.dtype != operands.w.dtype
            || operands.x.dtype != operands.c.dtype
        {
            return Err("Fixed Ada half swizzle requires matching bf16/f16 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_sm89_half_swizzle(ctx, operands.x.dtype, &args);
    }
    if tile == FixedTile::Tc128Sm89S3 {
        if operands.x.dtype == WeightDtype::F32
            || operands.x.dtype != operands.w.dtype
            || operands.x.dtype != operands.c.dtype
        {
            return Err("Fixed Ada half s3 requires matching bf16/f16 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_sm89_half_s3(ctx, operands.x.dtype, &args);
    }
    if matches!(tile, FixedTile::TcM64N64Sm89S3 | FixedTile::TcM128N64Sm89S2) {
        if operands.x.dtype != WeightDtype::F16
            || operands.w.dtype != WeightDtype::F16
            || operands.c.dtype != WeightDtype::F16
        {
            return Err("Fixed Ada N64 finalists require matching f16 operands".into());
        }
        let args = FixedArgs::try_new(operands, shape)?;
        return launch_sm89_half_n64(ctx, tile, &args);
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
            | FixedTile::Tf32RnaM128N128S3
            | FixedTile::Tf32RnaM128N96S3
            | FixedTile::Tf32M64S2
            | FixedTile::Tf32M64S3
            | FixedTile::Tf32M16S4
            | FixedTile::Tf32Sm120M128S2
            | FixedTile::Tf32Sm120M128S3
            | FixedTile::Tf32Sm120M64N128S2
            | FixedTile::Tf32Sm120M64N128S3
            | FixedTile::Tf32Sm120M64S2ProducerWarp
            | FixedTile::Tf32Sm120M64S2
            | FixedTile::Tf32Sm120M64S2PairStore
    ) {
        if operands.c.dtype != WeightDtype::F32
            || operands.x.dtype != WeightDtype::F32
            || operands.w.dtype != WeightDtype::F32
        {
            return Err("forced Fixed TF32 launch requires f32 operands".into());
        }
        if matches!(
            tile,
            FixedTile::Tf32RnaM128N128S3 | FixedTile::Tf32RnaM128N96S3
        ) && (shape.m == 0 || shape.n == 0)
        {
            return Ok(());
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

// Actual NVRTC CUDA12.8/13.0/13.2 qualification on the 142-SM Ada target.
// The preferred route is the reviewed direct-pair winner; compatibility
// fallback remains limited to independently qualified holders. Returning None
// preserves the general architecture/portable ladder below this overlay.
fn fixed_select_sm89_half_auto_tile(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    pipeline_available: bool,
    swizzle_available: bool,
    s3_available: bool,
) -> Option<FixedTile> {
    use FixedTile::{
        Tc128Sm89Pipeline as Pipeline, Tc128Sm89S3 as S3, Tc128Sm89Swizzle as Swizzle,
    };

    if !nvrtc_library_known
        || device.compute_capability != (8, 9)
        || device.multiprocessors != 142
        || !matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
        || operands.c.dtype == WeightDtype::F32
        || operands.c.dtype != operands.x.dtype
        || operands.x.dtype != operands.w.dtype
        || [operands.c.ptr, operands.x.ptr, operands.w.ptr]
            .into_iter()
            .any(|ptr| ptr == 0 || !ptr.is_multiple_of(16))
        || operands.bias_ptr.is_some_and(|ptr| !ptr.is_multiple_of(4))
        || !matches!(
            (shape.m, shape.k, shape.n),
            (4621, 384, 1928)
                | (4621, 768, 2304)
                | (4621, 1928, 384)
                | (2048, 768, 2304)
                | (2048, 2304, 768)
        )
    {
        return None;
    }

    if s3_available
        && nvrtc == (13, 2)
        && (shape.m, shape.k, shape.n) == (4621, 768, 2304)
        && operands.bias_ptr.is_none()
    {
        return Some(S3);
    }

    let dims = (shape.m, shape.k, shape.n);
    let preferred = match (nvrtc, operands.c.dtype, dims, operands.bias_ptr.is_some()) {
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), false) => Pipeline,
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 384, 1928), _) => Pipeline,
        ((12, 8) | (13, 0), _, _, _) => Swizzle,
        ((13, 2), WeightDtype::Bf16, (4621, 384, 1928) | (4621, 1928, 384), _) => Pipeline,
        (
            (13, 2),
            WeightDtype::F16,
            (4621, 384, 1928) | (4621, 1928, 384) | (2048, 2304, 768),
            _,
        ) => Pipeline,
        ((13, 2), _, _, _) => Swizzle,
        _ => return None,
    };

    match (nvrtc, preferred) {
        ((12, 8) | (13, 0), Pipeline) if pipeline_available => Some(Pipeline),
        ((12, 8) | (13, 0), Swizzle) if swizzle_available => Some(Swizzle),
        ((12, 8) | (13, 0), Pipeline) if swizzle_available => Some(Swizzle),
        ((12, 8) | (13, 0), Swizzle) if pipeline_available => Some(Pipeline),
        ((13, 2), Pipeline) if pipeline_available => Some(Pipeline),
        ((13, 2), Swizzle) if swizzle_available => Some(Swizzle),
        ((13, 2), Swizzle) if pipeline_available => Some(Pipeline),
        _ => None,
    }
}

// Literal phase-2 rows from the three-toolkit forced qualification. A missing
// optional holder declines only its own row; the revision-43 selector remains
// the independent compatibility fallback.
fn fixed_select_sm89_half_finalist_auto_tile(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    d_available: bool,
    e_available: bool,
) -> Option<FixedTile> {
    if !nvrtc_library_known
        || device.compute_capability != (8, 9)
        || device.multiprocessors != 142
        || nvrtc != (13, 2)
        || operands.c.dtype != WeightDtype::F16
        || operands.x.dtype != WeightDtype::F16
        || operands.w.dtype != WeightDtype::F16
        || operands.bias_ptr.is_some()
        || [operands.c.ptr, operands.x.ptr, operands.w.ptr]
            .into_iter()
            .any(|ptr| ptr == 0 || !ptr.is_multiple_of(16))
    {
        return None;
    }
    match (shape.m, shape.k, shape.n) {
        (2048, 768, 2304) if d_available => Some(FixedTile::TcM64N64Sm89S3),
        (2048, 2304, 768) if e_available => Some(FixedTile::TcM128N64Sm89S2),
        _ => None,
    }
}

#[cfg(test)]
mod sm89_pipeline_auto_tests {
    use super::*;

    const P: FixedTile = FixedTile::Tc128Sm89Pipeline;
    const S: FixedTile = FixedTile::Tc128Sm89Swizzle;
    const S3: FixedTile = FixedTile::Tc128Sm89S3;

    fn operands(dtype: WeightDtype, has_bias: bool) -> FixedFwdOperands {
        FixedFwdOperands {
            c: TypedPtr { ptr: 0x1000, dtype },
            x: TypedPtr { ptr: 0x2000, dtype },
            w: TypedPtr { ptr: 0x3000, dtype },
            bias_ptr: has_bias.then_some(0x4004),
        }
    }

    fn select(
        nvrtc: (i32, i32),
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        has_bias: bool,
        known_library: bool,
        pipeline_available: bool,
        swizzle_available: bool,
        s3_available: bool,
    ) -> Option<FixedTile> {
        fixed_select_sm89_half_auto_tile(
            operands(dtype, has_bias),
            FixedShape {
                m: dims.0,
                k: dims.1,
                n: dims.2,
            },
            FixedTileDevice {
                multiprocessors: 142,
                compute_capability: (8, 9),
            },
            nvrtc,
            known_library,
            pipeline_available,
            swizzle_available,
            s3_available,
        )
    }

    #[test]
    fn sm89_half_auto_v43_promotes_only_measured_cuda132_b0_no_bias_cells() {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            assert_eq!(
                select(
                    (13, 2),
                    dtype,
                    (4621, 768, 2304),
                    false,
                    true,
                    true,
                    true,
                    true,
                ),
                Some(S3),
                "CUDA13.2 {dtype:?} B0/no-bias must select the measured S3 winner"
            );
        }
    }

    #[test]
    fn sm89_half_auto_v45_promotes_only_measured_cuda132_f16_n64_cells() {
        let device = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        for (dims, expected) in [
            ((2048, 768, 2304), FixedTile::TcM64N64Sm89S3),
            ((2048, 2304, 768), FixedTile::TcM128N64Sm89S2),
        ] {
            let shape = FixedShape {
                m: dims.0,
                k: dims.1,
                n: dims.2,
            };
            assert_eq!(
                fixed_select_sm89_half_finalist_auto_tile(
                    operands(WeightDtype::F16, false),
                    shape,
                    device,
                    (13, 2),
                    true,
                    true,
                    true,
                ),
                Some(expected),
                "CUDA13.2 F16 no-bias {dims:?} must select its measured N64 finalist"
            );
        }
    }

    #[test]
    fn sm89_half_auto_v45_declines_unmeasured_or_unavailable_finalist_rows() {
        let device = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        let d = FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        };
        let e = FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        };
        let select_finalist = |operands, shape, device, nvrtc, known, d_available, e_available| {
            fixed_select_sm89_half_finalist_auto_tile(
                operands,
                shape,
                device,
                nvrtc,
                known,
                d_available,
                e_available,
            )
        };
        let f16 = operands(WeightDtype::F16, false);
        for (shape, own_d, own_e, expected) in [
            (d, true, false, FixedTile::TcM64N64Sm89S3),
            (e, false, true, FixedTile::TcM128N64Sm89S2),
        ] {
            assert_eq!(
                select_finalist(f16, shape, device, (13, 2), true, own_d, own_e),
                Some(expected),
                "the unrelated finalist holder may be absent"
            );
            assert_eq!(
                select_finalist(f16, shape, device, (13, 2), true, !own_d, !own_e),
                None,
                "the selected finalist's own holder is required"
            );
            assert_eq!(
                select_finalist(f16, shape, device, (13, 2), false, true, true),
                None,
                "unknown NVRTC library must decline"
            );
            for nvrtc in [(12, 8), (13, 0), (13, 1), (13, 3)] {
                assert_eq!(
                    select_finalist(f16, shape, device, nvrtc, true, true, true),
                    None,
                    "unmeasured toolkit {nvrtc:?} must retain the old selector"
                );
            }
            for bad_device in [
                FixedTileDevice {
                    compute_capability: (8, 6),
                    ..device
                },
                FixedTileDevice {
                    multiprocessors: 141,
                    ..device
                },
            ] {
                assert_eq!(
                    select_finalist(f16, shape, bad_device, (13, 2), true, true, true),
                    None
                );
            }
            for bad in [
                operands(WeightDtype::F16, true),
                operands(WeightDtype::Bf16, false),
                FixedFwdOperands {
                    c: TypedPtr {
                        dtype: WeightDtype::F32,
                        ..f16.c
                    },
                    ..f16
                },
                FixedFwdOperands {
                    x: TypedPtr {
                        dtype: WeightDtype::F32,
                        ..f16.x
                    },
                    ..f16
                },
                FixedFwdOperands {
                    w: TypedPtr {
                        dtype: WeightDtype::F32,
                        ..f16.w
                    },
                    ..f16
                },
            ] {
                assert_eq!(
                    select_finalist(bad, shape, device, (13, 2), true, true, true),
                    None
                );
            }
            for field in 0..3 {
                for ptr in [0, 0x1008] {
                    let mut bad = f16;
                    match field {
                        0 => bad.c.ptr = ptr,
                        1 => bad.x.ptr = ptr,
                        _ => bad.w.ptr = ptr,
                    }
                    assert_eq!(
                        select_finalist(bad, shape, device, (13, 2), true, true, true),
                        None,
                        "field {field} pointer {ptr:#x} must decline"
                    );
                }
            }
            assert_eq!(
                select_finalist(
                    f16,
                    FixedShape { m: 2047, ..shape },
                    device,
                    (13, 2),
                    true,
                    true,
                    true,
                ),
                None
            );
        }
    }

    #[test]
    fn sm89_half_auto_v43_matches_all_literal_cells_and_availability_states() {
        let cases = [
            ((12, 8), WeightDtype::Bf16, (4621, 384, 1928), false, P),
            ((12, 8), WeightDtype::Bf16, (4621, 384, 1928), true, S),
            ((12, 8), WeightDtype::Bf16, (4621, 768, 2304), false, S),
            ((12, 8), WeightDtype::Bf16, (4621, 768, 2304), true, S),
            ((12, 8), WeightDtype::Bf16, (4621, 1928, 384), false, S),
            ((12, 8), WeightDtype::Bf16, (4621, 1928, 384), true, S),
            ((12, 8), WeightDtype::Bf16, (2048, 768, 2304), false, S),
            ((12, 8), WeightDtype::Bf16, (2048, 768, 2304), true, S),
            ((12, 8), WeightDtype::Bf16, (2048, 2304, 768), false, S),
            ((12, 8), WeightDtype::Bf16, (2048, 2304, 768), true, S),
            ((12, 8), WeightDtype::F16, (4621, 384, 1928), false, P),
            ((12, 8), WeightDtype::F16, (4621, 384, 1928), true, P),
            ((12, 8), WeightDtype::F16, (4621, 768, 2304), false, S),
            ((12, 8), WeightDtype::F16, (4621, 768, 2304), true, S),
            ((12, 8), WeightDtype::F16, (4621, 1928, 384), false, S),
            ((12, 8), WeightDtype::F16, (4621, 1928, 384), true, S),
            ((12, 8), WeightDtype::F16, (2048, 768, 2304), false, S),
            ((12, 8), WeightDtype::F16, (2048, 768, 2304), true, S),
            ((12, 8), WeightDtype::F16, (2048, 2304, 768), false, S),
            ((12, 8), WeightDtype::F16, (2048, 2304, 768), true, S),
            ((13, 0), WeightDtype::Bf16, (4621, 384, 1928), false, P),
            ((13, 0), WeightDtype::Bf16, (4621, 384, 1928), true, S),
            ((13, 0), WeightDtype::Bf16, (4621, 768, 2304), false, S),
            ((13, 0), WeightDtype::Bf16, (4621, 768, 2304), true, S),
            ((13, 0), WeightDtype::Bf16, (4621, 1928, 384), false, S),
            ((13, 0), WeightDtype::Bf16, (4621, 1928, 384), true, S),
            ((13, 0), WeightDtype::Bf16, (2048, 768, 2304), false, S),
            ((13, 0), WeightDtype::Bf16, (2048, 768, 2304), true, S),
            ((13, 0), WeightDtype::Bf16, (2048, 2304, 768), false, S),
            ((13, 0), WeightDtype::Bf16, (2048, 2304, 768), true, S),
            ((13, 0), WeightDtype::F16, (4621, 384, 1928), false, P),
            ((13, 0), WeightDtype::F16, (4621, 384, 1928), true, P),
            ((13, 0), WeightDtype::F16, (4621, 768, 2304), false, S),
            ((13, 0), WeightDtype::F16, (4621, 768, 2304), true, S),
            ((13, 0), WeightDtype::F16, (4621, 1928, 384), false, S),
            ((13, 0), WeightDtype::F16, (4621, 1928, 384), true, S),
            ((13, 0), WeightDtype::F16, (2048, 768, 2304), false, S),
            ((13, 0), WeightDtype::F16, (2048, 768, 2304), true, S),
            ((13, 0), WeightDtype::F16, (2048, 2304, 768), false, S),
            ((13, 0), WeightDtype::F16, (2048, 2304, 768), true, S),
            ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), false, P),
            ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), true, P),
            ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false, S3),
            ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), true, S),
            ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), false, P),
            ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), true, P),
            ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), false, S),
            ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), true, S),
            ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), false, S),
            ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), true, S),
            ((13, 2), WeightDtype::F16, (4621, 384, 1928), false, P),
            ((13, 2), WeightDtype::F16, (4621, 384, 1928), true, P),
            ((13, 2), WeightDtype::F16, (4621, 768, 2304), false, S3),
            ((13, 2), WeightDtype::F16, (4621, 768, 2304), true, S),
            ((13, 2), WeightDtype::F16, (4621, 1928, 384), false, P),
            ((13, 2), WeightDtype::F16, (4621, 1928, 384), true, P),
            ((13, 2), WeightDtype::F16, (2048, 768, 2304), false, S),
            ((13, 2), WeightDtype::F16, (2048, 768, 2304), true, S),
            ((13, 2), WeightDtype::F16, (2048, 2304, 768), false, P),
            ((13, 2), WeightDtype::F16, (2048, 2304, 768), true, P),
        ];
        assert_eq!(cases.len(), 60);
        for (nvrtc, dtype, dims, has_bias, preferred) in cases {
            for pipeline_available in [false, true] {
                for swizzle_available in [false, true] {
                    for s3_available in [false, true] {
                        let old_preferred = if preferred == S3 { S } else { preferred };
                        let expected = if preferred == S3 && s3_available {
                            Some(S3)
                        } else {
                            match (nvrtc, old_preferred, pipeline_available, swizzle_available) {
                                ((12, 8) | (13, 0), P, true, _) => Some(P),
                                ((12, 8) | (13, 0), P, false, true) => Some(S),
                                ((12, 8) | (13, 0), S, _, true) => Some(S),
                                ((12, 8) | (13, 0), S, true, false) => Some(P),
                                ((13, 2), P, true, _) => Some(P),
                                ((13, 2), S, _, true) => Some(S),
                                ((13, 2), S, true, false) => Some(P),
                                _ => None,
                            }
                        };
                        assert_eq!(
                            select(
                                nvrtc,
                                dtype,
                                dims,
                                has_bias,
                                true,
                                pipeline_available,
                                swizzle_available,
                                s3_available,
                            ),
                            expected,
                            "{nvrtc:?} {dtype:?} {dims:?} bias={has_bias} pipeline={pipeline_available} swizzle={swizzle_available} s3={s3_available}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sm89_half_auto_v43_declines_every_common_gate_failure() {
        let device = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        let shape = FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        };
        let ops = operands(WeightDtype::Bf16, false);
        let choose = |o, s, d, v, known, pipeline, swizzle| {
            fixed_select_sm89_half_auto_tile(o, s, d, v, known, pipeline, swizzle, true)
        };

        assert_eq!(choose(ops, shape, device, (13, 2), false, true, true), None);
        assert_eq!(
            fixed_select_sm89_half_auto_tile(
                ops,
                shape,
                device,
                (13, 2),
                true,
                false,
                false,
                false,
            ),
            None
        );
        for version in [(12, 7), (12, 9), (13, 1), (13, 3), (14, 0)] {
            assert_eq!(choose(ops, shape, device, version, true, true, true), None);
        }
        for cc in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (12, 0), (12, 1)] {
            assert_eq!(
                choose(
                    ops,
                    shape,
                    FixedTileDevice {
                        compute_capability: cc,
                        ..device
                    },
                    (13, 2),
                    true,
                    true,
                    true
                ),
                None
            );
        }
        for multiprocessors in [0, 141, 143, 170] {
            assert_eq!(
                choose(
                    ops,
                    shape,
                    FixedTileDevice {
                        multiprocessors,
                        ..device
                    },
                    (13, 2),
                    true,
                    true,
                    true
                ),
                None
            );
        }
        for adjacent in [
            FixedShape {
                m: shape.m - 1,
                ..shape
            },
            FixedShape {
                m: shape.m + 1,
                ..shape
            },
            FixedShape {
                k: shape.k - 1,
                ..shape
            },
            FixedShape {
                k: shape.k + 1,
                ..shape
            },
            FixedShape {
                n: shape.n - 1,
                ..shape
            },
            FixedShape {
                n: shape.n + 1,
                ..shape
            },
        ] {
            assert_eq!(
                choose(ops, adjacent, device, (13, 2), true, true, true),
                None
            );
        }
        for ptr in [0, 0x1001, 0x1002, 0x1004, 0x1008] {
            for wrong in [
                FixedFwdOperands {
                    c: TypedPtr { ptr, ..ops.c },
                    ..ops
                },
                FixedFwdOperands {
                    x: TypedPtr { ptr, ..ops.x },
                    ..ops
                },
                FixedFwdOperands {
                    w: TypedPtr { ptr, ..ops.w },
                    ..ops
                },
            ] {
                assert_eq!(
                    choose(wrong, shape, device, (13, 2), true, true, true),
                    None
                );
            }
        }
        for wrong in [
            operands(WeightDtype::F32, false),
            FixedFwdOperands {
                c: TypedPtr {
                    dtype: WeightDtype::F32,
                    ..ops.c
                },
                ..ops
            },
            FixedFwdOperands {
                x: TypedPtr {
                    dtype: WeightDtype::F16,
                    ..ops.x
                },
                ..ops
            },
            FixedFwdOperands {
                w: TypedPtr {
                    dtype: WeightDtype::F16,
                    ..ops.w
                },
                ..ops
            },
            FixedFwdOperands {
                bias_ptr: Some(0x4001),
                ..ops
            },
            FixedFwdOperands {
                bias_ptr: Some(0x4002),
                ..ops
            },
        ] {
            assert_eq!(
                choose(wrong, shape, device, (13, 2), true, true, true),
                None
            );
        }
    }
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
        | FixedTile::F32Sm89N64CopyPlan
        | FixedTile::F32Sm120N64CopyPlan
        | FixedTile::F32Sm120N64CopyPlanT256
        | FixedTile::F32Sm120M128N64CopyPlanT256
        | FixedTile::F32Sm120N64Sliced
        | FixedTile::F32Sm120TmaFmaM128N64
        | FixedTile::F32Sm120TmaFmaM64N128
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        | FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96
        | FixedTile::Tf32M128S2
        | FixedTile::Tf32M128S3
        | FixedTile::Tf32M128N128S3
        | FixedTile::Tf32RnaM128N128S3
        | FixedTile::Tf32RnaM128N96S3
        | FixedTile::Tf32M64S2
        | FixedTile::Tf32M64S3
        | FixedTile::Tf32M16S4
        | FixedTile::Tf32Sm120M128S2
        | FixedTile::Tf32Sm120M128S3
        | FixedTile::Tf32Sm120M64N128S2
        | FixedTile::Tf32Sm120M64N128S3
        | FixedTile::Tf32Sm120M64S2ProducerWarp
        | FixedTile::Tf32Sm120M64S2
        | FixedTile::Tf32Sm120M64S2PairStore
        | FixedTile::Sm120Half(_)
        | FixedTile::Tc128Sm89Pipeline
        | FixedTile::Tc128Sm89Swizzle
        | FixedTile::Tc128Sm89S3
        | FixedTile::TcM64N64Sm89S3
        | FixedTile::TcM128N64Sm89S2 => unreachable!("tile has its own launcher"),
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

fn prepare_sm89_exact_n64_launch(
    operands: FixedFwdOperands,
    shape: FixedShape,
    compute_capability: (u32, u32),
) -> Result<Option<(FixedArgs, u32)>, String> {
    if compute_capability != (8, 9) {
        return Err("Fixed Ada exact N64 requires CC8.9".into());
    }
    if [operands.c.dtype, operands.x.dtype, operands.w.dtype]
        .into_iter()
        .any(|dtype| dtype != WeightDtype::F32)
    {
        return Err("Fixed Ada exact N64 requires homogeneous f32 operands".into());
    }
    // No output means no pointer arithmetic, shape conversion or work. This
    // deliberately also admits null pointers and an unused unrepresentable K.
    if shape.m == 0 || shape.n == 0 {
        return Ok(None);
    }
    let args = FixedArgs::try_new(operands, shape)?;
    if args.c == 0 || !args.c.is_multiple_of(4) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed Ada exact N64 requires non-null f32-aligned C and null-or-f32-aligned bias"
                .into(),
        );
    }
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
    {
        return Err(
            "Fixed Ada exact N64 requires non-null f32-aligned A and B for nonzero K".into(),
        );
    }
    args.m
        .checked_add(63)
        .ok_or("Fixed Ada exact N64 padded M exceeds i32")?;
    args.n
        .checked_add(63)
        .ok_or("Fixed Ada exact N64 padded N exceeds i32")?;
    args.k
        .checked_add(31)
        .ok_or("Fixed Ada exact N64 padded K exceeds i32")?;
    let grid = (args.m as u32)
        .div_ceil(64)
        .checked_mul((args.n as u32).div_ceil(64))
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada exact N64 launch grid exceeds i32")?;
    fixed_sm89_exact_n64_byte_end(args.c, args.m, args.n, "C")?;
    if args.k != 0 {
        fixed_sm89_exact_n64_byte_end(args.a, args.m, args.k, "A")?;
        fixed_sm89_exact_n64_byte_end(args.b, args.k, args.n, "B")?;
    }
    if args.bias != 0 {
        fixed_sm89_exact_n64_byte_end(args.bias, 1, args.n, "bias")?;
    }
    // Allocation length, residency and nonaliasing remain the caller's raw
    // pointer contract; this gate rules out numeric/address wrap before launch.
    Ok(Some((args, grid)))
}

fn fixed_sm89_exact_n64_byte_end(
    pointer: u64,
    rows: i32,
    cols: i32,
    label: &str,
) -> Result<(), String> {
    let bytes = u64::try_from(rows)
        .ok()
        .and_then(|rows| {
            u64::try_from(cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| format!("Fixed Ada exact N64 {label} byte extent exceeds u64"))?;
    pointer
        .checked_add(bytes)
        .ok_or_else(|| format!("Fixed Ada exact N64 {label} byte endpoint exceeds u64"))?;
    Ok(())
}

fn launch_sm89_exact_n64(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
) -> Result<(), String> {
    let Some((args, grid)) =
        prepare_sm89_exact_n64_launch(operands, shape, ctx.compute_capability())?
    else {
        return Ok(());
    };
    // Module loading already performed the exact Driver ABI/resource census
    // and candidate-only carveout preference. Capture configures nothing.
    let function = ctx
        .kernels
        .fixed_sm89_f32_n64_copyplan
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm89_f32_n64_copyplan_rejection
                .clone()
                .unwrap_or_else(|| "Fixed Ada exact N64 copy-plan is not admitted".into())
        })?;
    let params = FixedSm89ExactF32Params::forward(&args);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&args.a)
        .arg(&args.b)
        .arg(&args.bias)
        .arg(&params);
    // A failed explicit force is not retried through a different arithmetic
    // route. The caller receives the real launch error.
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed Ada exact N64 copy-plan: {error:?}"))
}

fn prepare_sm120_exact_n64_launch(
    operands: FixedFwdOperands,
    shape: FixedShape,
    compute_capability: (u32, u32),
) -> Result<Option<(FixedArgs, u32)>, String> {
    if compute_capability != (12, 0) {
        return Err("Fixed SM120 exact N64 requires CC12.0".into());
    }
    if [operands.c.dtype, operands.x.dtype, operands.w.dtype]
        .into_iter()
        .any(|dtype| dtype != WeightDtype::F32)
    {
        return Err("Fixed SM120 exact N64 requires homogeneous f32 operands".into());
    }
    if shape.m == 0 || shape.n == 0 {
        return Ok(None);
    }
    let args = FixedArgs::try_new(operands, shape)?;
    if args.c == 0 || !args.c.is_multiple_of(4) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed SM120 exact N64 requires non-null f32-aligned C and null-or-f32-aligned bias"
                .into(),
        );
    }
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
    {
        return Err(
            "Fixed SM120 exact N64 requires non-null f32-aligned A and B for nonzero K".into(),
        );
    }
    args.m
        .checked_add(63)
        .ok_or("Fixed SM120 exact N64 padded M exceeds i32")?;
    args.n
        .checked_add(63)
        .ok_or("Fixed SM120 exact N64 padded N exceeds i32")?;
    args.k
        .checked_add(31)
        .ok_or("Fixed SM120 exact N64 padded K exceeds i32")?;
    let grid = (args.m as u32)
        .div_ceil(64)
        .checked_mul((args.n as u32).div_ceil(64))
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed SM120 exact N64 launch grid exceeds i32")?;
    fixed_sm120_exact_n64_byte_end(args.c, args.m, args.n, "C")?;
    if args.k != 0 {
        fixed_sm120_exact_n64_byte_end(args.a, args.m, args.k, "A")?;
        fixed_sm120_exact_n64_byte_end(args.b, args.k, args.n, "B")?;
    }
    if args.bias != 0 {
        fixed_sm120_exact_n64_byte_end(args.bias, 1, args.n, "bias")?;
    }
    Ok(Some((args, grid)))
}

fn fixed_sm120_exact_n64_byte_end(
    pointer: u64,
    rows: i32,
    cols: i32,
    label: &str,
) -> Result<(), String> {
    let bytes = u64::try_from(rows)
        .ok()
        .and_then(|rows| {
            u64::try_from(cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| format!("Fixed SM120 exact N64 {label} byte extent exceeds u64"))?;
    pointer
        .checked_add(bytes)
        .ok_or_else(|| format!("Fixed SM120 exact N64 {label} byte endpoint exceeds u64"))?;
    Ok(())
}

fn launch_sm120_exact_n64(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
) -> Result<(), String> {
    let Some((args, grid)) =
        prepare_sm120_exact_n64_launch(operands, shape, ctx.compute_capability())?
    else {
        return Ok(());
    };
    let function = ctx
        .kernels
        .fixed_sm120_f32_n64_copyplan
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm120_f32_n64_copyplan_rejection
                .clone()
                .unwrap_or_else(|| "Fixed SM120 exact N64 copy-plan is not admitted".into())
        })?;
    let params = FixedSm89ExactF32Params::forward(&args);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&args.a)
        .arg(&args.b)
        .arg(&args.bias)
        .arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 exact N64 copy-plan: {error:?}"))
}

fn launch_sm120_copyplan_t256(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> Result<(), String> {
    if !fixed_sm120_tma_fma_force_physical_eligible(
        operands,
        shape,
        FixedTileDevice {
            multiprocessors: ctx.kernels.multiprocessor_count(),
            compute_capability: ctx.compute_capability(),
        },
        ctx.f32_triad_policy(),
        tile,
    ) {
        return Err("Fixed SM120 CopyPlan T256 is outside its force-only hot-row contract".into());
    }
    let Some((args, grid64)) =
        prepare_sm120_exact_n64_launch(operands, shape, ctx.compute_capability())?
    else {
        return Ok(());
    };
    let (function, rejection, grid) = if tile == FixedTile::F32Sm120M128N64CopyPlanT256 {
        (
            &ctx.kernels.fixed_sm120_f32_m128n64_copyplan_t256,
            &ctx.kernels.fixed_sm120_f32_m128n64_copyplan_t256_rejection,
            (args.m as u32).div_ceil(128) * (args.n as u32).div_ceil(64),
        )
    } else {
        (
            &ctx.kernels.fixed_sm120_f32_n64_copyplan_t256,
            &ctx.kernels.fixed_sm120_f32_n64_copyplan_t256_rejection,
            grid64,
        )
    };
    let function = function.as_ref().ok_or_else(|| {
        rejection
            .clone()
            .unwrap_or_else(|| format!("Fixed SM120 {tile:?} is not admitted"))
    })?;
    let params = FixedSm89ExactF32Params::forward(&args);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&args.a)
        .arg(&args.b)
        .arg(&args.bias)
        .arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("Fixed SM120 CopyPlan T256: {error:?}"))
}

fn launch_sm120_sliced(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
) -> Result<(), String> {
    let Some((args, grid)) =
        prepare_sm120_exact_n64_launch(operands, shape, ctx.compute_capability())?
    else {
        return Ok(());
    };
    let function = ctx
        .kernels
        .fixed_sm120_f32_n64_sliced
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm120_f32_n64_sliced_rejection
                .clone()
                .unwrap_or_else(|| "Fixed SM120 sliced N64 is not admitted".into())
        })?;
    let params = FixedSm89ExactF32Params::forward(&args);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&args.a)
        .arg(&args.b)
        .arg(&args.bias)
        .arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 sliced N64: {error:?}"))
}

fn launch_sm120_tma_fma(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> Result<(), String> {
    if !fixed_sm120_tma_fma_force_physical_eligible(
        operands,
        shape,
        FixedTileDevice {
            multiprocessors: ctx.kernels.multiprocessor_count(),
            compute_capability: ctx.compute_capability(),
        },
        ctx.f32_triad_policy(),
        tile,
    ) {
        return Err(
            "Fixed SM120 exact-TMA forced launch is outside its tile-specific hot-shape and bias contract"
                .into(),
        );
    }
    let physical_tile = match tile {
        FixedTile::F32Sm120TmaFmaM128N64 => super::gemm_bi_triad::FixedSm120ExactTmaTile::M128N64,
        FixedTile::F32Sm120TmaFmaM64N128 => super::gemm_bi_triad::FixedSm120ExactTmaTile::M64N128,
        _ => unreachable!("exact-TMA launcher received a non-TMA Fixed tile"),
    };
    let launched = super::gemm_bi_triad::launch_cached_fixed_sm120_exact_tma(
        ctx,
        (shape.m, shape.k, shape.n),
        super::gemm_bi_triad::F32TriadOperands {
            output: operands.c.ptr,
            a: operands.x.ptr,
            b: operands.w.ptr,
            bias: operands.bias_ptr,
            alpha: 1.0,
            beta: 0.0,
        },
        physical_tile,
    )?;
    if !launched {
        return Err("Fixed SM120 exact-TMA symbol is not qualified on this context".into());
    }
    Ok(())
}

fn launch_sm120_tma_postbias(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> Result<(), String> {
    if !fixed_sm120_tma_fma_force_physical_eligible(
        operands,
        shape,
        FixedTileDevice {
            multiprocessors: ctx.kernels.multiprocessor_count(),
            compute_capability: ctx.compute_capability(),
        },
        ctx.f32_triad_policy(),
        tile,
    ) {
        return Err("Fixed SM120 post-bias launch is outside its exact measured contract".into());
    }
    let args = FixedArgs::try_new(operands, shape)?;
    let kernels = ctx
        .kernels
        .fixed_sm120_fma_postbias
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm120_fma_postbias_rejection
                .clone()
                .unwrap_or_else(|| "Fixed SM120 post-bias kernels are not admitted".into())
        })?;
    let (function, tile_m, tile_n) = match tile {
        FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256 => (
            kernels.nobias_m128n64_t256.as_ref().ok_or_else(|| {
                kernels
                    .nobias_m128n64_t256_rejection
                    .clone()
                    .unwrap_or_else(|| "Fixed SM120 no-bias T256 kernel is not admitted".into())
            })?,
            128_u32,
            64_u32,
        ),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64 => (&kernels.m128n64, 128_u32, 64_u32),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => (
            kernels.m128n64_t256.as_ref().ok_or_else(|| {
                kernels
                    .m128n64_t256_rejection
                    .clone()
                    .unwrap_or_else(|| "Fixed SM120 post-bias T256 kernel is not admitted".into())
            })?,
            128_u32,
            64_u32,
        ),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4 => (
            kernels.m128n64_k4.as_ref().ok_or_else(|| {
                kernels
                    .m128n64_k4_rejection
                    .clone()
                    .unwrap_or_else(|| "Fixed SM120 post-bias K4 kernel is not admitted".into())
            })?,
            128_u32,
            64_u32,
        ),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128 => (&kernels.m64n128, 64_u32, 128_u32),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96 => (&kernels.m128n96, 128_u32, 96_u32),
        _ => unreachable!("post-bias launcher received a non-post-bias Fixed tile"),
    };
    let rows = u32::try_from(args.m).map_err(|_| "Fixed post-bias M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "Fixed post-bias N is negative")?;
    let grid = rows
        .div_ceil(tile_m)
        .checked_mul(cols.div_ceil(tile_n))
        .ok_or("Fixed post-bias launch grid exceeds u32")?;
    let maps = fixed_sm120_postbias_maps(ctx, &args, tile_m, tile_n)?;
    let tiles_per_split = args
        .k
        .checked_add(15)
        .ok_or("Fixed post-bias K tile count exceeds i32")?
        / 16;
    let params = FixedSm120PostBiasParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        n: args.n,
        k: args.k,
        ldc: args.n,
        splits: 1,
        tiles_per_split,
    };
    let (threads, shared_mem_bytes) = match tile {
        FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256 => (256, 24_592),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => (256, 24_592),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96 => (256, 28_688),
        _ => (128, 24_592),
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let null_partials = 0_u64;
    let null_flags = 0_u64;
    let bias = args.bias;
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&null_partials)
        .arg(&null_flags)
        .arg(&maps[0])
        .arg(&maps[1])
        .arg(&bias)
        .arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed SM120 post-bias ({tile:?}): {error:?}"))
}

/// Exact F32 owns its compact ABI independently of the half and TF32 bundles.
#[derive(Clone, Copy)]
#[repr(C)]
struct FixedSm89ExactF32Params {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}
unsafe impl DeviceRepr for FixedSm89ExactF32Params {}
pub(crate) const FIXED_SM89_EXACT_F32_PARAMS_SIZE: usize =
    std::mem::size_of::<FixedSm89ExactF32Params>();

impl FixedSm89ExactF32Params {
    fn forward(args: &FixedArgs) -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            m: args.m,
            n: args.n,
            k: args.k,
            lda: args.k,
            ldb: args.n,
            ldc: args.n,
        }
    }
}
const _: () = {
    assert!(FIXED_SM89_EXACT_F32_PARAMS_SIZE == 32);
    assert!(std::mem::align_of::<FixedSm89ExactF32Params>() == 4);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, alpha) == 0);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, beta) == 4);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, m) == 8);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, n) == 12);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, k) == 16);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, lda) == 20);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, ldb) == 24);
    assert!(std::mem::offset_of!(FixedSm89ExactF32Params, ldc) == 28);
};

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

pub(crate) const FIXED_TF32_WIDE_PARAMS_SIZE: usize = std::mem::size_of::<FixedTf32WideParams>();

const _: () = {
    assert!(FIXED_TF32_WIDE_PARAMS_SIZE == 32);
    assert!(std::mem::align_of::<FixedTf32WideParams>() == 4);
    assert!(std::mem::offset_of!(FixedTf32WideParams, alpha) == 0);
    assert!(std::mem::offset_of!(FixedTf32WideParams, beta) == 4);
    assert!(std::mem::offset_of!(FixedTf32WideParams, m) == 8);
    assert!(std::mem::offset_of!(FixedTf32WideParams, k) == 12);
    assert!(std::mem::offset_of!(FixedTf32WideParams, n) == 16);
    assert!(std::mem::offset_of!(FixedTf32WideParams, lda) == 20);
    assert!(std::mem::offset_of!(FixedTf32WideParams, ldb) == 24);
    assert!(std::mem::offset_of!(FixedTf32WideParams, ldc) == 28);
};

/// Ada Fixed half owns a distinct five-argument ABI. Its dimension order is
/// M,N,K, unlike the borrowed Triad TF32 wide bundle's M,K,N order.
#[derive(Clone, Copy)]
#[repr(C)]
struct FixedSm89HalfParams {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for FixedSm89HalfParams {}

pub(crate) const FIXED_SM89_HALF_PARAMS_SIZE: usize = std::mem::size_of::<FixedSm89HalfParams>();
pub(crate) const FIXED_SM89_HALF_SWIZZLE_PARAMS_SIZE: usize =
    std::mem::size_of::<FixedSm89HalfParams>();

const _: () = {
    assert!(FIXED_SM89_HALF_PARAMS_SIZE == 32);
    assert!(std::mem::align_of::<FixedSm89HalfParams>() == 4);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, alpha) == 0);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, beta) == 4);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, m) == 8);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, n) == 12);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, k) == 16);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, lda) == 20);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, ldb) == 24);
    assert!(std::mem::offset_of!(FixedSm89HalfParams, ldc) == 28);
};

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
struct FixedPostBiasMapKey {
    a: CUptr,
    b: CUptr,
    m: usize,
    k: usize,
    n: usize,
    tile_m: u32,
    tile_n: u32,
}

struct FixedPostBiasMapEntry {
    key: FixedPostBiasMapKey,
    maps: [FixedTensorMap; 2],
    epoch: ManagedAllocationEpochStamp,
}

#[derive(Default)]
pub(crate) struct FixedPostBiasMapCache {
    entries: VecDeque<FixedPostBiasMapEntry>,
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

impl FixedPostBiasMapCache {
    fn get(&mut self, key: FixedPostBiasMapKey) -> Option<[FixedTensorMap; 2]> {
        self.entries.retain(|entry| entry.epoch.is_current());
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        let entry = self.entries.remove(index)?;
        let maps = entry.maps;
        self.entries.push_front(entry);
        Some(maps)
    }

    fn insert(
        &mut self,
        key: FixedPostBiasMapKey,
        maps: [FixedTensorMap; 2],
        epoch: ManagedAllocationEpochStamp,
    ) {
        self.entries.retain(|entry| entry.key != key);
        self.entries
            .push_front(FixedPostBiasMapEntry { key, maps, epoch });
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
struct FixedSm120PostBiasParams {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    ldc: i32,
    splits: i32,
    tiles_per_split: i32,
}

unsafe impl DeviceRepr for FixedSm120PostBiasParams {}

pub(crate) const FIXED_SM120_POSTBIAS_PARAMS_SIZE: usize =
    std::mem::size_of::<FixedSm120PostBiasParams>();

const _: () = {
    assert!(FIXED_SM120_POSTBIAS_PARAMS_SIZE == 32);
    assert!(std::mem::align_of::<FixedSm120PostBiasParams>() == 4);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, alpha) == 0);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, beta) == 4);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, m) == 8);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, n) == 12);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, k) == 16);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, ldc) == 20);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, splits) == 24);
    assert!(std::mem::offset_of!(FixedSm120PostBiasParams, tiles_per_split) == 28);
};

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

fn encode_fixed_postbias_map(
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
) -> Result<FixedTensorMap, String> {
    if base == 0 || !base.is_multiple_of(16) {
        return Err("Fixed SM120 post-bias map base must be non-null and 16-byte aligned".into());
    }
    let row_bytes = global_dimensions[0]
        .checked_mul(4)
        .ok_or("Fixed SM120 post-bias map row width overflows")?;
    let inner_bytes = box_dimensions[0]
        .checked_mul(4)
        .ok_or("Fixed SM120 post-bias map box width overflows")?;
    if global_dimensions.contains(&0)
        || box_dimensions.contains(&0)
        || box_dimensions[1] > 256
        || outer_byte_stride < row_bytes
        || !outer_byte_stride.is_multiple_of(16)
        || outer_byte_stride >= (1_u64 << 40)
        || !inner_bytes.is_multiple_of(16)
        || inner_bytes > 1024
    {
        return Err("Fixed SM120 post-bias map dimensions or stride are unsupported".into());
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
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()
        .map_err(|error| format!("Fixed SM120 post-bias tensor-map encode: {error:?}"))?;
        Ok(FixedTensorMap(raw.assume_init()))
    }
}

fn fixed_sm120_postbias_maps(
    ctx: &GpuCtx,
    args: &FixedArgs,
    tile_m: u32,
    tile_n: u32,
) -> Result<[FixedTensorMap; 2], String> {
    let key = FixedPostBiasMapKey {
        a: args.a,
        b: args.b,
        m: args.m as usize,
        k: args.k as usize,
        n: args.n as usize,
        tile_m,
        tile_n,
    };
    if let Some(maps) = ctx.fixed_postbias_maps.borrow_mut().get(key) {
        return Ok(maps);
    }
    if ctx
        .stream
        .capture_status()
        .map_err(|error| format!("query Fixed post-bias capture status: {error:?}"))?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err(
            "Fixed SM120 post-bias tensor maps must be prepared before graph capture".into(),
        );
    }
    let m = u64::try_from(args.m).map_err(|_| "Fixed post-bias M is negative")?;
    let k = u64::try_from(args.k).map_err(|_| "Fixed post-bias K is negative")?;
    let n = u64::try_from(args.n).map_err(|_| "Fixed post-bias N is negative")?;
    let a_stride = k
        .checked_mul(4)
        .ok_or("Fixed post-bias A stride overflows")?;
    let b_stride = n
        .checked_mul(4)
        .ok_or("Fixed post-bias B stride overflows")?;
    let maps = [
        encode_fixed_postbias_map(args.a, [k, m], a_stride, [16, tile_m])?,
        encode_fixed_postbias_map(args.b, [n, k], b_stride, [tile_n, 16])?,
    ];
    let a_bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or("Fixed post-bias A span overflows")?;
    let b_bytes = k
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or("Fixed post-bias B span overflows")?;
    let context_handle = ctx.stream.context().cu_ctx() as usize;
    if let Some(epoch) =
        managed_allocation_epoch_for_ranges(context_handle, &[(args.a, a_bytes), (args.b, b_bytes)])
    {
        ctx.fixed_postbias_maps
            .borrow_mut()
            .insert(key, maps, epoch);
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
    has_bias: bool,
    output_aligned: bool,
    nvrtc_library_known: bool,
) -> FixedTile {
    if rows <= 16 || cols <= 32 {
        return FixedTile::Tf32M16S4;
    }
    // C0/C1 M64S2 beat prior AUTO M128S2 in all eight 101-window eager/graph,
    // candidate-first/incumbent-first cohorts (worst paired p95 0.978250),
    // with identical output bits and physical graph identity. This is scoped
    // to the measured RTX 6000 Ada / known NVRTC13.2 stack:
    // internal/perf/fixed-tf32-all6-ada-20260906/c-m64s2-confirm101.log
    // SHA256 908b0ffd1fd1e5127b51d7e440d530080a6bd07d985e2aa93720f095accd926a.
    if !sm120_tma
        && compute_capability == (8, 9)
        && multiprocessors == 142
        && nvrtc_version == (13, 2)
        && nvrtc_library_known
        && matches!(
            (rows, inner, cols, has_bias),
            (4621, 1928, 384, false) | (4621, 1928, 384, true)
        )
    {
        return FixedTile::Tf32M64S2;
    }
    if sm120_tma {
        if compute_capability == (12, 0) && multiprocessors == 170 && nvrtc_version == (13, 2) {
            match (rows, inner, cols, has_bias) {
                // A0/A1 M128S2 beat prior AUTO and FAST_TF32 in all four
                // 101-window eager/graph AFV/VFA cohorts (worst current p95
                // 0.955694 / 0.953945). Measured aligned, known NVRTC13.2:
                // internal/perf/sm120-fixed-internal-winners-confirm101-20260906/cuda-13.2/
                // fixed-internal-winners-confirm101-cuda13.2-tf32-a01-v1.log
                // SHA256 71d1f97f3733997b63e6463cb88686bd4b996b5eafa8442bcb8136c623b68faf.
                (4621, 384, 1928, _) if output_aligned && nvrtc_library_known => {
                    return FixedTile::Tf32Sm120M128S2;
                }
                (4621, 768, 2304, _) => return FixedTile::Tf32Sm120M128S2,
                // Both D bias rows beat incumbent ProducerWarp in all four
                // 101-window eager/graph, AFV/VFA cohorts. D1 also beats FAST;
                // D0 is an internal-only win; both measured C pointers are
                // 8-byte aligned. Evidence under internal/perf/
                // sm120-fixed-tf32-pair-store-force-20260906/:
                // d0-confirm101-v1.log SHA256
                // 7adbbd340a0aefbebbf8608c5f65f03bb54ca0e668f00ff1199ba495c805866c;
                // d1-confirm101-v1.log SHA256
                // f399427b695d2342d71dd62a764e27b3b7d68263730f77cf781c8ade4810b790.
                (2048, 768, 2304, false) | (2048, 768, 2304, true) => {
                    return if output_aligned {
                        FixedTile::Tf32Sm120M64S2PairStore
                    } else {
                        FixedTile::Tf32Sm120M64S2ProducerWarp
                    };
                }
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
    if tile == FixedTile::Tf32RnaM128N96S3 {
        return launch_tf32_rna_n96(ctx, args);
    }
    if matches!(
        tile,
        FixedTile::Tf32M128N128S3 | FixedTile::Tf32RnaM128N128S3
    ) {
        return launch_tf32_wide(ctx, args, tile == FixedTile::Tf32RnaM128N128S3);
    }
    if matches!(
        tile,
        FixedTile::Tf32Sm120M128S2
            | FixedTile::Tf32Sm120M128S3
            | FixedTile::Tf32Sm120M64N128S2
            | FixedTile::Tf32Sm120M64N128S3
            | FixedTile::Tf32Sm120M64S2ProducerWarp
            | FixedTile::Tf32Sm120M64S2
            | FixedTile::Tf32Sm120M64S2PairStore
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

fn launch_tf32_rna_n96(ctx: &GpuCtx, args: &FixedArgs) -> Result<(), String> {
    if ctx.compute_capability() != (8, 9) {
        return Err("Fixed Ada TF32 RNA N96 requires CC8.9".into());
    }
    if args.k % 4 != 0 || args.n % 4 != 0 {
        return Err("Fixed Ada TF32 RNA N96 requires K and N divisible by four".into());
    }
    args.k
        .checked_add(31)
        .ok_or("Fixed Ada TF32 RNA N96 padded K exceeds i32")?;
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
    {
        return Err("Fixed Ada TF32 RNA N96 requires non-null, 16-byte-aligned A and B".into());
    }
    if args.c == 0 || !args.c.is_multiple_of(4) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed Ada TF32 RNA N96 requires non-null f32-aligned C and f32-aligned bias".into(),
        );
    }
    let rows = u32::try_from(args.m).map_err(|_| "Fixed Ada TF32 RNA N96 M is negative")?;
    let padded_columns = args
        .n
        .checked_add(95)
        .ok_or("Fixed Ada TF32 RNA N96 padded N exceeds i32")?;
    let column_tiles =
        u32::try_from(padded_columns).map_err(|_| "Fixed Ada TF32 RNA N96 N is negative")? / 96;
    let grid = rows
        .div_ceil(128)
        .checked_mul(column_tiles)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada TF32 RNA N96 launch grid exceeds i32")?;
    let function = ctx
        .kernels
        .fixed_sm89_tf32_rna_n96
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm89_tf32_rna_n96_rejection
                .clone()
                .unwrap_or_else(|| "Fixed Ada TF32 RNA N96 symbol is not admitted".into())
        })?;
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
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 86_016,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi forced Fixed Ada TF32 RNA N96: {error:?}"))
}

fn launch_tf32_wide(ctx: &GpuCtx, args: &FixedArgs, fixed_rna: bool) -> Result<(), String> {
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
    let function = if fixed_rna {
        ctx.kernels
            .fixed_sm89_tf32_rna_wide
            .as_ref()
            .ok_or_else(|| {
                ctx.kernels
                    .fixed_sm89_tf32_rna_wide_rejection
                    .clone()
                    .unwrap_or_else(|| "Fixed SM89 RNA-wide symbol is not admitted".into())
            })?
    } else {
        ctx.kernels
            .tf32_function("gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3")
            .ok_or("Fixed TF32 wide Triad symbol is not bound")?
    };
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
        .map_err(|error| {
            let route = if fixed_rna { "RNA-wide" } else { "Triad-wide" };
            format!("gemm_bi forced Fixed TF32 {route}: {error:?}")
        })
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
        FixedTile::Tf32Sm120M64S2PairStore => (&kernels.m64n64_s2_pair_store, 64, 64, 128, 32_896),
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

fn launch_sm89_half_pipeline(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    args: &FixedArgs,
) -> Result<(), String> {
    if ctx.compute_capability() != (8, 9) {
        return Err("Fixed Ada half pipeline requires CC8.9".into());
    }
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    if args.c == 0 || !args.c.is_multiple_of(2) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed Ada half pipeline requires non-null half-aligned C and f32-aligned bias".into(),
        );
    }
    // Scalar staging supports odd strides and independently two-byte-aligned
    // A/B views. K0 never reads A/B, so actual null inputs remain legal.
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|p| p == 0 || !p.is_multiple_of(2))
    {
        return Err("Fixed Ada half pipeline requires non-null half-aligned A and B".into());
    }
    args.m
        .checked_add(127)
        .ok_or("Fixed Ada half padded M exceeds i32")?;
    let padded_n = args
        .n
        .checked_add(127)
        .ok_or("Fixed Ada half padded N exceeds i32")?;
    args.k
        .checked_add(63)
        .ok_or("Fixed Ada half padded K exceeds i32")?;
    let grid = (args.m as u32)
        .div_ceil(128)
        .checked_mul(padded_n as u32 / 128)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada half launch grid exceeds i32")?;
    // Opt-in, live Driver ABI and resource admission happened during module
    // loading. Capture performs no allocation or function reconfiguration.
    let kernels = ctx
        .kernels
        .fixed_sm89_half_pipeline
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm89_half_pipeline_rejection
                .clone()
                .unwrap_or_else(|| "Fixed Ada half pipeline is not admitted".into())
        })?;
    let params = FixedSm89HalfParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        n: args.n,
        k: args.k,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 71_680,
    };
    let mut builder = ctx.stream.launch_builder(kernels.get(dtype));
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed Ada half pipeline: {error:?}"))
}

fn launch_sm89_half_swizzle(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    args: &FixedArgs,
) -> Result<(), String> {
    if ctx.compute_capability() != (8, 9) {
        return Err("Fixed Ada half swizzle requires CC8.9".into());
    }
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    if args.c == 0 || !args.c.is_multiple_of(2) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed Ada half swizzle requires non-null half-aligned C and f32-aligned bias".into(),
        );
    }
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|p| p == 0 || !p.is_multiple_of(2))
    {
        return Err("Fixed Ada half swizzle requires non-null half-aligned A and B".into());
    }
    args.m
        .checked_add(127)
        .ok_or("Fixed Ada half swizzle padded M exceeds i32")?;
    let padded_n = args
        .n
        .checked_add(127)
        .ok_or("Fixed Ada half swizzle padded N exceeds i32")?;
    args.k
        .checked_add(63)
        .ok_or("Fixed Ada half swizzle padded K exceeds i32")?;
    let grid = (args.m as u32)
        .div_ceil(128)
        .checked_mul(padded_n as u32 / 128)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada half swizzle launch grid exceeds i32")?;
    let kernels = ctx
        .kernels
        .fixed_sm89_half_swizzle
        .as_ref()
        .ok_or_else(|| {
            ctx.kernels
                .fixed_sm89_half_swizzle_rejection
                .clone()
                .unwrap_or_else(|| "Fixed Ada half swizzle is not admitted".into())
        })?;
    let params = FixedSm89HalfParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        n: args.n,
        k: args.k,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 69_632,
    };
    let mut builder = ctx.stream.launch_builder(kernels.get(dtype));
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed Ada half swizzle: {error:?}"))
}

fn validate_sm89_half_s3_k(k: i32) -> Result<(), &'static str> {
    // S3 computes (kt + 2) * 64 even when the last iteration does not refill.
    // Protect that lookahead as well as the usual (K + 63) tile-count rounding.
    k.checked_add(127)
        .map(|_| ())
        .ok_or("Fixed Ada half S3 padded K exceeds i32")
}

#[test]
fn sm89_half_s3_rejects_overflow_in_unconditional_refill_index() {
    for k in [2_147_483_521, 2_147_483_584, i32::MAX] {
        assert!(validate_sm89_half_s3_k(k).is_err());
    }
    for k in [0, 64, 128, 192, 256, 2_147_483_520] {
        validate_sm89_half_s3_k(k).expect("safe S3 refill headroom");
        let tiles = (i64::from(k) + 63) / 64;
        assert!((tiles + 1) * 64 <= i64::from(i32::MAX));
    }
}

fn launch_sm89_half_s3(ctx: &GpuCtx, dtype: WeightDtype, args: &FixedArgs) -> Result<(), String> {
    if ctx.compute_capability() != (8, 9) {
        return Err("Fixed Ada half s3 requires CC8.9".into());
    }
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    if args.c == 0 || !args.c.is_multiple_of(2) || !args.bias.is_multiple_of(4) {
        return Err(
            "Fixed Ada half s3 requires non-null half-aligned C and f32-aligned bias".into(),
        );
    }
    if args.k != 0
        && [args.a, args.b]
            .into_iter()
            .any(|p| p == 0 || !p.is_multiple_of(2))
    {
        return Err("Fixed Ada half s3 requires non-null half-aligned A and B".into());
    }
    args.m
        .checked_add(127)
        .ok_or("Fixed Ada half s3 padded M exceeds i32")?;
    let padded_n = args
        .n
        .checked_add(127)
        .ok_or("Fixed Ada half s3 padded N exceeds i32")?;
    validate_sm89_half_s3_k(args.k)?;
    let grid = (args.m as u32)
        .div_ceil(128)
        .checked_mul(padded_n as u32 / 128)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada half s3 launch grid exceeds i32")?;
    let kernels = ctx.kernels.fixed_sm89_half_s3.as_ref().ok_or_else(|| {
        ctx.kernels
            .fixed_sm89_half_s3_rejection
            .clone()
            .unwrap_or_else(|| "Fixed Ada half s3 is not admitted".into())
    })?;
    let params = FixedSm89HalfParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        n: args.n,
        k: args.k,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 98_304,
    };
    let mut builder = ctx.stream.launch_builder(kernels.get(dtype));
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi Fixed Ada half s3: {error:?}"))
}

fn launch_sm89_half_n64(ctx: &GpuCtx, tile: FixedTile, args: &FixedArgs) -> Result<(), String> {
    if ctx.compute_capability() != (8, 9) {
        return Err("Fixed Ada half N64 finalist requires CC8.9".into());
    }
    if args.bias != 0 {
        return Err("Fixed Ada half N64 finalist does not admit bias".into());
    }
    if !(1..=2048).contains(&args.m) {
        return Err("Fixed Ada half N64 finalist requires 1 <= M <= 2048".into());
    }
    if [args.c, args.a, args.b]
        .into_iter()
        .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
    {
        return Err(
            "Fixed Ada half N64 finalist requires non-null, 16-byte-aligned C, A, and B".into(),
        );
    }
    let (function, rejection, bm, k, n) = match tile {
        FixedTile::TcM64N64Sm89S3 => (
            ctx.kernels.fixed_sm89_half_m64n64_s3_f16.as_ref(),
            &ctx.kernels.fixed_sm89_half_m64n64_s3_f16_rejection,
            64_u32,
            768,
            2304,
        ),
        FixedTile::TcM128N64Sm89S2 => (
            ctx.kernels.fixed_sm89_half_m128n64_s2_f16.as_ref(),
            &ctx.kernels.fixed_sm89_half_m128n64_s2_f16_rejection,
            128_u32,
            2304,
            768,
        ),
        _ => return Err(format!("{tile:?} is not a Fixed Ada half N64 finalist")),
    };
    if args.k != k || args.n != n {
        return Err(format!(
            "Fixed Ada half N64 finalist {tile:?} requires exact K={k}, N={n}"
        ));
    }
    let function = function.ok_or_else(|| {
        rejection
            .clone()
            .unwrap_or_else(|| format!("Fixed Ada half N64 finalist {tile:?} is not admitted"))
    })?;
    let rows = u32::try_from(args.m).map_err(|_| "Fixed Ada half N64 M is negative")?;
    let columns = u32::try_from(args.n).map_err(|_| "Fixed Ada half N64 N is negative")?;
    let grid = rows
        .div_ceil(bm)
        .checked_mul(columns / 64)
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("Fixed Ada half N64 launch grid exceeds i32")?;
    let params = FixedSm89HalfParams {
        alpha: 1.0,
        beta: 0.0,
        m: args.m,
        n: args.n,
        k: args.k,
        lda: args.k,
        ldb: args.n,
        ldc: args.n,
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 49_152,
    };
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("gemm_bi forced Fixed Ada half N64 ({tile:?}): {error:?}"))
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
        | FixedTile::F32Sm89N64CopyPlan
        | FixedTile::F32Sm120N64CopyPlan
        | FixedTile::F32Sm120N64CopyPlanT256
        | FixedTile::F32Sm120M128N64CopyPlanT256
        | FixedTile::Tf32M128S2
        | FixedTile::F32Sm120N64Sliced
        | FixedTile::F32Sm120TmaFmaM128N64
        | FixedTile::F32Sm120TmaFmaM64N128
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        | FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96
        | FixedTile::Tf32M128S3
        | FixedTile::Tf32M128N128S3
        | FixedTile::Tf32RnaM128N128S3
        | FixedTile::Tf32RnaM128N96S3
        | FixedTile::Tf32M64S2
        | FixedTile::Tf32M64S3
        | FixedTile::Tf32M16S4
        | FixedTile::Tf32Sm120M128S2
        | FixedTile::Tf32Sm120M128S3
        | FixedTile::Tf32Sm120M64N128S2
        | FixedTile::Tf32Sm120M64N128S3
        | FixedTile::Tf32Sm120M64S2ProducerWarp
        | FixedTile::Tf32Sm120M64S2
        | FixedTile::Tf32Sm120M64S2PairStore
        | FixedTile::Sm120Half(_)
        | FixedTile::Tc128Sm89Pipeline
        | FixedTile::Tc128Sm89Swizzle
        | FixedTile::Tc128Sm89S3
        | FixedTile::TcM64N64Sm89S3
        | FixedTile::TcM128N64Sm89S2 => unreachable!("tile has its own launcher"),
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

// Production NVRTC12.8/13.0/13.2 on 142-SM Ada, A-E and both bias states.
// The complete force corpus is qualified in
// internal/perf/ada-rna-full-toolkit-qualification-20260906/. The paired101
// logs are internal/perf/ada-rna-toolkit-census-20260906/cuda-{12.8,13.0}/
// timing101.log (SHA256 955f252909c94607e359e2a2eeb47e9744f76c41a8f2fac128ef338a80e15dcb
// and d687caefea0b6c85a378f99230b4ee7f44c55a40e7ac6bfe1980f9967d3b0a37)
// and internal/perf/ada-rna-wide-auto-20260906/
// confirm101-old-m64-vs-auto-rna-final.log (SHA256
// 3dbb8754b651706d67c83ab1f96fb26ed66a63bc161e8ced015f36ef50831f46).
// RNA wins all ten internal p50/p95 cohorts on each toolkit; FAST wins are
// only A1 on12.8/13.0 and A1/B1 on13.2. C16 is the measured output alignment;
// forced RNA supports C4, but those views retain the ordinary AUTO picker.
fn fixed_sm89_rna_wide_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc_version: (i32, i32),
    nvrtc_library_known: bool,
    admitted: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    admitted
        && policy == super::context::F32TriadPolicy::AllowDeterministicTf32V1
        && device.compute_capability == (8, 9)
        && device.multiprocessors == 142
        && matches!(nvrtc_version, (12, 8) | (13, 0) | (13, 2))
        && nvrtc_library_known
        && [operands.c.dtype, operands.x.dtype, operands.w.dtype]
            .into_iter()
            .all(|dtype| dtype == WeightDtype::F32)
        && [operands.c.ptr, operands.x.ptr, operands.w.ptr]
            .into_iter()
            .all(|ptr| ptr != 0 && ptr.is_multiple_of(16))
        && operands.bias_ptr.unwrap_or(0).is_multiple_of(4)
        && matches!(
            (shape.m, shape.k, shape.n),
            (4621, 384, 1928)
                | (4621, 768, 2304)
                | (4621, 1928, 384)
                | (2048, 768, 2304)
                | (2048, 2304, 768)
        )
}

// The N96 finalist replaces the wide RNA route only at the measured E0/no-bias
// row. If its independently admitted holder is absent, the wide selector below
// remains available without changing the numeric family.
fn fixed_sm89_rna_n96_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc_version: (i32, i32),
    nvrtc_library_known: bool,
    admitted: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    fixed_sm89_rna_wide_auto_eligible(
        operands,
        shape,
        device,
        nvrtc_version,
        nvrtc_library_known,
        admitted,
        policy,
    ) && (shape.m, shape.k, shape.n) == (2048, 2304, 768)
        && operands.bias_ptr.is_none()
}

#[cfg(test)]
mod sm89_rna_auto_tests {
    use super::super::context::F32TriadPolicy;
    use super::*;

    #[test]
    fn fixed_sm89_rna_auto_exact_cells_and_independent_declines() {
        let ptr = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        let base = FixedFwdOperands {
            c: ptr(0x1000),
            x: ptr(0x2000),
            w: ptr(0x3000),
            bias_ptr: None,
        };
        let device = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        let policy = F32TriadPolicy::AllowDeterministicTf32V1;
        let eligible = |o, s, d, v, known, admitted, p| {
            fixed_sm89_rna_wide_auto_eligible(o, s, d, v, known, admitted, p)
        };
        for (m, k, n) in [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ] {
            let shape = FixedShape { m, k, n };
            for bias_ptr in [None, Some(0x4000)] {
                let operands = FixedFwdOperands { bias_ptr, ..base };
                for nvrtc in [(12, 8), (13, 0), (13, 2)] {
                    assert!(eligible(operands, shape, device, nvrtc, true, true, policy));
                }
                assert!(
                    !eligible(operands, shape, device, (13, 2), true, false, policy),
                    "missing holder"
                );
                assert!(
                    !eligible(operands, shape, device, (13, 2), false, true, policy),
                    "unknown library"
                );
                assert!(!eligible(
                    operands,
                    shape,
                    device,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
                for cc in [(8, 0), (8, 6), (9, 0), (10, 0), (12, 0), (12, 1)] {
                    assert!(!eligible(
                        operands,
                        shape,
                        FixedTileDevice {
                            compute_capability: cc,
                            ..device
                        },
                        (13, 2),
                        true,
                        true,
                        policy
                    ));
                }
                for sms in [141, 143] {
                    assert!(!eligible(
                        operands,
                        shape,
                        FixedTileDevice {
                            multiprocessors: sms,
                            ..device
                        },
                        (13, 2),
                        true,
                        true,
                        policy
                    ));
                }
                for nvrtc in [(12, 7), (12, 9), (13, 1), (13, 3), (14, 0)] {
                    assert!(!eligible(
                        operands, shape, device, nvrtc, true, true, policy
                    ));
                }
                for field in 0..3 {
                    for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
                        let mut bad = operands;
                        match field {
                            0 => bad.c.dtype = dtype,
                            1 => bad.x.dtype = dtype,
                            _ => bad.w.dtype = dtype,
                        }
                        assert!(
                            !eligible(bad, shape, device, (13, 2), true, true, policy),
                            "wrong dtype field={field}"
                        );
                    }
                    for address in [0, 0x1001, 0x1004, 0x1008, 0x100c] {
                        let mut bad = operands;
                        match field {
                            0 => bad.c.ptr = address,
                            1 => bad.x.ptr = address,
                            _ => bad.w.ptr = address,
                        }
                        assert!(
                            !eligible(bad, shape, device, (13, 2), true, true, policy),
                            "null/misaligned field={field} address={address:x}"
                        );
                    }
                }
                for bias in [0x4001, 0x4002, 0x4003] {
                    assert!(!eligible(
                        FixedFwdOperands {
                            bias_ptr: Some(bias),
                            ..operands
                        },
                        shape,
                        device,
                        (13, 2),
                        true,
                        true,
                        policy
                    ));
                }
                for bad in [
                    FixedShape { m: m - 1, ..shape },
                    FixedShape { m: m + 1, ..shape },
                    FixedShape { k: k - 1, ..shape },
                    FixedShape { k: k + 1, ..shape },
                    FixedShape { n: n - 1, ..shape },
                    FixedShape { n: n + 1, ..shape },
                    FixedShape { m: 0, ..shape },
                    FixedShape { m: 1, ..shape },
                    FixedShape { m: 16, ..shape },
                    FixedShape { k: 0, ..shape },
                ] {
                    assert!(
                        !eligible(operands, bad, device, (13, 2), true, true, policy),
                        "unqualified shape {bad:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_sm89_rna_n96_auto_v45_is_exactly_e0_all_three_toolkits() {
        let ptr = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        let operands = FixedFwdOperands {
            c: ptr(0x1000),
            x: ptr(0x2000),
            w: ptr(0x3000),
            bias_ptr: None,
        };
        let shape = FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        };
        let device = FixedTileDevice {
            multiprocessors: 142,
            compute_capability: (8, 9),
        };
        let policy = F32TriadPolicy::AllowDeterministicTf32V1;
        for nvrtc in [(12, 8), (13, 0), (13, 2)] {
            assert!(fixed_sm89_rna_n96_auto_eligible(
                operands, shape, device, nvrtc, true, true, policy,
            ));
            assert!(
                !fixed_sm89_rna_n96_auto_eligible(
                    operands, shape, device, nvrtc, true, false, policy,
                ),
                "missing N96 holder must decline independently"
            );
            assert!(
                fixed_sm89_rna_wide_auto_eligible(
                    operands, shape, device, nvrtc, true, true, policy,
                ),
                "missing N96 must retain the old admitted RNA-wide fallback"
            );
        }
        for nvrtc in [(12, 7), (12, 9), (13, 1), (13, 3)] {
            assert!(!fixed_sm89_rna_n96_auto_eligible(
                operands, shape, device, nvrtc, true, true, policy,
            ));
        }
        assert!(!fixed_sm89_rna_n96_auto_eligible(
            FixedFwdOperands {
                bias_ptr: Some(0x4000),
                ..operands
            },
            shape,
            device,
            (13, 2),
            true,
            true,
            policy,
        ));
        assert!(!fixed_sm89_rna_n96_auto_eligible(
            operands,
            FixedShape { m: 2047, ..shape },
            device,
            (13, 2),
            true,
            true,
            policy,
        ));
    }
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
        let compiler = ctx.kernels.compiler_identity();
        let tile = if fixed_sm89_rna_n96_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm89_tf32_rna_n96.is_some(),
            ctx.f32_triad_policy(),
        ) {
            FixedTile::Tf32RnaM128N96S3
        } else if fixed_sm89_rna_wide_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm89_tf32_rna_wide.is_some(),
            ctx.f32_triad_policy(),
        ) {
            FixedTile::Tf32RnaM128N128S3
        } else {
            fixed_pick_tf32(
                batch,
                n_in,
                n_out,
                ctx.kernels.multiprocessor_count(),
                ctx.compute_capability(),
                ctx.kernels.compiler_identity().nvrtc_version,
                sm120_tma,
                operands.bias_ptr.is_some(),
                operands.c.ptr.is_multiple_of(8),
                ctx.kernels.compiler_identity().nvrtc_library_known,
            )
        };
        launch_tf32(ctx, tile, &args, true)?;
        return Ok(tile);
    }
    if homogeneous_f32 {
        let compiler = ctx.kernels.compiler_identity();
        if fixed_sm120_tma_fma_a0_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.f32_triad_policy(),
        ) {
            let tile = FixedTile::F32Sm120TmaFmaM64N128;
            if super::gemm_bi_triad::launch_cached_fixed_sm120_exact_tma(
                ctx,
                (shape.m, shape.k, shape.n),
                super::gemm_bi_triad::F32TriadOperands {
                    output: operands.c.ptr,
                    a: operands.x.ptr,
                    b: operands.w.ptr,
                    bias: None,
                    alpha: 1.0,
                    beta: 0.0,
                },
                super::gemm_bi_triad::FixedSm120ExactTmaTile::M64N128,
            )? {
                return Ok(tile);
            }
        } else if let Some(tile) = fixed_sm120_tma_fma_a1_auto_tile(
            fixed_sm120_tma_fma_a1_auto_eligible(
                operands,
                shape,
                FixedTileDevice {
                    multiprocessors: ctx.kernels.multiprocessor_count(),
                    compute_capability: ctx.compute_capability(),
                },
                compiler.nvrtc_version,
                compiler.nvrtc_library_known,
                ctx.f32_triad_policy(),
            ),
            ctx.kernels
                .fixed_sm120_fma_postbias
                .as_ref()
                .map(|kernels| kernels.m128n64_t256.is_some()),
        ) {
            launch_sm120_tma_postbias(ctx, operands, shape, tile)?;
            return Ok(tile);
        } else if let Some(tile) = fixed_sm120_tma_fma_bc1_auto_tile(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm120_fma_postbias.is_some(),
            ctx.f32_triad_policy(),
        ) {
            launch_sm120_tma_postbias(ctx, operands, shape, tile)?;
            return Ok(tile);
        } else if fixed_sm120_tma_fma_b0_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.f32_triad_policy(),
        ) {
            let tile = FixedTile::F32Sm120TmaFmaM128N64;
            if super::gemm_bi_triad::launch_cached_fixed_sm120_exact_tma(
                ctx,
                (shape.m, shape.k, shape.n),
                super::gemm_bi_triad::F32TriadOperands {
                    output: operands.c.ptr,
                    a: operands.x.ptr,
                    b: operands.w.ptr,
                    bias: None,
                    alpha: 1.0,
                    beta: 0.0,
                },
                super::gemm_bi_triad::FixedSm120ExactTmaTile::M128N64,
            )? {
                return Ok(tile);
            }
        }
        if fixed_sm120_sliced_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm120_f32_n64_sliced.is_some(),
            ctx.f32_triad_policy(),
        ) {
            launch_sm120_sliced(ctx, operands, shape)?;
            return Ok(FixedTile::F32Sm120N64Sliced);
        }
        if fixed_sm120_exact_n64_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm120_f32_n64_copyplan.is_some(),
            ctx.f32_triad_policy(),
        ) {
            launch_sm120_exact_n64(ctx, operands, shape)?;
            return Ok(FixedTile::F32Sm120N64CopyPlan);
        }
        if fixed_sm89_exact_n64_auto_eligible(
            operands,
            shape,
            FixedTileDevice {
                multiprocessors: ctx.kernels.multiprocessor_count(),
                compute_capability: ctx.compute_capability(),
            },
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm89_f32_n64_copyplan.is_some(),
            ctx.f32_triad_policy(),
        ) {
            launch_sm89_exact_n64(ctx, operands, shape)?;
            return Ok(FixedTile::F32Sm89N64CopyPlan);
        }
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
    let compiler = ctx.kernels.compiler_identity();
    if homogeneous_half {
        let device = FixedTileDevice {
            multiprocessors: ctx.kernels.multiprocessor_count(),
            compute_capability: ctx.compute_capability(),
        };
        let selected = fixed_select_sm89_half_finalist_auto_tile(
            operands,
            shape,
            device,
            compiler.nvrtc_version,
            compiler.nvrtc_library_known,
            ctx.kernels.fixed_sm89_half_m64n64_s3_f16.is_some(),
            ctx.kernels.fixed_sm89_half_m128n64_s2_f16.is_some(),
        )
        .or_else(|| {
            fixed_select_sm89_half_auto_tile(
                operands,
                shape,
                device,
                compiler.nvrtc_version,
                compiler.nvrtc_library_known,
                ctx.kernels.fixed_sm89_half_pipeline.is_some(),
                ctx.kernels.fixed_sm89_half_swizzle.is_some(),
                ctx.kernels.fixed_sm89_half_s3.is_some(),
            )
        });
        match selected {
            Some(FixedTile::Tc128Sm89Pipeline) => {
                launch_sm89_half_pipeline(ctx, c.dtype, &args)?;
                return Ok(FixedTile::Tc128Sm89Pipeline);
            }
            Some(FixedTile::Tc128Sm89Swizzle) => {
                launch_sm89_half_swizzle(ctx, c.dtype, &args)?;
                return Ok(FixedTile::Tc128Sm89Swizzle);
            }
            Some(FixedTile::Tc128Sm89S3) => {
                launch_sm89_half_s3(ctx, c.dtype, &args)?;
                return Ok(FixedTile::Tc128Sm89S3);
            }
            Some(FixedTile::TcM64N64Sm89S3) => {
                launch_sm89_half_n64(ctx, FixedTile::TcM64N64Sm89S3, &args)?;
                return Ok(FixedTile::TcM64N64Sm89S3);
            }
            Some(FixedTile::TcM128N64Sm89S2) => {
                launch_sm89_half_n64(ctx, FixedTile::TcM128N64Sm89S2, &args)?;
                return Ok(FixedTile::TcM128N64Sm89S2);
            }
            Some(_) => unreachable!("Ada half AUTO selector returned a foreign tile"),
            None => {}
        }
    }
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

// No sliced AUTO rows are currently admitted. Standalone NVCC won B0, but
// production NVRTC 101-window PEDANTIC p95 was 1.005009/1.000447 eager and
// 1.013953/1.013734 graph. Retain force qualification only; the separately
// qualified exact-TMA bridge owns B0. Any future sliced admission must pass
// the production paired gate afresh.
fn fixed_sm120_sliced_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    loaded: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    const QUALIFIED_ROWS: &[FixedShape] = &[];
    QUALIFIED_ROWS.contains(&shape)
        && operands.bias_ptr.is_none()
        && fixed_sm120_exact_n64_auto_eligible(
            operands,
            shape,
            device,
            nvrtc,
            nvrtc_library_known,
            loaded,
            policy,
        )
}

// Production NVRTC13.2 / 170-SM CC12.0 Fixed B0 qualification. The forced
// bridge retained exact scalar-FMA bits and beat both the prior copy-plan and
// CUBLAS_COMPUTE_32F_PEDANTIC in all eager/graph ABBA/BAAB cohorts at 101
// windows. Worst paired p95: 0.894265 versus copy-plan, 0.909778 versus
// cuBLAS. Evidence: internal/perf/sm120-fixed-tma-fma-b0-20260906/
// sm120-fixed-auto-b0-post101-v1.log, SHA256
// 20245b1fe378434130f0e74bdfe3117790506afccf34224c27fd516bdf480932.
// B1 has a separate post-dot-bias admission below.
fn fixed_sm120_tma_fma_b0_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    fixed_sm120_tma_fma_force_eligible(
        operands,
        shape,
        device,
        nvrtc,
        nvrtc_library_known,
        policy,
        FixedTile::F32Sm120TmaFmaM128N64,
    ) && shape
        == (FixedShape {
            m: 4_621,
            k: 768,
            n: 2_304,
        })
        && operands.bias_ptr.is_none()
}

// Production NVRTC13.2 / 170-SM CC12.0 Fixed A0 qualification. The literal
// M64xN128 one-split route retained exact Fixed bits and won all eager/graph
// ABBA/BAAB cohorts at 101 windows. Before promotion its worst paired p95 was
// 0.953899 versus prior AUTO and 0.881677 versus PEDANTIC cuBLAS; evidence:
// internal/perf/sm120-fixed-tma-fma-a0-20260906/
// sm120-fixed-fma-a0-m64n128-admission101-v1.log, SHA256
// 7a8a46e2eee45109d2d670bd73f62a66fbbc2a28f8b92b474c4cfcb2a27e53f2.
// Post-AUTO worst p95 was 0.895330 versus Legacy and 0.880051 versus
// PEDANTIC cuBLAS; sm120-fixed-fma-a0-postauto101-v1.log, SHA256
// 0974cae2ed67c3b1c1b0325c662b882b7d832f09c8c2f7ca6644675e3c5f410c.
// Bias is excluded because Triad seeds it before the FMA chain while Fixed
// adds it after the dot product.
fn fixed_sm120_tma_fma_a0_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    fixed_sm120_tma_fma_force_eligible(
        operands,
        shape,
        device,
        nvrtc,
        nvrtc_library_known,
        policy,
        FixedTile::F32Sm120TmaFmaM64N128,
    ) && shape
        == (FixedShape {
            m: 4_621,
            k: 384,
            n: 1_928,
        })
        && operands.bias_ptr.is_none()
}

// Production NVRTC13.2 / 170-SM CC12.0 Fixed A1 qualification. The dedicated
// post-dot-bias M128xN64 route preserves the Fixed arithmetic order and won
// every eager/graph ABBA/BAAB cohort at 101 windows. Before promotion its
// worst paired p95 was 0.978964 versus prior AUTO, 0.917294 versus Legacy,
// and 0.765625 versus PEDANTIC cuBLAS; evidence:
// internal/perf/sm120-fixed-tma-fma-a1-postbias-20260906/
// sm120-fixed-fma-a1-postbias-m128-admission101-v1.log, SHA256
// 8887b5a16c246a07d55a47a3b442642bbb53d53ec75565098e9acff6f53ac7f4.
// Post-AUTO worst p95 was 0.914063 versus Legacy and 0.765326 versus
// PEDANTIC cuBLAS; sm120-fixed-fma-a1-postbias-postauto101-v1.log, SHA256
// 7af2dd4afc99448ec530102f2b7aaec771bbf9833dbd97c033de8ef91d561ac4.
fn fixed_sm120_tma_fma_a1_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    fixed_sm120_tma_fma_force_eligible(
        operands,
        shape,
        device,
        nvrtc,
        nvrtc_library_known,
        policy,
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64,
    ) && shape
        == (FixedShape {
            m: 4_621,
            k: 384,
            n: 1_928,
        })
        && operands.bias_ptr.is_some()
}

// The holder retains the qualified control even if T256 resource admission
// declines. None denotes a missing holder; Some(false) selects that control.
fn fixed_sm120_tma_fma_a1_auto_tile(
    eligible: bool,
    t256_loaded: Option<bool>,
) -> Option<FixedTile> {
    if !eligible {
        return None;
    }
    t256_loaded.map(|loaded| {
        if loaded {
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        } else {
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64
        }
    })
}

fn fixed_sm120_tma_fma_bc1_auto_tile(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    loaded: bool,
    policy: super::context::F32TriadPolicy,
) -> Option<FixedTile> {
    // Independent 101-window eager/graph AFV/VFA confirmations, all exact
    // AUTO bits. B1 worst current p95 0.913542, PEDANTIC 0.856601; C1 worst
    // current p95 0.936753, but PEDANTIC 1.042524 (internal-only win).
    // internal/perf/sm120-fixed-internal-winners-confirm101-20260906/cuda-13.2/
    // fixed-internal-winners-confirm101-cuda13.2-f32_exact-b1-v1.log SHA256
    // 59131d6904855513ce50aff4bcf3c7a1e35243cef71416fd465a5b0c102f9796;
    // fixed-internal-winners-confirm101-cuda13.2-f32_exact-c1-v1.log SHA256
    // 164c083bd962c54402b6360104aaba8a7324f1b79be1cdedd6ec0a6a704ac2e1.
    let tile = match (shape.m, shape.k, shape.n) {
        (4621, 768, 2304) => FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        (4621, 1928, 384) => FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96,
        _ => return None,
    };
    (loaded
        && fixed_sm120_tma_fma_force_eligible(
            operands,
            shape,
            device,
            nvrtc,
            nvrtc_library_known,
            policy,
            tile,
        ))
    .then_some(tile)
}

// Physical force-only surface for same-run qualification. This deliberately
// excludes compiler-version and measured-device-cohort gates: holder loading
// remains responsible for toolkit/resource admission, while these checks keep
// forced launches inside the kernel's CC, row, dtype, alignment, policy, and
// bias contract. AUTO remains independently bound to the fully qualified
// NVRTC13.2 / 170-SM cohort below.
fn fixed_sm120_tma_fma_force_physical_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    policy: super::context::F32TriadPolicy,
    tile: FixedTile,
) -> bool {
    const HOT_ROWS: &[FixedShape] = &[
        FixedShape {
            m: 4_621,
            k: 384,
            n: 1_928,
        },
        FixedShape {
            m: 4_621,
            k: 768,
            n: 2_304,
        },
        FixedShape {
            m: 4_621,
            k: 1_928,
            n: 384,
        },
        FixedShape {
            m: 2_048,
            k: 768,
            n: 2_304,
        },
        FixedShape {
            m: 2_048,
            k: 2_304,
            n: 768,
        },
    ];
    let bias_contract = match tile {
        FixedTile::F32Sm120N64CopyPlanT256 | FixedTile::F32Sm120M128N64CopyPlanT256 => operands
            .bias_ptr
            .is_none_or(|bias| bias != 0 && bias.is_multiple_of(4)),
        FixedTile::F32Sm120TmaFmaM128N64
        | FixedTile::F32Sm120TmaFmaM64N128
        | FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256 => operands.bias_ptr.is_none(),
        FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        | FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96 => operands
            .bias_ptr
            .is_some_and(|bias| bias != 0 && bias.is_multiple_of(4)),
        _ => false,
    };
    device.compute_capability == (12, 0)
        && policy == super::context::F32TriadPolicy::ExactScalarFmaV1
        && HOT_ROWS.contains(&shape)
        && bias_contract
        && [operands.c, operands.x, operands.w]
            .into_iter()
            .all(|operand| {
                operand.dtype == WeightDtype::F32
                    && operand.ptr != 0
                    && operand.ptr.is_multiple_of(16)
            })
}

fn fixed_sm120_tma_fma_force_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    policy: super::context::F32TriadPolicy,
    tile: FixedTile,
) -> bool {
    nvrtc_library_known
        && nvrtc == (13, 2)
        && device.multiprocessors == 170
        && fixed_sm120_tma_fma_force_physical_eligible(operands, shape, device, policy, tile)
}

// Actual production NVRTC13.2 / 170-SM CC12.0 paired qualification, 101
// windows in both orders and both eager/graph paths: exactly E/B, each bias
// row. All four rows beat both incumbent AUTO and explicit Legacy. B+bias
// additionally beats CUBLAS_COMPUTE_32F_PEDANTIC in all four paired cohorts.
// Evidence: sm120-fixed-eb-admission101-20260906-v1.jsonl, SHA256
// 62c96372caf6ea8fa6020299d266fb8303c99674d1ca927f1faa4ae91b45249d.
// D0/D1 additionally qualified by 101-window confirmation; candidate/current
// worst p95 ratios: D0 0.830458925, D1 0.819282926. Evidence under
// internal/perf/sm120-fixed-e0-copyplan-t256-20260906/:
// d0-current-copyplan-fast-confirm101-v1.log, SHA256
// de1144e0a91626fe254fe89dde92f8ca956ecedb52aec109bbdca046b91812bb;
// d1-current-copyplan-fast-confirm101-v1.log, SHA256
// 479d08e9d78f3439bb9a7e9a504d074d8d1952e8b79969a7297eacd78f3ff0e8.
fn fixed_sm120_exact_n64_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    loaded: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    loaded
        && nvrtc_library_known
        && nvrtc == (13, 2)
        && device.compute_capability == (12, 0)
        && device.multiprocessors == 170
        && policy == super::context::F32TriadPolicy::ExactScalarFmaV1
        && [operands.c, operands.x, operands.w]
            .into_iter()
            .all(|operand| {
                operand.dtype == WeightDtype::F32
                    && operand.ptr != 0
                    && operand.ptr.is_multiple_of(16)
            })
        && operands
            .bias_ptr
            .is_none_or(|ptr| ptr != 0 && ptr.is_multiple_of(4))
        && matches!(
            (shape.m, shape.k, shape.n, operands.bias_ptr.is_some()),
            (2048, 2304, 768, false)
                | (2048, 2304, 768, true)
                | (4621, 768, 2304, false)
                | (4621, 768, 2304, true)
                | (2048, 768, 2304, false)
                | (2048, 768, 2304, true)
        )
}

#[cfg(test)]
mod sm120_exact_n64_auto_tests {
    use super::super::context::F32TriadPolicy;
    use super::*;

    const DEVICE: FixedTileDevice = FixedTileDevice {
        multiprocessors: 170,
        compute_capability: (12, 0),
    };

    fn operands(bias: bool) -> FixedFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        FixedFwdOperands {
            c: typed(0x1000),
            x: typed(0x2000),
            w: typed(0x3000),
            bias_ptr: bias.then_some(0x4004),
        }
    }

    #[test]
    fn fixed_sm120_bc1_postbias_auto_promotes_only_confirmed_tiles() {
        for (shape, expected) in [
            (
                FixedShape {
                    m: 4621,
                    k: 768,
                    n: 2304,
                },
                FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64,
            ),
            (
                FixedShape {
                    m: 4621,
                    k: 1928,
                    n: 384,
                },
                FixedTile::F32Sm120TmaFmaFixedPostBiasM128N96,
            ),
        ] {
            let pick = |ops, shape, device, nvrtc, known, loaded, policy| {
                fixed_sm120_tma_fma_bc1_auto_tile(ops, shape, device, nvrtc, known, loaded, policy)
            };
            assert_eq!(
                pick(
                    operands(true),
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ),
                Some(expected)
            );
            assert_eq!(
                pick(
                    operands(false),
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ),
                None
            );
            for nvrtc in [(12, 8), (13, 0), (13, 1), (13, 3), (14, 0)] {
                assert_eq!(
                    pick(
                        operands(true),
                        shape,
                        DEVICE,
                        nvrtc,
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ),
                    None
                );
            }
            for (known, loaded) in [(false, true), (true, false), (false, false)] {
                assert_eq!(
                    pick(
                        operands(true),
                        shape,
                        DEVICE,
                        (13, 2),
                        known,
                        loaded,
                        F32TriadPolicy::ExactScalarFmaV1
                    ),
                    None
                );
            }
            for device in [
                FixedTileDevice {
                    multiprocessors: 169,
                    ..DEVICE
                },
                FixedTileDevice {
                    compute_capability: (12, 1),
                    ..DEVICE
                },
            ] {
                assert_eq!(
                    pick(
                        operands(true),
                        shape,
                        device,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ),
                    None
                );
            }
            for bad in [
                FixedFwdOperands {
                    c: TypedPtr {
                        ptr: 0x1004,
                        ..operands(true).c
                    },
                    ..operands(true)
                },
                FixedFwdOperands {
                    x: TypedPtr {
                        ptr: 0x2008,
                        ..operands(true).x
                    },
                    ..operands(true)
                },
                FixedFwdOperands {
                    w: TypedPtr {
                        ptr: 0x300c,
                        ..operands(true).w
                    },
                    ..operands(true)
                },
                FixedFwdOperands {
                    bias_ptr: Some(0),
                    ..operands(true)
                },
                FixedFwdOperands {
                    bias_ptr: Some(0x4001),
                    ..operands(true)
                },
                FixedFwdOperands {
                    c: TypedPtr {
                        dtype: WeightDtype::F16,
                        ..operands(true).c
                    },
                    ..operands(true)
                },
            ] {
                assert_eq!(
                    pick(
                        bad,
                        shape,
                        DEVICE,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ),
                    None
                );
            }
            assert_eq!(
                pick(
                    operands(true),
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::AllowDeterministicTf32V1
                ),
                None
            );
            for adjacent in [
                FixedShape {
                    m: shape.m + 1,
                    ..shape
                },
                FixedShape {
                    k: shape.k + 4,
                    ..shape
                },
                FixedShape {
                    n: shape.n + 4,
                    ..shape
                },
            ] {
                assert_eq!(
                    pick(
                        operands(true),
                        adjacent,
                        DEVICE,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ),
                    None
                );
            }
        }
        for (m, k, n) in [(4621, 384, 1928), (2048, 768, 2304), (2048, 2304, 768)] {
            assert_eq!(
                fixed_sm120_tma_fma_bc1_auto_tile(
                    operands(true),
                    FixedShape { m, k, n },
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ),
                None
            );
        }
    }

    #[test]
    fn fixed_sm120_force_physical_gate_is_independent_of_nvrtc_and_sm_cohort() {
        let shape = FixedShape {
            m: 4_621,
            k: 768,
            n: 2_304,
        };
        let tile = FixedTile::F32Sm120TmaFmaM128N64;
        for multiprocessors in [1, 169, 170, 256] {
            assert!(fixed_sm120_tma_fma_force_physical_eligible(
                operands(false),
                shape,
                FixedTileDevice {
                    multiprocessors,
                    compute_capability: (12, 0),
                },
                F32TriadPolicy::ExactScalarFmaV1,
                tile,
            ));
        }

        // The production AUTO gate remains bound to its qualified compiler and
        // device cohort even though force-only screening is not.
        assert!(!fixed_sm120_tma_fma_force_eligible(
            operands(false),
            shape,
            FixedTileDevice {
                multiprocessors: 169,
                compute_capability: (12, 0),
            },
            (13, 0),
            false,
            F32TriadPolicy::ExactScalarFmaV1,
            tile,
        ));

        assert!(!fixed_sm120_tma_fma_force_physical_eligible(
            operands(false),
            shape,
            FixedTileDevice {
                multiprocessors: 170,
                compute_capability: (8, 9),
            },
            F32TriadPolicy::ExactScalarFmaV1,
            tile,
        ));
        assert!(!fixed_sm120_tma_fma_force_physical_eligible(
            operands(true),
            shape,
            DEVICE,
            F32TriadPolicy::ExactScalarFmaV1,
            tile,
        ));
    }

    #[test]
    fn fixed_sm120_sliced_auto_is_withdrawn_and_copyplan_retained() {
        for (m, k, n, bias, want) in [
            (4621, 768, 2304, false, false),
            (4621, 768, 2304, true, false),
            (2048, 2304, 768, false, false),
            (2048, 2304, 768, true, false),
            (4621, 384, 1928, false, false),
            (4621, 1928, 384, false, false),
            (2048, 768, 2304, false, false),
        ] {
            assert_eq!(
                fixed_sm120_sliced_auto_eligible(
                    operands(bias),
                    FixedShape { m, k, n },
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ),
                want,
                "{m}/{k}/{n}/{bias}"
            );
        }
        assert!(
            fixed_sm120_exact_n64_auto_eligible(
                operands(false),
                FixedShape {
                    m: 4621,
                    k: 768,
                    n: 2304
                },
                DEVICE,
                (13, 2),
                true,
                true,
                F32TriadPolicy::ExactScalarFmaV1
            ),
            "B0 copyplan must remain available as the qualified fallback"
        );
    }

    #[test]
    fn fixed_sm120_tma_fma_b0_gate_is_exact_and_cohort_bound() {
        let b0 = FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        };
        assert!(fixed_sm120_tma_fma_b0_auto_eligible(
            operands(false),
            b0,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        assert!(!fixed_sm120_tma_fma_b0_auto_eligible(
            operands(true),
            b0,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        for shape in [
            FixedShape { m: 4620, ..b0 },
            FixedShape { m: 4622, ..b0 },
            FixedShape { k: 767, ..b0 },
            FixedShape { k: 769, ..b0 },
            FixedShape { n: 2303, ..b0 },
            FixedShape { n: 2305, ..b0 },
            FixedShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ] {
            assert!(!fixed_sm120_tma_fma_b0_auto_eligible(
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ));
        }
    }

    #[test]
    fn fixed_sm120_tma_fma_a0_gate_is_exact_and_cohort_bound() {
        let a0 = FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        };
        assert!(fixed_sm120_tma_fma_a0_auto_eligible(
            operands(false),
            a0,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        assert!(!fixed_sm120_tma_fma_a0_auto_eligible(
            operands(true),
            a0,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        assert!(!fixed_sm120_tma_fma_a0_auto_eligible(
            operands(false),
            FixedShape { m: 4620, ..a0 },
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
    }

    #[test]
    fn fixed_sm120_tma_fma_a1_auto_prefers_t256_and_retains_control_fallback() {
        for (eligible, t256_loaded, expected) in [
            (
                true,
                Some(true),
                Some(FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256),
            ),
            (
                true,
                Some(false),
                Some(FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64),
            ),
            (true, None, None),
            (false, Some(true), None),
            (false, Some(false), None),
            (false, None, None),
        ] {
            assert_eq!(
                fixed_sm120_tma_fma_a1_auto_tile(eligible, t256_loaded),
                expected,
                "eligibility={eligible}, loaded={t256_loaded:?}"
            );
        }
    }

    #[test]
    fn fixed_sm120_tma_fma_a1_postbias_gate_is_exact_and_cohort_bound() {
        let a1 = FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        };
        assert!(fixed_sm120_tma_fma_a1_auto_eligible(
            operands(true),
            a1,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        assert!(!fixed_sm120_tma_fma_a1_auto_eligible(
            operands(false),
            a1,
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
        assert!(!fixed_sm120_tma_fma_a1_auto_eligible(
            operands(true),
            FixedShape { n: 1929, ..a1 },
            DEVICE,
            (13, 2),
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        ));
    }

    #[test]
    fn fixed_sm120_copyplan_t256_force_gate_keeps_hot_rows_and_bias_contract() {
        for (m, k, n) in [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ] {
            for bias in [false, true] {
                let eligible = |ops, shape| {
                    fixed_sm120_tma_fma_force_eligible(
                        ops,
                        shape,
                        DEVICE,
                        (13, 2),
                        true,
                        F32TriadPolicy::ExactScalarFmaV1,
                        FixedTile::F32Sm120N64CopyPlanT256,
                    )
                };
                let shape = FixedShape { m, k, n };
                assert!(eligible(operands(bias), shape));
                assert!(!eligible(operands(bias), FixedShape { m: m - 1, ..shape }));
                assert!(!eligible(
                    FixedFwdOperands {
                        bias_ptr: Some(0),
                        ..operands(bias)
                    },
                    shape
                ));
                assert!(!eligible(
                    FixedFwdOperands {
                        x: TypedPtr {
                            ptr: 0x2004,
                            dtype: WeightDtype::F32
                        },
                        ..operands(bias)
                    },
                    shape
                ));
            }
        }
    }

    #[test]
    fn fixed_sm120_nobias_force_gate_accepts_only_qualified_unbiased_hot_rows() {
        let tile = FixedTile::F32Sm120TmaFmaFixedNoBiasM128N64T256;
        let eligible = |ops, shape, device, nvrtc, known, policy| {
            fixed_sm120_tma_fma_force_eligible(ops, shape, device, nvrtc, known, policy, tile)
        };
        for (m, k, n) in [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ] {
            let shape = FixedShape { m, k, n };
            assert!(eligible(
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1
            ));
            for bias_ptr in [Some(0), Some(0x4004)] {
                assert!(!eligible(
                    FixedFwdOperands {
                        bias_ptr,
                        ..operands(false)
                    },
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
            }
        }
        let shape = FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        };
        for (ops, shape, device, nvrtc, known, policy) in [
            (
                operands(false),
                FixedShape { m: 2047, ..shape },
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                operands(false),
                shape,
                FixedTileDevice {
                    compute_capability: (12, 1),
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                operands(false),
                shape,
                FixedTileDevice {
                    multiprocessors: 169,
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                operands(false),
                shape,
                DEVICE,
                (13, 1),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                false,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::AllowDeterministicTf32V1,
            ),
        ] {
            assert!(!eligible(ops, shape, device, nvrtc, known, policy));
        }
        for bad in [
            TypedPtr {
                ptr: 0,
                dtype: WeightDtype::F32,
            },
            TypedPtr {
                ptr: 0x1004,
                dtype: WeightDtype::F32,
            },
            TypedPtr {
                ptr: 0x1000,
                dtype: WeightDtype::Bf16,
            },
        ] {
            for ops in [
                FixedFwdOperands {
                    c: bad,
                    ..operands(false)
                },
                FixedFwdOperands {
                    x: bad,
                    ..operands(false)
                },
                FixedFwdOperands {
                    w: bad,
                    ..operands(false)
                },
            ] {
                assert!(!eligible(
                    ops,
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
            }
        }
    }

    #[test]
    fn fixed_sm120_tma_fma_force_gate_admits_only_unbiased_hot_rows() {
        for (m, k, n) in [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ] {
            let shape = FixedShape { m, k, n };
            assert!(fixed_sm120_tma_fma_force_eligible(
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
                FixedTile::F32Sm120TmaFmaM128N64,
            ));
            assert!(!fixed_sm120_tma_fma_force_eligible(
                operands(true),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
                FixedTile::F32Sm120TmaFmaM128N64,
            ));
        }
        for shape in [
            FixedShape {
                m: 2047,
                k: 768,
                n: 2304,
            },
            FixedShape {
                m: 2049,
                k: 768,
                n: 2304,
            },
            FixedShape {
                m: 2048,
                k: 767,
                n: 2304,
            },
            FixedShape {
                m: 2048,
                k: 768,
                n: 2303,
            },
        ] {
            assert!(!fixed_sm120_tma_fma_force_eligible(
                operands(false),
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
                FixedTile::F32Sm120TmaFmaM128N64,
            ));
        }
    }

    #[test]
    fn fixed_sm120_tma_fma_force_gate_separates_bias_arithmetic_variants() {
        let shape = FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        };
        let eligible = |ops, tile| {
            fixed_sm120_tma_fma_force_eligible(
                ops,
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
                tile,
            )
        };
        assert!(eligible(operands(false), FixedTile::F32Sm120TmaFmaM64N128));
        assert!(!eligible(operands(true), FixedTile::F32Sm120TmaFmaM64N128));
        assert!(eligible(
            operands(true),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
        ));
        assert!(!eligible(
            operands(false),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM64N128
        ));
        assert!(eligible(
            operands(true),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        ));
        assert!(!eligible(
            operands(false),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        ));
        assert!(eligible(
            operands(true),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        ));
        assert!(!eligible(
            operands(false),
            FixedTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        ));
    }

    #[test]
    fn fixed_sm120_tma_fma_force_declines_identity_neighbors_and_operand_drift() {
        let shape = FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        };
        let ops = operands(false);
        let accept = |ops, shape, device, nvrtc, known, policy| {
            fixed_sm120_tma_fma_force_eligible(
                ops,
                shape,
                device,
                nvrtc,
                known,
                policy,
                FixedTile::F32Sm120TmaFmaM128N64,
            )
        };
        for (device, nvrtc, known, policy) in [
            (
                FixedTileDevice {
                    compute_capability: (12, 1),
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                FixedTileDevice {
                    compute_capability: (8, 9),
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                FixedTileDevice {
                    multiprocessors: 169,
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (
                FixedTileDevice {
                    multiprocessors: 171,
                    ..DEVICE
                },
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ),
            (DEVICE, (12, 8), true, F32TriadPolicy::ExactScalarFmaV1),
            (DEVICE, (13, 3), true, F32TriadPolicy::ExactScalarFmaV1),
            (DEVICE, (13, 2), false, F32TriadPolicy::ExactScalarFmaV1),
            (
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::AllowDeterministicTf32V1,
            ),
        ] {
            assert!(!accept(ops, shape, device, nvrtc, known, policy));
        }
        for bad in [
            FixedShape { m: 4620, ..shape },
            FixedShape { m: 4622, ..shape },
            FixedShape { k: 767, ..shape },
            FixedShape { k: 769, ..shape },
            FixedShape { n: 2303, ..shape },
            FixedShape { n: 2305, ..shape },
            FixedShape { m: 1, ..shape },
            FixedShape { k: 0, ..shape },
        ] {
            assert!(!accept(
                ops,
                bad,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1
            ));
        }
        for index in 0..3 {
            for ptr in [0, 0x1001, 0x1004, 0x1008, 0x100c] {
                let mut bad = ops;
                [&mut bad.c, &mut bad.x, &mut bad.w][index].ptr = ptr;
                assert!(!accept(
                    bad,
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
            }
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                let mut bad = ops;
                [&mut bad.c, &mut bad.x, &mut bad.w][index].dtype = dtype;
                assert!(!accept(
                    bad,
                    shape,
                    DEVICE,
                    (13, 2),
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
            }
        }
        for bias in [
            Some(0),
            Some(0x4000),
            Some(0x4001),
            Some(0x4002),
            Some(0x4003),
        ] {
            assert!(!accept(
                FixedFwdOperands {
                    bias_ptr: bias,
                    ..ops
                },
                shape,
                DEVICE,
                (13, 2),
                true,
                F32TriadPolicy::ExactScalarFmaV1
            ));
        }
    }

    fn eligible(
        operands: FixedFwdOperands,
        shape: FixedShape,
        device: FixedTileDevice,
        nvrtc: (i32, i32),
        known: bool,
        loaded: bool,
        policy: F32TriadPolicy,
    ) -> bool {
        fixed_sm120_exact_n64_auto_eligible(operands, shape, device, nvrtc, known, loaded, policy)
    }

    #[test]
    fn fixed_sm120_exact_n64_auto_admits_the_six_qualified_rows() {
        for (m, k, n) in [(2048, 2304, 768), (4621, 768, 2304), (2048, 768, 2304)] {
            for bias in [false, true] {
                assert!(fixed_sm120_exact_n64_auto_eligible(
                    operands(bias),
                    FixedShape { m, k, n },
                    DEVICE,
                    (13, 2),
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1,
                ));
            }
        }
    }

    #[test]
    fn fixed_sm120_exact_n64_auto_declines_unqualified_identity_or_policy() {
        let shape = FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        };
        let ops = operands(true);
        for cc in [(8, 9), (9, 0), (10, 0), (10, 3), (12, 1)] {
            assert!(!eligible(
                ops,
                shape,
                FixedTileDevice {
                    compute_capability: cc,
                    ..DEVICE
                },
                (13, 2),
                true,
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ));
        }
        for multiprocessors in [0, 1, 142, 169, 171] {
            assert!(!eligible(
                ops,
                shape,
                FixedTileDevice {
                    multiprocessors,
                    ..DEVICE
                },
                (13, 2),
                true,
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ));
        }
        for nvrtc in [(0, 0), (12, 8), (13, 0), (13, 1), (13, 3), (14, 0)] {
            assert!(!eligible(
                ops,
                shape,
                DEVICE,
                nvrtc,
                true,
                true,
                F32TriadPolicy::ExactScalarFmaV1,
            ));
        }
        for (known, loaded) in [(false, true), (true, false), (false, false)] {
            assert!(!eligible(
                ops,
                shape,
                DEVICE,
                (13, 2),
                known,
                loaded,
                F32TriadPolicy::ExactScalarFmaV1,
            ));
        }
        assert!(!eligible(
            ops,
            shape,
            DEVICE,
            (13, 2),
            true,
            true,
            F32TriadPolicy::AllowDeterministicTf32V1,
        ));
    }

    #[test]
    fn fixed_sm120_exact_n64_auto_declines_unmeasured_shape_alignment_and_dtype() {
        for (m, k, n) in [(2048, 2304, 768), (4621, 768, 2304), (2048, 768, 2304)] {
            for bias in [false, true] {
                let shape = FixedShape { m, k, n };
                let good = operands(bias);
                for adjacent in [
                    FixedShape { m: m - 1, k, n },
                    FixedShape { m: m + 1, k, n },
                    FixedShape { m, k: k - 1, n },
                    FixedShape { m, k: k + 1, n },
                    FixedShape { m, k, n: n - 1 },
                    FixedShape { m, k, n: n + 1 },
                    FixedShape { m: 0, k, n },
                ] {
                    assert!(!eligible(
                        good,
                        adjacent,
                        DEVICE,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1,
                    ));
                }
                for pointer in [0, 0x1001, 0x1004, 0x1008, 0x100c] {
                    for bad in [
                        FixedFwdOperands {
                            c: TypedPtr {
                                ptr: pointer,
                                ..good.c
                            },
                            ..good
                        },
                        FixedFwdOperands {
                            x: TypedPtr {
                                ptr: pointer,
                                ..good.x
                            },
                            ..good
                        },
                        FixedFwdOperands {
                            w: TypedPtr {
                                ptr: pointer,
                                ..good.w
                            },
                            ..good
                        },
                    ] {
                        assert!(!eligible(
                            bad,
                            shape,
                            DEVICE,
                            (13, 2),
                            true,
                            true,
                            F32TriadPolicy::ExactScalarFmaV1,
                        ));
                    }
                }
                for bias_ptr in [Some(0), Some(0x4001), Some(0x4002), Some(0x4003)] {
                    assert!(!eligible(
                        FixedFwdOperands { bias_ptr, ..good },
                        shape,
                        DEVICE,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1,
                    ));
                }
                for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                    for bad in [
                        FixedFwdOperands {
                            c: TypedPtr { dtype, ..good.c },
                            ..good
                        },
                        FixedFwdOperands {
                            x: TypedPtr { dtype, ..good.x },
                            ..good
                        },
                        FixedFwdOperands {
                            w: TypedPtr { dtype, ..good.w },
                            ..good
                        },
                    ] {
                        assert!(!eligible(
                            bad,
                            shape,
                            DEVICE,
                            (13, 2),
                            true,
                            true,
                            F32TriadPolicy::ExactScalarFmaV1,
                        ));
                    }
                }
            }
        }
    }
}

// Actual production NVRTC12.8/13.0/13.2 / 142-SM Ada paired qualification,
// 101 windows in both orders and both eager/graph paths: exactly A/B/D/E,
// each bias row. This is a finite version-scoped measured route, not a claim
// for every CUDA 12/13 stack or any future toolkit.
// A/D confirmation: fixed-exact-promotions-ada-20260906-318b3fbd09c8,
// SHA256 9224161eec3a578f472c07810647952c56604a980d699876b187887a929c575a.
// CUDA12.8 screen/confirm: d43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf /
// eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96.
// CUDA13.0 screen/confirm: cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682 /
// 0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711.
// This dispatch-only promotion reuses the unchanged loaded Fixed artifact.
fn fixed_sm89_exact_n64_auto_eligible(
    operands: FixedFwdOperands,
    shape: FixedShape,
    device: FixedTileDevice,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    loaded: bool,
    policy: super::context::F32TriadPolicy,
) -> bool {
    loaded
        && nvrtc_library_known
        && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
        && device.compute_capability == (8, 9)
        && device.multiprocessors == 142
        && policy == super::context::F32TriadPolicy::ExactScalarFmaV1
        && [operands.c, operands.x, operands.w]
            .into_iter()
            .all(|operand| {
                operand.dtype == WeightDtype::F32
                    && operand.ptr != 0
                    && operand.ptr.is_multiple_of(16)
            })
        && operands
            .bias_ptr
            .is_none_or(|ptr| ptr != 0 && ptr.is_multiple_of(4))
        && matches!(
            (shape.m, shape.k, shape.n, operands.bias_ptr.is_some()),
            (4621, 384, 1928, false)
                | (4621, 384, 1928, true)
                | (2048, 768, 2304, false)
                | (2048, 768, 2304, true)
                | (2048, 2304, 768, false)
                | (2048, 2304, 768, true)
                | (4621, 768, 2304, false)
                | (4621, 768, 2304, true)
        )
}

#[cfg(test)]
mod sm89_exact_n64_auto_tests {
    use super::super::context::F32TriadPolicy;
    use super::*;

    const DEVICE: FixedTileDevice = FixedTileDevice {
        multiprocessors: 142,
        compute_capability: (8, 9),
    };
    const QUALIFIED_NVRTC: [(i32, i32); 3] = [(12, 8), (13, 0), (13, 2)];
    // Literal admitted rows from production-paired-101-v1.jsonl, SHA256
    // 91493d7994999bfa48473803a1e29a7cc516a98b5264478c264457c3ba34db86.
    // B/no-bias wins against own AUTO/Legacy, not against PEDANTIC.
    // A/D extend that baseline using the independent production 101-window
    // confirmation referenced immediately above the AUTO predicate.
    const ROWS: [(usize, usize, usize, bool); 8] = [
        (4621, 384, 1928, false),
        (4621, 384, 1928, true),
        (2048, 768, 2304, false),
        (2048, 768, 2304, true),
        (2048, 2304, 768, false),
        (2048, 2304, 768, true),
        (4621, 768, 2304, false),
        (4621, 768, 2304, true),
    ];

    fn operands(bias: bool) -> FixedFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        FixedFwdOperands {
            c: typed(0x1000),
            x: typed(0x2000),
            w: typed(0x3000),
            // A real four-byte-aligned bias need not be sixteen-byte aligned.
            bias_ptr: bias.then_some(0x4004),
        }
    }

    fn eligible_for(nvrtc: (i32, i32), operands: FixedFwdOperands, shape: FixedShape) -> bool {
        fixed_sm89_exact_n64_auto_eligible(
            operands,
            shape,
            DEVICE,
            nvrtc,
            true,
            true,
            F32TriadPolicy::ExactScalarFmaV1,
        )
    }

    fn eligible(operands: FixedFwdOperands, shape: FixedShape) -> bool {
        eligible_for((13, 2), operands, shape)
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_routes_all_twenty_four_qualified_toolkit_literals() {
        for nvrtc in QUALIFIED_NVRTC {
            for (m, k, n, bias) in ROWS {
                assert!(
                    eligible_for(nvrtc, operands(bias), FixedShape { m, k, n }),
                    "qualified CopyPlan route missing: NVRTC={nvrtc:?} M={m} K={k} N={n} bias={bias}"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_routes_a_and_d_copyplan_winners() {
        for (m, k, n) in [(4621, 384, 1928), (2048, 768, 2304)] {
            for bias in [false, true] {
                assert!(
                    eligible(operands(bias), FixedShape { m, k, n }),
                    "qualified copy-plan must be reachable from AUTO: M={m} K={k} N={n} bias={bias}"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_admits_only_eight_measured_positive_rows() {
        let observed = ROWS.map(|(m, k, n, bias)| eligible(operands(bias), FixedShape { m, k, n }));
        assert_eq!(observed, [true; 8], "measured A0/A1/B0/B1/D0/D1/E0/E1 rows");
        for (m, k, n, bias) in ROWS {
            for offset in [0, 4, 8, 12] {
                let good = FixedFwdOperands {
                    bias_ptr: bias.then_some(0x4000 + offset),
                    ..operands(bias)
                };
                assert!(
                    eligible(good, FixedShape { m, k, n }),
                    "every valid F32 bias alignment"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_declines_adjacent_unmeasured_and_thin_shapes() {
        for nvrtc in QUALIFIED_NVRTC {
            for (m, k, n, bias) in ROWS {
                for (m, k, n) in [
                    (m - 1, k, n),
                    (m + 1, k, n),
                    (m, k - 1, n),
                    (m, k + 1, n),
                    (m, k, n - 1),
                    (m, k, n + 1),
                    (0, k, n),
                    (1, k, n),
                    (17, k, n),
                    (63, k, n),
                    (64, k, n),
                    (65, k, n),
                    (127, k, n),
                    (128, k, n),
                    (129, k, n),
                    (m, 0, n),
                    (m, k, 0),
                    (4621, 1928, 384),
                ] {
                    assert!(
                        !eligible_for(nvrtc, operands(bias), FixedShape { m, k, n }),
                        "unmeasured NVRTC={nvrtc:?} M={m} K={k} N={n} bias={bias}"
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_declines_unknown_device_toolchain_holder_and_policy() {
        for (m, k, n, bias) in ROWS {
            let ops = operands(bias);
            let shape = FixedShape { m, k, n };
            let check = |device, nvrtc, known, loaded, policy| {
                fixed_sm89_exact_n64_auto_eligible(ops, shape, device, nvrtc, known, loaded, policy)
            };
            for nvrtc in QUALIFIED_NVRTC {
                for cc in [
                    (8, 0),
                    (8, 6),
                    (8, 7),
                    (9, 0),
                    (10, 0),
                    (10, 3),
                    (11, 0),
                    (12, 0),
                    (12, 1),
                ] {
                    assert!(!check(
                        FixedTileDevice {
                            compute_capability: cc,
                            ..DEVICE
                        },
                        nvrtc,
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ));
                }
                for multiprocessors in [0, 1, 141, 143, 170] {
                    assert!(!check(
                        FixedTileDevice {
                            multiprocessors,
                            ..DEVICE
                        },
                        nvrtc,
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ));
                }
                for (known, loaded) in [(false, true), (true, false), (false, false)] {
                    assert!(!check(
                        DEVICE,
                        nvrtc,
                        known,
                        loaded,
                        F32TriadPolicy::ExactScalarFmaV1
                    ));
                }
                assert!(!check(
                    DEVICE,
                    nvrtc,
                    true,
                    true,
                    F32TriadPolicy::AllowDeterministicTf32V1
                ));
            }
            for version in [(0, 0), (12, 7), (12, 9), (13, 1), (13, 3), (14, 0)] {
                assert!(!check(
                    DEVICE,
                    version,
                    true,
                    true,
                    F32TriadPolicy::ExactScalarFmaV1
                ));
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_declines_invalid_alignment_null_bias_and_dtype() {
        for nvrtc in QUALIFIED_NVRTC {
            for (m, k, n, bias) in ROWS {
                let good = operands(bias);
                let shape = FixedShape { m, k, n };
                for pointer in [0, 0x1001, 0x1002, 0x1003, 0x1004, 0x1008, 0x100c] {
                    for bad in [
                        FixedFwdOperands {
                            c: TypedPtr {
                                ptr: pointer,
                                ..good.c
                            },
                            ..good
                        },
                        FixedFwdOperands {
                            x: TypedPtr {
                                ptr: pointer,
                                ..good.x
                            },
                            ..good
                        },
                        FixedFwdOperands {
                            w: TypedPtr {
                                ptr: pointer,
                                ..good.w
                            },
                            ..good
                        },
                    ] {
                        assert!(
                            !eligible_for(nvrtc, bad, shape),
                            "invalid or non-A16/B16/C16 operand"
                        );
                    }
                }
                for pointer in [0, 0x4001, 0x4002, 0x4003] {
                    assert!(!eligible_for(
                        nvrtc,
                        FixedFwdOperands {
                            bias_ptr: Some(pointer),
                            ..good
                        },
                        shape
                    ));
                }
                for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                    for bad in [
                        FixedFwdOperands {
                            c: TypedPtr { dtype, ..good.c },
                            ..good
                        },
                        FixedFwdOperands {
                            x: TypedPtr { dtype, ..good.x },
                            ..good
                        },
                        FixedFwdOperands {
                            w: TypedPtr { dtype, ..good.w },
                            ..good
                        },
                        FixedFwdOperands {
                            c: TypedPtr { dtype, ..good.c },
                            x: TypedPtr { dtype, ..good.x },
                            w: TypedPtr { dtype, ..good.w },
                            ..good
                        },
                    ] {
                        assert!(
                            !eligible_for(nvrtc, bad, shape),
                            "only homogeneous exact F32 is admitted"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_auto_keeps_old_cc12_exact_selector_cells() {
        for cc in [(12, 0), (12, 1)] {
            let device = FixedTileDevice {
                compute_capability: cc,
                multiprocessors: 170,
            };
            for (m, k, n, old_cc120) in [
                (4621, 384, 1928, FixedTile::F32N128S2),
                (4621, 768, 2304, FixedTile::F32N128S2),
                (4621, 1928, 384, FixedTile::Legacy),
                (2048, 768, 2304, FixedTile::Legacy),
                (2048, 2304, 768, FixedTile::Legacy),
            ] {
                for bias in [false, true] {
                    assert!(!fixed_sm89_exact_n64_auto_eligible(
                        operands(bias),
                        FixedShape { m, k, n },
                        device,
                        (13, 2),
                        true,
                        true,
                        F32TriadPolicy::ExactScalarFmaV1
                    ));
                }
                let expected = if cc == (12, 0) {
                    old_cc120
                } else {
                    FixedTile::Legacy
                };
                assert_eq!(fixed_pick_f32_exact(m, k, n, device), expected);
            }
        }
    }
}

#[cfg(test)]
mod sm89_exact_n64_request_tests {
    use super::*;

    fn operands() -> FixedFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        FixedFwdOperands {
            c: typed(0x1000),
            x: typed(0x2000),
            w: typed(0x3000),
            bias_ptr: Some(0x4000),
        }
    }
    fn shape() -> FixedShape {
        FixedShape {
            m: 65,
            k: 96,
            n: 136,
        }
    }
    fn prepare(
        ops: FixedFwdOperands,
        dims: FixedShape,
    ) -> Result<Option<(FixedArgs, u32)>, String> {
        prepare_sm89_exact_n64_launch(ops, dims, (8, 9))
    }

    #[test]
    fn fixed_sm89_exact_n64_operand_and_arch_guards_are_fail_closed() {
        assert!(prepare(operands(), shape()).unwrap().is_some());
        for cc in [(8, 0), (8, 6), (9, 0), (12, 0), (12, 1)] {
            assert!(prepare_sm89_exact_n64_launch(operands(), shape(), cc).is_err());
        }
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let good = operands();
            for bad in [
                FixedFwdOperands {
                    x: TypedPtr { dtype, ..good.x },
                    ..good
                },
                FixedFwdOperands {
                    w: TypedPtr { dtype, ..good.w },
                    ..good
                },
                FixedFwdOperands {
                    c: TypedPtr { dtype, ..good.c },
                    ..good
                },
            ] {
                assert!(prepare(bad, shape()).is_err(), "mixed dtype admitted");
            }
        }
        let good = operands();
        for ptr in [0, 1, 2, 3, 0x2001] {
            for bad in [
                FixedFwdOperands {
                    x: TypedPtr { ptr, ..good.x },
                    ..good
                },
                FixedFwdOperands {
                    w: TypedPtr { ptr, ..good.w },
                    ..good
                },
                FixedFwdOperands {
                    c: TypedPtr { ptr, ..good.c },
                    ..good
                },
            ] {
                assert!(
                    prepare(bad, shape()).is_err(),
                    "null/unaligned pointer admitted"
                );
            }
        }
        for ptr in [1, 2, 3, 0x4001] {
            assert!(
                prepare(
                    FixedFwdOperands {
                        bias_ptr: Some(ptr),
                        ..good
                    },
                    shape()
                )
                .is_err()
            );
        }
        // Four-byte views must stay force-admissible; vector alignment only
        // selects an internal staging/store path and is not a force restriction.
        for role in 0..3 {
            let mut shifted = good;
            match role {
                0 => shifted.x.ptr += 4,
                1 => shifted.w.ptr += 4,
                _ => shifted.c.ptr += 4,
            }
            assert!(prepare(shifted, shape()).unwrap().is_some());
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_signed_padding_and_grid_boundaries() {
        let good = operands();
        let (_, grid) = prepare(good, shape()).unwrap().unwrap();
        assert_eq!(grid, 6, "65x136 needs 2x3 N64 CTAs");
        let max = i32::MAX as usize;
        for dims in [
            FixedShape {
                m: max - 63,
                k: 0,
                n: 1,
            },
            FixedShape {
                m: 1,
                k: max - 31,
                n: 1,
            },
            FixedShape {
                m: 1,
                k: 0,
                n: max - 63,
            },
        ] {
            assert!(
                prepare(good, dims).unwrap().is_some(),
                "exact padded boundary {dims:?}"
            );
        }
        for dims in [
            FixedShape {
                m: max - 62,
                k: 0,
                n: 1,
            },
            FixedShape {
                m: 1,
                k: max - 30,
                n: 1,
            },
            FixedShape {
                m: 1,
                k: 0,
                n: max - 62,
            },
            FixedShape {
                m: usize::MAX,
                ..shape()
            },
            FixedShape {
                k: usize::MAX,
                ..shape()
            },
            FixedShape {
                n: usize::MAX,
                ..shape()
            },
        ] {
            assert!(
                prepare(good, dims).is_err(),
                "signed/padded overflow {dims:?}"
            );
        }
        let accepted = FixedShape {
            m: 2_097_152,
            k: 0,
            n: 4_194_240,
        };
        assert_eq!(prepare(good, accepted).unwrap().unwrap().1, 2_147_450_880);
        let rejected = FixedShape {
            n: 4_194_304,
            ..accepted
        };
        assert!(
            prepare(good, rejected).is_err(),
            "2^31 CTAs overflow signed grouped indexing"
        );
    }

    #[test]
    fn fixed_sm89_exact_n64_operand_byte_endpoints_cannot_wrap() {
        let good = operands();
        assert!(prepare(good, shape()).unwrap().is_some());
        for bad in [
            FixedFwdOperands {
                x: TypedPtr {
                    ptr: u64::MAX - 3,
                    ..good.x
                },
                ..good
            },
            FixedFwdOperands {
                w: TypedPtr {
                    ptr: u64::MAX - 3,
                    ..good.w
                },
                ..good
            },
            FixedFwdOperands {
                c: TypedPtr {
                    ptr: u64::MAX - 3,
                    ..good.c
                },
                ..good
            },
            FixedFwdOperands {
                bias_ptr: Some(u64::MAX - 3),
                ..good
            },
        ] {
            assert!(prepare(bad, shape()).is_err(), "address endpoint wrapped");
        }
        let one = FixedShape { m: 1, k: 1, n: 1 };
        let last = TypedPtr {
            ptr: u64::MAX - 7,
            dtype: WeightDtype::F32,
        };
        assert!(
            prepare(
                FixedFwdOperands {
                    c: last,
                    x: last,
                    w: last,
                    bias_ptr: Some(last.ptr)
                },
                one
            )
            .unwrap()
            .is_some()
        );
        let max = i32::MAX as usize;
        let far = TypedPtr {
            ptr: 1 << 40,
            dtype: WeightDtype::F32,
        };
        assert!(
            prepare(
                FixedFwdOperands { x: far, ..good },
                FixedShape {
                    m: max - 63,
                    k: max - 31,
                    n: 1
                }
            )
            .is_err(),
            "A byte extent wrapped"
        );
        assert!(
            prepare(
                FixedFwdOperands { w: far, ..good },
                FixedShape {
                    m: 1,
                    k: max - 31,
                    n: max - 63
                }
            )
            .is_err(),
            "B byte extent wrapped"
        );
    }

    #[test]
    fn fixed_sm89_exact_n64_empty_and_k0_requests_do_not_require_inputs() {
        let good = operands();
        assert!(prepare(good, shape()).unwrap().is_some());
        let null = TypedPtr {
            ptr: 0,
            dtype: WeightDtype::F32,
        };
        let empty = FixedFwdOperands {
            c: null,
            x: null,
            w: null,
            bias_ptr: None,
        };
        for dims in [
            FixedShape {
                m: 0,
                k: usize::MAX,
                n: usize::MAX,
            },
            FixedShape {
                m: usize::MAX,
                k: usize::MAX,
                n: 0,
            },
        ] {
            assert!(prepare(empty, dims).unwrap().is_none());
        }
        for bias_ptr in [None, Some(0), Some(0x4000)] {
            let no_inputs = FixedFwdOperands {
                x: null,
                w: null,
                bias_ptr,
                ..good
            };
            let (args, grid) = prepare(no_inputs, FixedShape { k: 0, ..shape() })
                .unwrap()
                .unwrap();
            assert_eq!((args.a, args.b, args.k, grid), (0, 0, 0, 6));
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_forward_bundle_has_exact_m_n_k_and_stride_words() {
        let (args, _) = prepare(operands(), shape()).unwrap().unwrap();
        let params = FixedSm89ExactF32Params::forward(&args);
        let words = unsafe {
            (&params as *const FixedSm89ExactF32Params)
                .cast::<[u32; 8]>()
                .read()
        };
        assert_eq!(words, [0x3f80_0000, 0, 65, 136, 96, 96, 136, 136]);
        assert_eq!(std::mem::size_of_val(&params), 32);
        assert_eq!(std::mem::align_of::<FixedSm89ExactF32Params>(), 4);
    }
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
        FixedHalfMapCache, FixedHalfMapKey, FixedPostBiasMapCache, FixedPostBiasMapKey,
        FixedSm120HalfExactDevice, FixedSm120HalfExactRequest, FixedSm120HalfTile, FixedTensorMap,
        FixedTf32MapCache, FixedTf32MapKey, FixedTile, FixedTileDevice, fixed_adjust_arch_tile,
        fixed_pick_f32_exact, fixed_pick_f32out_tile, fixed_pick_sm120_f32out,
        fixed_pick_sm120_half, fixed_pick_sm120_half_exact, fixed_pick_tf32, fixed_pick_tile,
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
    fn sm120_postbias_bk16_maps_are_isolated_from_tf32_bk32_maps() {
        let tf32 = FixedTf32MapKey {
            a: 0x1000,
            b: 0x2000,
            m: 4621,
            k: 384,
            n: 1928,
            tile_rows: 128,
        };
        let postbias = FixedPostBiasMapKey {
            a: tf32.a,
            b: tf32.b,
            m: tf32.m,
            k: tf32.k,
            n: tf32.n,
            tile_m: 128,
            tile_n: 64,
        };
        let tf32_cache = FixedTf32MapCache::default();
        let postbias_cache = FixedPostBiasMapCache::default();
        assert!(tf32_cache.entries.is_empty());
        assert!(postbias_cache.entries.is_empty());
        assert_eq!([32, tf32.tile_rows], [32, 128]);
        assert_eq!([16, postbias.tile_m], [16, 128]);
        assert_eq!([postbias.tile_n, 16], [64, 16]);
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
            (5, 45, 8),
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
            fixed_pick_tf32(
                4621,
                768,
                2304,
                170,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M128S2
        );
        for has_bias in [false, true] {
            assert_eq!(
                fixed_pick_tf32(
                    2048,
                    768,
                    2304,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    true,
                    true
                ),
                FixedTile::Tf32Sm120M64S2PairStore,
                "both D0 and D1 have independent confirmation"
            );
            assert_eq!(
                fixed_pick_tf32(
                    2048,
                    768,
                    2304,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    false,
                    true
                ),
                FixedTile::Tf32Sm120M64S2ProducerWarp,
                "unaligned C must retain the previously qualified schedule"
            );
            for (m, k, n, sms, cc, nvrtc, loaded) in [
                (2049, 768, 2304, 170, (12, 0), (13, 2), true),
                (2048, 769, 2304, 170, (12, 0), (13, 2), true),
                (2048, 768, 2305, 170, (12, 0), (13, 2), true),
                (2048, 768, 2304, 169, (12, 0), (13, 2), true),
                (2048, 768, 2304, 170, (12, 1), (13, 2), true),
                (2048, 768, 2304, 170, (12, 0), (12, 8), true),
                (2048, 768, 2304, 170, (12, 0), (13, 0), true),
                (2048, 768, 2304, 170, (12, 0), (13, 1), true),
                (2048, 768, 2304, 170, (12, 0), (13, 3), true),
                (2048, 768, 2304, 170, (12, 0), (13, 2), false),
            ] {
                assert_eq!(
                    fixed_pick_tf32(m, k, n, sms, cc, nvrtc, loaded, has_bias, true, true),
                    if loaded {
                        FixedTile::Tf32Sm120M64S2
                    } else {
                        FixedTile::Tf32M64S2
                    }
                );
            }
        }
        for nvrtc_version in [(12, 8), (13, 0), (13, 1), (13, 3)] {
            assert_eq!(
                fixed_pick_tf32(
                    4621,
                    768,
                    2304,
                    170,
                    (12, 0),
                    nvrtc_version,
                    true,
                    false,
                    true,
                    true
                ),
                FixedTile::Tf32Sm120M64S2,
                "unqualified compiler {nvrtc_version:?} must use the portable SM120 schedule"
            );
            assert_eq!(
                fixed_pick_tf32(
                    2048,
                    768,
                    2304,
                    170,
                    (12, 0),
                    nvrtc_version,
                    true,
                    false,
                    true,
                    true
                ),
                FixedTile::Tf32Sm120M64S2,
                "unqualified compiler {nvrtc_version:?} must not use the promoted pair-store schedule"
            );
        }
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                2304,
                169,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                2304,
                170,
                (12, 1),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4622,
                768,
                2304,
                170,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                769,
                2304,
                170,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                2305,
                170,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
    }

    #[test]
    fn sm120_tf32_selector_promotes_confirmed_a_bias_rows() {
        for has_bias in [false, true] {
            for (m, k, n, sms, cc, nvrtc, loaded, aligned, known) in [
                (4622, 384, 1928, 170, (12, 0), (13, 2), true, true, true),
                (4621, 388, 1928, 170, (12, 0), (13, 2), true, true, true),
                (4621, 384, 1932, 170, (12, 0), (13, 2), true, true, true),
                (4621, 384, 1928, 169, (12, 0), (13, 2), true, true, true),
                (4621, 384, 1928, 170, (12, 1), (13, 2), true, true, true),
                (4621, 384, 1928, 170, (12, 0), (12, 8), true, true, true),
                (4621, 384, 1928, 170, (12, 0), (13, 0), true, true, true),
                (4621, 384, 1928, 170, (12, 0), (13, 1), true, true, true),
                (4621, 384, 1928, 170, (12, 0), (13, 3), true, true, true),
                (4621, 384, 1928, 170, (12, 0), (13, 2), true, true, false),
                (4621, 384, 1928, 170, (12, 0), (13, 2), true, false, true),
                (4621, 384, 1928, 170, (12, 0), (13, 2), false, true, true),
            ] {
                assert_eq!(
                    fixed_pick_tf32(m, k, n, sms, cc, nvrtc, loaded, has_bias, aligned, known),
                    if loaded {
                        FixedTile::Tf32Sm120M64S2
                    } else {
                        FixedTile::Tf32M64S2
                    }
                );
            }
            // Unknown libraries do not change previously qualified B/D behavior.
            assert_eq!(
                fixed_pick_tf32(
                    4621,
                    768,
                    2304,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    true,
                    false
                ),
                FixedTile::Tf32Sm120M128S2
            );
            assert_eq!(
                fixed_pick_tf32(
                    2048,
                    768,
                    2304,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    true,
                    false
                ),
                FixedTile::Tf32Sm120M64S2PairStore
            );
            assert_eq!(
                fixed_pick_tf32(
                    4621,
                    384,
                    1928,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    true,
                    true
                ),
                FixedTile::Tf32Sm120M128S2
            );
            assert_eq!(
                fixed_pick_tf32(
                    4621,
                    384,
                    1928,
                    170,
                    (12, 0),
                    (13, 2),
                    true,
                    has_bias,
                    false,
                    true
                ),
                FixedTile::Tf32Sm120M64S2
            );
        }
    }

    #[test]
    fn sm89_tf32_selector_promotes_only_confirmed_c_bias_rows() {
        for has_bias in [false, true] {
            assert_eq!(
                fixed_pick_tf32(
                    16,
                    1928,
                    384,
                    142,
                    (8, 9),
                    (13, 2),
                    false,
                    has_bias,
                    true,
                    true,
                ),
                FixedTile::Tf32M16S4,
                "Ada thin route must remain ahead of the C promotion",
            );
            for output_aligned in [false, true] {
                assert_eq!(
                    fixed_pick_tf32(
                        4621,
                        1928,
                        384,
                        142,
                        (8, 9),
                        (13, 2),
                        false,
                        has_bias,
                        output_aligned,
                        true,
                    ),
                    FixedTile::Tf32M64S2,
                    "confirmed Ada C row bias={has_bias} output_aligned={output_aligned}",
                );
            }
            for (m, k, n, sms, cc, nvrtc, known) in [
                (4620, 1928, 384, 142, (8, 9), (13, 2), true),
                (4622, 1928, 384, 142, (8, 9), (13, 2), true),
                (4621, 1927, 384, 142, (8, 9), (13, 2), true),
                (4621, 1929, 384, 142, (8, 9), (13, 2), true),
                (4621, 1928, 383, 142, (8, 9), (13, 2), true),
                (4621, 1928, 385, 142, (8, 9), (13, 2), true),
                (4621, 1928, 384, 141, (8, 9), (13, 2), true),
                (4621, 1928, 384, 143, (8, 9), (13, 2), true),
                (4621, 1928, 384, 142, (8, 8), (13, 2), true),
                (4621, 1928, 384, 142, (9, 0), (13, 2), true),
                (4621, 1928, 384, 142, (8, 9), (12, 8), true),
                (4621, 1928, 384, 142, (8, 9), (13, 0), true),
                (4621, 1928, 384, 142, (8, 9), (13, 1), true),
                (4621, 1928, 384, 142, (8, 9), (13, 3), true),
                (4621, 1928, 384, 142, (8, 9), (13, 2), false),
            ] {
                assert_eq!(
                    fixed_pick_tf32(m, k, n, sms, cc, nvrtc, false, has_bias, true, known),
                    FixedTile::Tf32M128S2,
                    "unqualified Ada C boundary M={m} K={k} N={n} SMs={sms} CC={cc:?} NVRTC={nvrtc:?} known={known}",
                );
            }
        }
    }

    #[test]
    fn tf32_selector_keeps_portable_and_thin_fallbacks() {
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                1928,
                170,
                (12, 0),
                (13, 2),
                false,
                false,
                true,
                true
            ),
            FixedTile::Tf32M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                1928,
                170,
                (12, 0),
                (13, 2),
                true,
                false,
                true,
                true
            ),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_pick_tf32(1, 768, 1928, 170, (12, 0), (13, 2), true, false, true, true),
            FixedTile::Tf32M16S4
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                2304,
                142,
                (8, 9),
                (13, 2),
                false,
                false,
                true,
                true
            ),
            FixedTile::Tf32M64S2
        );
        assert_eq!(
            fixed_pick_tf32(
                4621,
                768,
                384,
                142,
                (8, 9),
                (13, 2),
                false,
                false,
                true,
                true
            ),
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
