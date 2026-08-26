use super::super::dtype::WeightDtype;
use super::super::kernels::MambaKernels as GpuKernels;
use super::contract::{GemmDims, checked_mul3, checked_tile_grid, checked_usize};

// ── Split-M TN partition heuristic (ported from SQV-RS blas_bi.rs) ──

/// Target CTA count factor for the split-M TN partition: aim to fill the
/// GPU with at least this many blocks when the base (K-tile × N-tile) grid
/// underfills it.
pub(super) const SPLITM_TN_TARGET_GRID_FACTOR: u32 = 284;
/// Scratch cap for split-M partials, in f32 elements. Must not exceed the
/// `splitk_scratch` allocation in kernels.rs.
pub(super) const SPLITM_TN_SCRATCH_CAP: usize = 1 << 23;
/// m_chunk alignment (BK of the TN tile).
pub(super) const SPLITM_TN_BK_ALIGN: u32 = 16;

/// Decide the split-M factor for the TN (dW) kernel on underfilled grids.
/// Returns `(m_chunk, f_final)` or `None` when the plain kernel is fine.
#[inline]
pub(super) fn splitm_tn_partition(
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Option<(usize, usize)> {
    // No n_in floor: the partial kernel predicates K_out < 128 exactly
    // like the plain kernel, and a small-K dW against a large batch
    // reduction underfills the grid without the split (K_out=24, N=768
    // ran six CTAs). The split changes the dW summation order versus
    // the plain kernel; run-to-run and per-shape determinism hold — the
    // partition is a pure function of (batch, n_in, n_out).
    if !(n_out >= 128 && batch >= 256) {
        return None;
    }
    let batch_u32 = u32::try_from(batch).ok()?;
    let k_tiles = u32::try_from(n_in).ok()?.div_ceil(128);
    let n_tiles = u32::try_from(n_out).ok()?.div_ceil(128);
    let base_blocks = k_tiles.checked_mul(n_tiles)?;
    if base_blocks == 0 || base_blocks >= SPLITM_TN_TARGET_GRID_FACTOR {
        return None;
    }
    let f_grid = SPLITM_TN_TARGET_GRID_FACTOR.div_ceil(base_blocks);
    let output_elements = n_in.checked_mul(n_out)?;
    let f_scratch_cap = u32::try_from(SPLITM_TN_SCRATCH_CAP / output_elements).ok()?;
    let f = f_grid.min(f_scratch_cap).max(1);
    let m_chunk_raw = batch_u32.div_ceil(f);
    let m_chunk = m_chunk_raw.checked_add(SPLITM_TN_BK_ALIGN - 1)? & !(SPLITM_TN_BK_ALIGN - 1);
    let f_final = batch_u32.div_ceil(m_chunk);
    let scratch_elements = usize::try_from(f_final)
        .ok()?
        .checked_mul(output_elements)?;
    if f_final < 2 || scratch_elements > SPLITM_TN_SCRATCH_CAP {
        return None;
    }
    Some((
        usize::try_from(m_chunk).ok()?,
        usize::try_from(f_final).ok()?,
    ))
}

/// Minimum N (output cols) before the dispatcher switches from Slim-N tiles
/// to Big-N tiles. Below this, Slim-N (BN=64) packs better; above it Big-N
/// (BN=128) wins on wave occupancy. Historic name (`SGEMM_CUSTOM_MIN`) is a
/// leftover from when the threshold gated a cuBLAS fallback — the fallback
/// is gone (zero-cuBLAS contract), the constant remains as a tile-pick
/// boundary only.
pub(super) const SGEMM_CUSTOM_MIN: usize = 128;

/// Boundary between Slim-N and Big tile variants (by output N dimension).
pub(super) const SGEMM_SLIM_MAX: usize = 512;

/// v6.5 Phase C-1.5av: separate Slim Split-K NT-via-T n_in cap for backward dx.
/// The forward Slim NN path uses N as output dim → SGEMM_SLIM_MAX=512 bounds
/// wave-fill correctness there. But NT-via-T backward dx reads n_in (input dim
/// of original forward), and the kernel itself tiles arbitrary n_in via N-axis
/// tiling — the 512 cap is conservative, not load-bearing. v6.5 multi-step
/// critic_in_proj has n_in = critic_in = d_model + action_dim + 2*emb = 641
/// at default config. Bumping to 768 lets this shape hit Slim Split-K NT-via-T
/// with F=4 K-tile partials (576 blocks vs plain Big NT 144 blocks).
/// Determinism preserved: F is shape-keyed (function of n_out, not batch).
pub(super) const SGEMM_SLIM_NT_NIN_MAX: usize = 768;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
pub(super) const SGEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements.
/// Must match `splitk_scratch` allocation in `kernels.rs` (1 << 23 = 8M f32 = 32 MB).
/// All Split-K dispatch gates (NN fwd, NT bwd_dx, Split-M TN bwd_dw) read this.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;

/// SM count for dispatch wave-fill heuristics. Calibrated for Ada RTX 6000 (142 SMs).
/// Over-shoot on smaller GPUs (A100=108) is correctness-safe — Split-K gates fire
/// slightly more aggressively. TODO: query `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
/// at init for true per-GPU tuning; for now a single source-of-truth constant.
pub(super) const NUM_SMS: u32 = 142;

/// Pick (kernel function, BN tile size) with M-aware wave-quantization fix.
/// Slim-N for narrow output, or for small M with wide N.
/// Later buckets extend this dispatcher with narrow / GEMV / small-K buckets.
pub(super) fn dispatch_slim_or_big<'k>(
    _kernels: &'k GpuKernels,
    m: usize,
    n_out: usize,
    func_slim: &'k cudarc::driver::CudaFunction,
    func_big: &'k cudarc::driver::CudaFunction,
) -> (&'k cudarc::driver::CudaFunction, u32) {
    let slim = n_out <= SGEMM_SLIM_MAX || (m < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    let func = if slim { func_slim } else { func_big };
    let bn: u32 = if slim { 64 } else { 128 };
    (func, bn)
}

// ---------------------------------------------------------------------------
// f32-cascade routing predicates. Each returns true iff the f32
// dispatcher would run the BIG kernel (BN=128, 2-stage 33 KB smem) for this
// shape — i.e. NO earlier bucket in the cascade claims it AND the final
// slim/big split picks Big. The typed dispatch uses these so a native typed
// Big kernel fires exactly where the f32 reference runs the same FMA chain;
// any drift between a predicate and the real cascade shows up as a bit
// mismatch in tests/gemm_bi_typed_parity.rs.
// ---------------------------------------------------------------------------

/// NN forward: mirrors `sgemm_bi_forward` (gemv, ultra-thin, narrow tiers,
/// split-K thin-M K-tail/main, split-K slim, gap-fill, then big/slim).
pub(super) fn nn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    let Ok(dims) = GemmDims::nn((batch, n_in, n_out), n_in) else {
        return false;
    };
    if n_out == 1 {
        return false; // gemv (or panic tail) — never Big
    }
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        return false; // ultra-thin
    }
    if (2..=127).contains(&n_out) {
        return false; // narrow tiers
    }
    let Ok(plain_slim_blocks) = checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64) else {
        return false;
    };
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-K thin-M K-tail
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32
            && checked_mul3(k_main / 32, batch, n_out, "NN route scratch")
                .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            return false;
        }
    }
    // split-K thin-M main
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && checked_mul3(n_in / 32, batch, n_out, "NN route scratch")
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        && underfill
    {
        return false;
    }
    // split-K slim
    if batch > 1024
        && (128..=SGEMM_SLIM_MAX).contains(&n_out)
        && n_in >= 64
        && n_in.is_multiple_of(32)
    {
        let f_final = dims.k_u32.div_ceil(64);
        if f_final >= 6
            && checked_mul3(
                checked_usize(f_final, "NN route chunks").unwrap_or(usize::MAX),
                batch,
                n_out,
                "NN route scratch",
            )
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            let base_blocks =
                checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64).unwrap_or(u32::MAX);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if batch < 128 {
        return false; // gap-fill territory
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// TN dW: mirrors `sgemm_bi_backward_dw` (gemv, narrow, split-M, big/slim
/// keyed on output rows = `n_in`).
pub(super) fn tn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    if GemmDims::tn((batch, n_in, n_out)).is_err() {
        return false;
    }
    if n_out == 1 || (2..=127).contains(&n_out) {
        return false; // gemv / narrow
    }
    if splitm_tn_partition(batch, n_in, n_out).is_some() {
        return false;
    }
    if !(n_in >= 1 && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (n_in < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// NT dX: mirrors `sgemm_bi_backward_dx` (narrow, col-gemv, gemv, split-N
/// K-tail/main, split-N slim, gap-fill, big/slim keyed on (`batch`, `n_in`)).
pub(super) fn nt_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    let Ok(dims) = GemmDims::nt((batch, n_in, n_out)) else {
        return false;
    };
    if (2..=127).contains(&n_out) {
        return false; // NT narrow (small reduction N)
    }
    if batch < 32 && n_out >= 128 {
        return false; // dx_col_gemv
    }
    if n_out == 1 {
        return false; // NT gemv
    }
    const SPLITK_NT_TRANSPOSE_CAP: usize = 1 << 22;
    let Ok(plain_slim_blocks) = checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64) else {
        return false;
    };
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-N K-tail
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32
            && k_main
                .checked_mul(n_out)
                .is_some_and(|elements| elements <= SPLITK_NT_TRANSPOSE_CAP)
            && checked_mul3(n_out / 32, batch, k_main, "NT route scratch")
                .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            return false;
        }
    }
    // split-N main
    let n_main = n_out - n_out % 32;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(4)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main >= 32
        && dims.kn <= SPLITK_NT_TRANSPOSE_CAP
        && checked_mul3(n_main / 32, batch, n_in, "NT route scratch")
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        && underfill
    {
        return false;
    }
    // split-N slim
    if batch > 1024
        && (128..=SGEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && dims.kn <= SPLITK_NT_TRANSPOSE_CAP
    {
        let f_final = dims.n_u32.div_ceil(64);
        if f_final >= 2
            && checked_mul3(
                checked_usize(f_final, "NT route chunks").unwrap_or(usize::MAX),
                batch,
                n_in,
                "NT route scratch",
            )
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            let base_blocks =
                checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64).unwrap_or(u32::MAX);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if (32..128).contains(&batch) {
        return false; // gap-fill NT
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_in >= 1) {
        return false;
    }
    let slim = n_in <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_in >= SGEMM_CUSTOM_MIN);
    !slim
}
/// Which tensor-core tile variant a TC entry point launched. Returned on
/// success so callers and tests can assert launch reality (0.4.0 lesson:
/// a kernel that silently never fires must be impossible to miss).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcTile {
    /// 128x128 CTA tile, 256 threads / 8 warps (`sgemm_bi_*_tc_*`).
    Tile128,
    /// 64x64 CTA tile, 128 threads / 4 warps (`sgemm_bi_*_tc64_*`).
    Tile64,
    /// 16x32 CTA tile, 128 threads / 4 warps, 4-stage cp.async
    /// (`sgemm_bi_nn_tc16_*`) - the decode rung of the ladder. NN
    /// forward only; picked by `tc_pick_tile_forward` for the small-M
    /// and narrow-N bands.
    Thin16,
}

/// Prefer the smaller tile while a 128x128 launch has too few independent
/// CTAs. Both tile families issue the same ascending MMA reduction for an
/// output element, so this changes occupancy without changing its bits.
pub const TC64_PREFER_MAX_TILES128: u32 =
    super::super::kernel_identity::LegacySm80Policy::current().tile128_prefer_min_tiles;

/// Choose between the square tiles once both output axes reach one Tile64.
pub(super) fn tc_pick_tile_large(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if rows >= policy.large_tile_min && cols >= policy.large_tile_min {
        let tiles128 = u32::try_from(rows)
            .ok()?
            .div_ceil(128)
            .checked_mul(u32::try_from(cols).ok()?.div_ceil(128))?;
        if tiles128 >= TC64_PREFER_MAX_TILES128 {
            return Some(TcTile::Tile128);
        }
        return Some(TcTile::Tile64);
    }
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return Some(TcTile::Tile64);
    }
    None
}

/// The forward thin tile covers the narrow rows or columns below Tile64.
/// Its per-element MMA order matches the square tiles, so crossing this
/// scheduling boundary preserves the forward numeric contract.
pub(super) fn tc_pick_tile_forward(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if cols < policy.forward_min_columns {
        return None;
    }
    if rows <= policy.forward_thin_max_rows
        || (policy.forward_thin_below_square_columns && cols < policy.square_tile_min)
    {
        return Some(TcTile::Thin16);
    }
    tc_pick_tile_large(rows, cols)
}

/// Tile64 predicates both output tails, but two short axes would waste the
/// whole square tile and remain on the scalar fallback.
pub(super) fn tc_pick_tile_backward_bridge(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if policy.reject_zero_axes && (rows == 0 || cols == 0) {
        return None;
    }
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return tc_pick_tile_large(rows, cols);
    }
    if policy.backward_one_axis_tile64
        && (rows >= policy.square_tile_min || cols >= policy.square_tile_min)
    {
        return Some(TcTile::Tile64);
    }
    (!policy.backward_two_small_fallback).then_some(TcTile::Tile64)
}

/// Large shapes keep the existing square-tile policy. A one-axis tail is
/// selected automatically only when its full operation is frozen above.
pub(super) fn tc_pick_tile_backward(
    op: super::super::kernel_identity::PolicyOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
) -> Option<TcTile> {
    let (batch, n_in, n_out) = dims;
    let (rows, cols) = match op {
        super::super::kernel_identity::PolicyOp::Dw => (n_in, n_out),
        super::super::kernel_identity::PolicyOp::Dx => (batch, n_in),
    };
    let tile = tc_pick_tile_backward_bridge(rows, cols)?;
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return Some(tile);
    }

    let dtype = match dtype {
        WeightDtype::F32 => super::super::kernel_identity::PolicyDtype::F32,
        WeightDtype::F16 => super::super::kernel_identity::PolicyDtype::F16,
        WeightDtype::Bf16 => super::super::kernel_identity::PolicyDtype::Bf16,
    };
    super::super::kernel_identity::LegacySm80Policy::current()
        .admits(op, dtype, dims)
        .then_some(tile)
}
