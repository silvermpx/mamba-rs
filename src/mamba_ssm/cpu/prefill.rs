//! Full-sequence CPU inference forward ("prefill").
//!
//! The per-step inference path costs one matvec pipeline per token — a full
//! page/prompt of T tokens pays T dispatches and re-streams every weight
//! matrix T times. This module runs the SAME math as the training forward's
//! batched-SGEMM pipeline (`train::forward::forward_mamba_backbone_batched`)
//! WITHOUT saving the activation tape, on inference types (`MambaWeights` —
//! the precomputed `a_neg` is the SSM decay source — and `MambaState`,
//! carried in AND out so `forward_step` continues seamlessly:
//! prefill-then-decode).
//!
//! Hard math constraints (the per-step `target.rs` template is wrong on
//! both): `norm_f` is applied at EVERY position — consumers pool over all
//! positions — and the RMSNorm epsilon comes from `dims.rms_norm_eps`,
//! never a hardcoded constant.
//!
//! Layout doctrine (measured, not aesthetic): the per-channel phases
//! (conv1d, SSM recurrence) touch ONLY channel-major `[di * T]` buffers —
//! one row-major access inside a per-channel loop is a multi-KB-stride
//! cache-miss per element and dominated the whole prefill at real T before
//! the re-layout. Cache-blocked transposes bridge to the row-major GEMM
//! I/O; gating runs as a sequential row-major pass AFTER the transpose
//! (the same two-operand product, so it stays bit-identical to the fused
//! form). In [`PrefillMode::Parallel`] EVERY phase parallelizes — GEMMs via
//! [`sgemm_forward_par`], per-channel phases via `par_chunks_mut` over
//! channels, elementwise/transposes via `par_chunks_mut` over rows/tiles —
//! with no cross-task reductions, so Parallel is bit-equal to Single by
//! construction (the parallel-invariance test pins it), and both are
//! bit-equal to the training forward (the anchor test pins that).
//!
//! Conv-state convention adaptation: the inference `MambaLayerState` stores
//! `d_conv - 1` history entries per channel; the training-style register is
//! `d_conv` wide. Prefill widens on entry (history in slots `1..d_conv`;
//! slot 0 is dead — the first shift discards it) and writes the last
//! `d_conv - 1` register entries back on exit.

use rayon::prelude::*;

use crate::ops::blas::{sgemm_forward, sgemm_forward_par};
use crate::ops::dims::MambaDims;
use crate::ops::fast_math::{fast_exp_inplace, fast_exp_scalar};
use crate::state::MambaState;
use crate::weights::{MambaLayerWeights, MambaWeights};

/// Maximum `d_state` the stack-allocated per-channel `da` buffer covers.
/// Matches the GPU dispatch ceiling for the sequential kernels.
const MAX_D_STATE: usize = 64;

/// Row-major forward GEMM `(out, x, w, bias, m, k, n)` — the shared
/// signature of [`sgemm_forward`] and [`sgemm_forward_par`].
type GemmFn = fn(&mut [f32], &[f32], &[f32], Option<&[f32]>, usize, usize, usize);

/// Transpose tile edge: 64 x 64 f32 = two 16 KB panels — both stay
/// L1-resident, so the strided side costs one miss per LINE instead of one
/// miss per ELEMENT (the naive column walk was a measured prefill
/// bottleneck at real T).
const TRANSPOSE_TILE: usize = 64;

/// GEMM / phase execution mode for the prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    /// Serial everything — the deterministic single-thread reference.
    Single,
    /// Rayon-parallel GEMMs + channel-parallel conv/SSM + row-parallel
    /// elementwise phases and transposes — the single-page serving latency
    /// mode. Bit-equal to `Single` (no cross-task reductions; tile-local
    /// GEMM accumulation).
    Parallel,
}

/// Run a per-row transform over `dst` rows zipped with `src` rows,
/// serial or rayon-parallel per `mode`. Rows are disjoint, so both
/// dispatches perform the identical per-element arithmetic.
fn zip_rows<F>(mode: PrefillMode, dst: (&mut [f32], usize), src: (&[f32], usize), f: F)
where
    F: Fn(&mut [f32], &[f32]) + Sync + Send,
{
    let (dst, dst_w) = dst;
    let (src, src_w) = src;
    match mode {
        PrefillMode::Single => dst
            .chunks_mut(dst_w)
            .zip(src.chunks(src_w))
            .for_each(|(d, s)| f(d, s)),
        PrefillMode::Parallel => dst
            .par_chunks_mut(dst_w)
            .zip(src.par_chunks(src_w))
            .for_each(|(d, s)| f(d, s)),
    }
}

/// Run a per-row transform over `buf` rows, serial or rayon-parallel.
fn for_rows<F>(mode: PrefillMode, buf: &mut [f32], row_w: usize, f: F)
where
    F: Fn(&mut [f32]) + Sync + Send,
{
    match mode {
        PrefillMode::Single => buf.chunks_mut(row_w).for_each(f),
        PrefillMode::Parallel => buf.par_chunks_mut(row_w).for_each(f),
    }
}

/// Reusable prefill scratch. `ensure` re-allocates only when the dims
/// fingerprint changes — zero allocation on the serving hot path once
/// warmed (keyed on the FULL dims, the `parallel.rs` OOB lesson).
pub struct PrefillScratch {
    fingerprint: (usize, usize, usize, usize, usize, usize, usize),
    post_norm: Vec<f32>,
    proj: Vec<f32>,
    gate_silu: Vec<f32>,
    /// Channel-major conv/SSM I/O `[di * T]` (see the module doctrine).
    x_cm: Vec<f32>,
    u_cm: Vec<f32>,
    delta_cm: Vec<f32>,
    y_cm: Vec<f32>,
    /// Row-major GEMM I/O `[T * di]`.
    u_rm: Vec<f32>,
    delta_rm: Vec<f32>,
    y_rm: Vec<f32>,
    xdbl: Vec<f32>,
    dt_in: Vec<f32>,
    out: Vec<f32>,
    /// Training-convention conv registers, channel-major `[di * dc]`.
    conv_reg: Vec<f32>,
}

fn fingerprint(dims: &MambaDims) -> (usize, usize, usize, usize, usize, usize, usize) {
    (
        dims.seq_len,
        dims.d_model,
        dims.d_inner,
        dims.d_state,
        dims.d_conv,
        dims.dt_rank,
        dims.mamba_input_dim,
    )
}

impl PrefillScratch {
    /// Allocate scratch sized for `dims`.
    pub fn new(dims: &MambaDims) -> Self {
        let t = dims.seq_len;
        let di = dims.d_inner;
        Self {
            fingerprint: fingerprint(dims),
            post_norm: vec![0.0; t * dims.d_model],
            proj: vec![0.0; t * 2 * di],
            gate_silu: vec![0.0; t * di],
            x_cm: vec![0.0; di * t],
            u_cm: vec![0.0; di * t],
            delta_cm: vec![0.0; di * t],
            y_cm: vec![0.0; di * t],
            u_rm: vec![0.0; t * di],
            delta_rm: vec![0.0; t * di],
            y_rm: vec![0.0; t * di],
            xdbl: vec![0.0; t * dims.xdbl_dim],
            dt_in: vec![0.0; t * dims.dt_rank],
            out: vec![0.0; t * dims.d_model],
            conv_reg: vec![0.0; di * dims.d_conv],
        }
    }

    /// Re-size for a different dims fingerprint (no-op when unchanged).
    pub fn ensure(&mut self, dims: &MambaDims) {
        if fingerprint(dims) != self.fingerprint {
            *self = Self::new(dims);
        }
    }
}

/// Conv1d + fused SiLU for ONE channel over the full sequence.
/// `x_col`/`u_col`/`reg` are this channel's contiguous slices — every hot
/// access is sequential. The first `d_conv - 1` positions run the shift
/// register (they tap carried history); the bulk runs tap-outer vectorized
/// FMA passes whose per-element accumulation chain
/// (`bias, +w0*x, +w1*x, ...`) is IDENTICAL to the register form, so the
/// split is bit-exact.
fn conv_channel(
    d: usize,
    u_col: &mut [f32],
    reg: &mut [f32],
    x_col: &[f32],
    lw: &MambaLayerWeights,
    dims: &MambaDims,
) {
    let dc = dims.d_conv;
    let hist = dc - 1;
    let t_len = u_col.len();
    let w_base = d * dc;
    let bias = lw.conv1d_bias[d];
    let head = hist.min(t_len);

    // Head: shift-register form over the carried history.
    for (u_td, &xb) in u_col[..head].iter_mut().zip(&x_col[..head]) {
        for k in 0..dc - 1 {
            reg[k] = reg[k + 1];
        }
        reg[dc - 1] = xb;
        let mut val = bias;
        for (r, w_k) in reg.iter().zip(&lw.conv1d_weight[w_base..w_base + dc]) {
            val += r * w_k;
        }
        *u_td = val;
    }

    // Bulk: tap-outer vector FMA (taps k of position t read x[t - hist + k]).
    if t_len > hist {
        let bulk = &mut u_col[hist..];
        bulk.fill(bias);
        for (k, &w_k) in lw.conv1d_weight[w_base..w_base + dc].iter().enumerate() {
            for (u, &xv) in bulk.iter_mut().zip(&x_col[k..k + t_len - hist]) {
                *u += xv * w_k;
            }
        }
        // Refresh the register for state writeback (slot 0 is dead).
        for (r, &xv) in reg[1..dc].iter_mut().zip(&x_col[t_len - hist..]) {
            *r = xv;
        }
    }

    // SiLU sweep (same per-element expression as the fused form).
    for u in u_col.iter_mut() {
        *u = *u / (1.0 + fast_exp_scalar(-*u));
    }
}

/// Channels per SSM task. The per-(t, d) output `y` is a serial chain of
/// `d_state` dependent FMAs (~4-cycle latency each) — the measured prefill
/// hot spot. Interleaving `SSM_BLOCK` independent channels in the state
/// loop runs that many chains concurrently (latency hiding) and amortizes
/// the shared B/C row loads, while each channel's own accumulation ORDER is
/// untouched — so the block form stays bit-exact vs the one-channel form
/// and the training forward.
const SSM_BLOCK: usize = 8;

/// SSM recurrence for a BLOCK of up to [`SSM_BLOCK`] channels over the full
/// sequence (ungated — the gate multiply happens in a sequential row-major
/// pass afterwards). `y_cols`/`ssm` are the block's channel-major slices
/// (`nch` inferred from their length); `xdbl` streams row-sequentially (B/C
/// are shared per-position vectors, loaded once per block).
fn ssm_channels(
    block: usize,
    y_cols: &mut [f32],
    ssm: &mut [f32],
    io: (&[f32], &[f32], &[f32]),
    lw: &MambaLayerWeights,
    dims: &MambaDims,
) {
    let (u_cols, delta_cols, xdbl) = io;
    let ds = dims.d_state;
    let dr = dims.dt_rank;
    let xd = dims.xdbl_dim;
    let t_len = dims.seq_len;
    let d0 = block * SSM_BLOCK;
    let nch = y_cols.len() / t_len;
    let b_offset = dr;
    let c_offset = dr + ds;

    // Lane-major locals: index [n * SSM_BLOCK + c] with the CONSTANT block
    // stride — the inner state loop compiles to straight 4-wide SIMD with
    // one contiguous load per operand. Lanes past `nch` (a tail block when
    // d_inner % SSM_BLOCK != 0) compute garbage that never mixes into valid
    // lanes (per-lane arithmetic only) and is never written back (`ssm` and
    // `y_cols` hold exactly `nch` channels).
    let mut da = [0.0f32; MAX_D_STATE * SSM_BLOCK];
    let mut hloc = [0.0f32; MAX_D_STATE * SSM_BLOCK];
    let mut du = [0.0f32; SSM_BLOCK];
    let mut y_acc = [0.0f32; SSM_BLOCK];

    // Load the block's ssm state into lane-major.
    for (c, ssm_ch) in ssm.chunks(ds).enumerate() {
        for (n, &s) in ssm_ch.iter().enumerate() {
            hloc[n * SSM_BLOCK + c] = s;
        }
    }

    for t in 0..t_len {
        let xdbl_row = t * xd;
        for (c, du_c) in du[..nch].iter_mut().enumerate() {
            let delta_d = delta_cols[c * t_len + t];
            *du_c = delta_d * u_cols[c * t_len + t];
            let a_base = (d0 + c) * ds;
            for (n, a_n) in lw.a_neg[a_base..a_base + ds].iter().enumerate() {
                da[n * SSM_BLOCK + c] = delta_d * a_n;
            }
        }
        fast_exp_inplace(&mut da[..ds * SSM_BLOCK]);
        y_acc.fill(0.0);
        for (n, (da_l, h_l)) in da
            .chunks_exact(SSM_BLOCK)
            .zip(hloc.chunks_exact_mut(SSM_BLOCK))
            .enumerate()
            .take(ds)
        {
            let b_n = xdbl[xdbl_row + b_offset + n];
            let c_n = xdbl[xdbl_row + c_offset + n];
            for c in 0..SSM_BLOCK {
                let h = da_l[c] * h_l[c] + du[c] * b_n;
                h_l[c] = h;
                y_acc[c] += h * c_n;
            }
        }
        for (c, &y_c) in y_acc[..nch].iter().enumerate() {
            let u_td = u_cols[c * t_len + t];
            y_cols[c * t_len + t] = y_c + lw.d_param[d0 + c] * u_td;
        }
    }

    // Write the block's ssm state back from lane-major.
    for (c, ssm_ch) in ssm.chunks_mut(ds).enumerate() {
        for (n, s) in ssm_ch.iter_mut().enumerate() {
            *s = hloc[n * SSM_BLOCK + c];
        }
    }
}

/// `dst[t * di + d] = src[d * t_len + t]` — channel-major → row-major,
/// cache-blocked; parallel over row-tiles in `Parallel` (disjoint dst rows).
fn transpose_cm_to_rm(dst: &mut [f32], src: &[f32], di: usize, t_len: usize, mode: PrefillMode) {
    let row_block = TRANSPOSE_TILE * di;
    let work = |(tb, dst_tile): (usize, &mut [f32])| {
        let t0 = tb * TRANSPOSE_TILE;
        let rows = dst_tile.len() / di;
        for d0 in (0..di).step_by(TRANSPOSE_TILE) {
            let d1 = (d0 + TRANSPOSE_TILE).min(di);
            for d in d0..d1 {
                let col = &src[d * t_len + t0..d * t_len + t0 + rows];
                for (r, &v) in col.iter().enumerate() {
                    dst_tile[r * di + d] = v;
                }
            }
        }
    };
    match mode {
        PrefillMode::Single => dst[..t_len * di]
            .chunks_mut(row_block)
            .enumerate()
            .for_each(work),
        PrefillMode::Parallel => dst[..t_len * di]
            .par_chunks_mut(row_block)
            .enumerate()
            .for_each(work),
    }
}

/// `dst[d * t_len + t] = src[t * src_stride + src_off + d]` — strided
/// row-major → channel-major, cache-blocked; parallel over channel-tiles in
/// `Parallel` (disjoint dst columns). `src_stride`/`src_off` let the x
/// branch be pulled straight out of the interleaved `[T, 2*di]` in_proj
/// output without a compaction pass.
fn transpose_rows_to_cm(
    dst: &mut [f32],
    src: &[f32],
    di: usize,
    t_len: usize,
    stride: (usize, usize),
    mode: PrefillMode,
) {
    let (src_stride, src_off) = stride;
    let col_block = TRANSPOSE_TILE * t_len;
    let work = |(db, dst_tile): (usize, &mut [f32])| {
        let d0 = db * TRANSPOSE_TILE;
        for t0 in (0..t_len).step_by(TRANSPOSE_TILE) {
            let t1 = (t0 + TRANSPOSE_TILE).min(t_len);
            for (c, col) in dst_tile.chunks_mut(t_len).enumerate() {
                for (i, v) in col[t0..t1].iter_mut().enumerate() {
                    *v = src[(t0 + i) * src_stride + src_off + d0 + c];
                }
            }
        }
    };
    match mode {
        PrefillMode::Single => dst[..di * t_len]
            .chunks_mut(col_block)
            .enumerate()
            .for_each(work),
        PrefillMode::Parallel => dst[..di * t_len]
            .par_chunks_mut(col_block)
            .enumerate()
            .for_each(work),
    }
}

/// Full-sequence CPU inference forward of the M1 backbone (serial mode).
///
/// - `temporal_out`: `[T * d_model]` — the POST-norm_f output at EVERY
///   position (consumers pool over all of them).
/// - `mamba_input_flat`: `[T * mamba_input_dim]`.
/// - `w`: inference weights. `a_neg` is the SSM decay source — callers that
///   mutate `a_log` must refresh `a_neg` before prefilling.
/// - `state`: recurrent conv/SSM state, carried in AND out
///   (prefill-then-decode into [`crate::inference`]'s step path).
/// - `scratch`: caller-owned; `ensure`d to `dims` internally.
pub fn forward_mamba_backbone_prefill(
    temporal_out: &mut [f32],
    mamba_input_flat: &[f32],
    w: &MambaWeights,
    state: &mut MambaState,
    scratch: &mut PrefillScratch,
    dims: &MambaDims,
) {
    forward_mamba_backbone_prefill_mode(
        temporal_out,
        mamba_input_flat,
        w,
        state,
        scratch,
        dims,
        PrefillMode::Single,
    );
}

/// [`forward_mamba_backbone_prefill`] with an explicit execution mode.
pub fn forward_mamba_backbone_prefill_mode(
    temporal_out: &mut [f32],
    mamba_input_flat: &[f32],
    w: &MambaWeights,
    state: &mut MambaState,
    scratch: &mut PrefillScratch,
    dims: &MambaDims,
    mode: PrefillMode,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dr = dims.dt_rank;
    let xd = dims.xdbl_dim;
    let mid = dims.mamba_input_dim;
    let t_len = dims.seq_len;
    let hist = dc - 1;
    let eps = dims.rms_norm_eps;

    assert!(ds <= MAX_D_STATE, "d_state > {MAX_D_STATE} unsupported");
    assert_eq!(temporal_out.len(), t_len * dm, "temporal_out shape");
    assert_eq!(mamba_input_flat.len(), t_len * mid, "input shape");
    assert_eq!(state.layers.len(), dims.n_layers, "state layer count");
    scratch.ensure(dims);

    let gemm: GemmFn = match mode {
        PrefillMode::Single => sgemm_forward,
        PrefillMode::Parallel => sgemm_forward_par,
    };

    // Input projection (identity D2D for HF checkpoints with empty proj).
    if w.input_proj_w.is_empty() {
        debug_assert_eq!(mid, dm, "identity input_proj requires input_dim == d_model");
        temporal_out[..t_len * dm].copy_from_slice(&mamba_input_flat[..t_len * dm]);
    } else {
        gemm(
            temporal_out,
            mamba_input_flat,
            &w.input_proj_w,
            Some(&w.input_proj_b),
            t_len,
            mid,
            dm,
        );
    }

    for (layer_idx, lw) in w.layers.iter().enumerate() {
        let lstate = &mut state.layers[layer_idx];

        // F1: RMSNorm at every position (dims eps — never hardcoded).
        // `temporal_out` doubles as the residual: it is only read here and
        // updated in-place at F6, so no residual copy is needed.
        zip_rows(
            mode,
            (&mut scratch.post_norm[..t_len * dm], dm),
            (&temporal_out[..t_len * dm], dm),
            |dst, src| {
                let mut sum_sq = 0.0_f32;
                for &v in src {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / dm as f32 + eps).sqrt();
                for ((y, &s), nw) in dst.iter_mut().zip(src).zip(&lw.norm_weight[..dm]) {
                    *y = s * inv_rms * nw;
                }
            },
        );

        // F2: in_proj GEMM.
        gemm(
            &mut scratch.proj,
            &scratch.post_norm,
            &lw.in_proj_w,
            None,
            t_len,
            dm,
            2 * di,
        );

        // F3: gate SiLU (the x branch stays in `proj` for the conv phase).
        zip_rows(
            mode,
            (&mut scratch.gate_silu[..t_len * di], di),
            (&scratch.proj[..t_len * 2 * di], 2 * di),
            |gs, proj_row| {
                for (g_out, &g) in gs.iter_mut().zip(&proj_row[di..2 * di]) {
                    *g_out = g * (1.0 / (1.0 + fast_exp_scalar(-g)));
                }
            },
        );

        // Pull the x branch out of the interleaved in_proj output into
        // channel-major (blocked transpose — the per-channel conv loop must
        // never touch the row-major `proj`).
        transpose_rows_to_cm(
            &mut scratch.x_cm,
            &scratch.proj,
            di,
            t_len,
            (2 * di, 0),
            mode,
        );

        // Widen the inference conv history into the training-style register.
        for d in 0..di {
            scratch.conv_reg[d * dc] = 0.0;
            scratch.conv_reg[d * dc + 1..d * dc + dc]
                .copy_from_slice(&lstate.conv_state[d * hist..(d + 1) * hist]);
        }

        // F4a: conv1d + SiLU, per channel (contiguous columns only).
        {
            let x_cm = &scratch.x_cm[..];
            match mode {
                PrefillMode::Single => {
                    for (d, (u_col, reg)) in scratch
                        .u_cm
                        .chunks_mut(t_len)
                        .zip(scratch.conv_reg.chunks_mut(dc))
                        .enumerate()
                    {
                        conv_channel(d, u_col, reg, &x_cm[d * t_len..(d + 1) * t_len], lw, dims);
                    }
                }
                PrefillMode::Parallel => {
                    scratch
                        .u_cm
                        .par_chunks_mut(t_len)
                        .zip(scratch.conv_reg.par_chunks_mut(dc))
                        .enumerate()
                        .for_each(|(d, (u_col, reg))| {
                            conv_channel(
                                d,
                                u_col,
                                reg,
                                &x_cm[d * t_len..(d + 1) * t_len],
                                lw,
                                dims,
                            );
                        });
                }
            }
        }
        transpose_cm_to_rm(&mut scratch.u_rm, &scratch.u_cm, di, t_len, mode);

        // F4b: x_proj GEMM + dt gather.
        gemm(
            &mut scratch.xdbl,
            &scratch.u_rm,
            &lw.x_proj_w,
            None,
            t_len,
            di,
            xd,
        );
        zip_rows(
            mode,
            (&mut scratch.dt_in[..t_len * dr], dr),
            (&scratch.xdbl[..t_len * xd], xd),
            |dst, src| dst.copy_from_slice(&src[..dr]),
        );

        // F4c: dt_proj GEMM + softplus (threshold 20, same as training).
        gemm(
            &mut scratch.delta_rm,
            &scratch.dt_in,
            &lw.dt_proj_w,
            Some(&lw.dt_proj_b),
            t_len,
            dr,
            di,
        );
        for_rows(mode, &mut scratch.delta_rm[..t_len * di], di, |row| {
            for v in row {
                if *v <= 20.0 {
                    *v = fast_exp_scalar(*v).ln_1p();
                }
            }
        });
        transpose_rows_to_cm(
            &mut scratch.delta_cm,
            &scratch.delta_rm,
            di,
            t_len,
            (di, 0),
            mode,
        );

        // F4d: SSM recurrence, per channel block (ungated).
        {
            let u_cm = &scratch.u_cm[..];
            let delta_cm = &scratch.delta_cm[..];
            let xdbl = &scratch.xdbl[..];
            let blk_t = SSM_BLOCK * t_len;
            match mode {
                PrefillMode::Single => {
                    for (blk, (y_cols, ssm)) in scratch
                        .y_cm
                        .chunks_mut(blk_t)
                        .zip(lstate.ssm_state.chunks_mut(SSM_BLOCK * ds))
                        .enumerate()
                    {
                        let span = blk * blk_t..blk * blk_t + y_cols.len();
                        let io = (&u_cm[span.clone()], &delta_cm[span], xdbl);
                        ssm_channels(blk, y_cols, ssm, io, lw, dims);
                    }
                }
                PrefillMode::Parallel => {
                    scratch
                        .y_cm
                        .par_chunks_mut(blk_t)
                        .zip(lstate.ssm_state.par_chunks_mut(SSM_BLOCK * ds))
                        .enumerate()
                        .for_each(|(blk, (y_cols, ssm))| {
                            let span = blk * blk_t..blk * blk_t + y_cols.len();
                            let io = (&u_cm[span.clone()], &delta_cm[span], xdbl);
                            ssm_channels(blk, y_cols, ssm, io, lw, dims);
                        });
                }
            }
        }
        transpose_cm_to_rm(&mut scratch.y_rm, &scratch.y_cm, di, t_len, mode);

        // F4e: gating in row-major (same two-operand product as the fused
        // per-channel version — bit-identical, cache-friendly).
        zip_rows(
            mode,
            (&mut scratch.y_rm[..t_len * di], di),
            (&scratch.gate_silu[..t_len * di], di),
            |y_row, g_row| {
                for (y, &g) in y_row.iter_mut().zip(g_row) {
                    *y *= g;
                }
            },
        );

        // Write the conv history back (last dc-1 register entries).
        for d in 0..di {
            lstate.conv_state[d * hist..(d + 1) * hist]
                .copy_from_slice(&scratch.conv_reg[d * dc + 1..d * dc + dc]);
        }

        // F5/F6: out_proj GEMM + in-place residual add.
        gemm(
            &mut scratch.out,
            &scratch.y_rm,
            &lw.out_proj_w,
            None,
            t_len,
            di,
            dm,
        );
        zip_rows(
            mode,
            (&mut temporal_out[..t_len * dm], dm),
            (&scratch.out[..t_len * dm], dm),
            |y_row, o_row| {
                for (y, &o) in y_row.iter_mut().zip(o_row) {
                    *y += o;
                }
            },
        );
    }

    // norm_f at EVERY position, dims eps.
    for_rows(mode, &mut temporal_out[..t_len * dm], dm, |row| {
        let mut sum_sq = 0.0_f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / dm as f32 + eps).sqrt();
        for (y, nw) in row.iter_mut().zip(&w.norm_f_weight[..dm]) {
            *y *= inv_rms * nw;
        }
    });
}

/// Prefill a batch of independent sequences in parallel (one rayon task per
/// sample; each task runs [`PrefillMode::Single`] internally — batch-level
/// parallelism already saturates the cores). Scratches are per-sample and
/// caller-owned so the serving layer controls its memory bound.
pub fn prefill_batch(
    outputs: &mut [f32],
    inputs: &[f32],
    w: &MambaWeights,
    states: &mut [MambaState],
    scratches: &mut [PrefillScratch],
    dims: &MambaDims,
) {
    let t_len = dims.seq_len;
    let dm = dims.d_model;
    let mid = dims.mamba_input_dim;
    let b = states.len();
    assert_eq!(scratches.len(), b, "one scratch per sample");
    assert_eq!(outputs.len(), b * t_len * dm, "outputs shape");
    assert_eq!(inputs.len(), b * t_len * mid, "inputs shape");

    outputs
        .par_chunks_mut(t_len * dm)
        .zip(states.par_iter_mut())
        .zip(scratches.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((out, state), scratch))| {
            let inp = &inputs[i * t_len * mid..(i + 1) * t_len * mid];
            forward_mamba_backbone_prefill(out, inp, w, state, scratch, dims);
        });
}
