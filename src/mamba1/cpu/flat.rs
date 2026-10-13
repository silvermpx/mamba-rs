//! O5 contiguous activation buffers for Mamba forward/backward.
//!
//! Replaces `Vec<MambaStepSaved>` (18 Vecs per step, ~600 heap allocations per
//! sample) with a single flat `Vec<f32>` per layer. All fields for all timesteps
//! are packed contiguously in memory: `data[t * step_stride + field_offset .. +len]`.
//!
//! Layout per timestep (13 fields, no `pre_norm` -- was duplicate of `residual`):
//!
//! | # | Field           | Size                         |
//! |---|-----------------|------------------------------|
//! | 0 | residual        | d_model                      |
//! | 1 | rms_val         | 1                            |
//! | 2 | post_norm       | d_model                      |
//! | 3 | x_branch        | d_inner                      |
//! | 4 | post_conv       | d_inner                      |
//! | 5 | u               | d_inner                      |
//! | 6 | xdbl            | xdbl_dim (dt_rank+2*d_state) |
//! | 7 | delta_raw       | d_inner                      |
//! | 8 | delta           | d_inner                      |
//! | 9 | y               | d_inner                      |
//! |10 | gate_pre_silu   | d_inner                      |
//! |11 | gate_post_silu  | d_inner                      |
//! |12 | gated           | d_inner                      |
//! |13 | bcdt_rstd       | 3 with FalconMamba's norms, else 0 |
//!
//! The states are not kept per step: they are `d_inner * d_state` wide, and
//! kept so they were most of the tape. The window keeps the conv register
//! it starts from, whose every later state is that register shifted through
//! the window's `x_branch` rows, and the SSM state at every `stride`-th
//! step; the backward replays each segment's states from its checkpoint
//! with the forward's arithmetic, so the gradients are the same bits at
//! every stride.

use crate::ops::dims::MambaDims;

// ---------------------------------------------------------------------------
// Field offsets within one timestep
// ---------------------------------------------------------------------------

/// Byte offsets (in f32 units) for each field within one timestep of the flat
/// activation buffer. Computed once from [`MambaDims`], reused for all accessors.
#[derive(Debug, Clone, Copy)]
pub struct FieldOffsets {
    pub residual: usize,
    pub rms_val: usize,
    pub post_norm: usize,
    pub x_branch: usize,
    pub post_conv: usize,
    pub u: usize,
    pub xdbl: usize,
    pub delta_raw: usize,
    pub delta: usize,
    pub y: usize,
    pub gate_pre_silu: usize,
    pub gate_post_silu: usize,
    pub gated: usize,
    /// FalconMamba's three dt, B and C norm scales (none for Mamba-1).
    pub bcdt_rstd: usize,
    /// Total f32 elements per timestep.
    pub step_stride: usize,
}

impl FieldOffsets {
    /// Compute field offsets from Mamba dimensions.
    pub fn new(dims: &MambaDims) -> Self {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let xdbl = dims.xdbl_dim;

        let mut off = 0usize;

        let residual = off;
        off += dm;

        let rms_val = off;
        off += 1;

        let post_norm = off;
        off += dm;

        let x_branch = off;
        off += di;

        let post_conv = off;
        off += di;

        let u = off;
        off += di;

        let xdbl_off = off;
        off += xdbl;

        let delta_raw = off;
        off += di;

        let delta = off;
        off += di;

        let y = off;
        off += di;

        let gate_pre_silu = off;
        off += di;

        let gate_post_silu = off;
        off += di;

        let gated = off;
        off += di;

        let bcdt_rstd = off;
        if dims.bcdt_rms_eps.is_some() {
            off += 3;
        }

        Self {
            residual,
            rms_val,
            post_norm,
            x_branch,
            post_conv,
            u,
            xdbl: xdbl_off,
            delta_raw,
            delta,
            y,
            gate_pre_silu,
            gate_post_silu,
            gated,
            bcdt_rstd,
            step_stride: off,
        }
    }
}

// ---------------------------------------------------------------------------
// MambaLayerFlat — one layer, all timesteps in a single Vec<f32>
// ---------------------------------------------------------------------------

/// Steps between the SSM checkpoints of a tape built by
/// [`MambaLayerFlat::zeros`]: the backward replays at most this many steps
/// of a layer's states at a time, in a segment buffer of as many states.
pub const TAPE_STRIDE: usize = 16;

/// Contiguous activation buffer for one Mamba layer across all timesteps.
///
/// Replaces `Vec<MambaStepSaved>` (33 steps x 18 Vecs = ~594 heap allocs)
/// with a single allocation of `seq_len * step_stride` f32 elements.
pub struct MambaLayerFlat {
    /// Packed activation data: `[seq_len * step_stride]`.
    pub data: Vec<f32>,
    /// Precomputed field offsets and step stride.
    pub offsets: FieldOffsets,
    /// Dimensions (kept for accessor bounds and bulk copy methods).
    pub dims: MambaDims,
    /// The conv register the window starts from: `[d_inner * d_conv]`.
    pub conv_init: Vec<f32>,
    /// The SSM state before steps 0, `stride`, `2 * stride`, ...:
    /// `[ceil(seq_len / stride) * d_inner * d_state]`.
    pub ssm_ckpt: Vec<f32>,
    /// Steps between the SSM checkpoints.
    pub stride: usize,
}

impl MambaLayerFlat {
    /// Allocate a zero-filled flat buffer for `dims.seq_len` timesteps,
    /// checkpointing the SSM state every [`TAPE_STRIDE`] steps.
    pub fn zeros(dims: MambaDims) -> Self {
        Self::zeros_strided(dims, TAPE_STRIDE)
    }

    /// [`Self::zeros`] checkpointing the SSM state every `stride` steps
    /// (at least one).
    pub fn zeros_strided(dims: MambaDims, stride: usize) -> Self {
        let offsets = FieldOffsets::new(&dims);
        let total = dims.seq_len * offsets.step_stride;
        let stride = stride.max(1);
        let state = dims.d_inner * dims.d_state;
        Self {
            data: vec![0.0; total],
            offsets,
            dims,
            conv_init: vec![0.0; dims.d_inner * dims.d_conv],
            ssm_ckpt: vec![0.0; dims.seq_len.div_ceil(stride) * state],
            stride,
        }
    }

    /// The SSM state checkpointed before step `j * stride`.
    #[inline]
    pub fn ssm_ckpt(&self, j: usize) -> &[f32] {
        let n = self.dims.d_inner * self.dims.d_state;
        &self.ssm_ckpt[j * n..(j + 1) * n]
    }

    /// Mutable [`Self::ssm_ckpt`].
    #[inline]
    pub fn ssm_ckpt_mut(&mut self, j: usize) -> &mut [f32] {
        let n = self.dims.d_inner * self.dims.d_state;
        &mut self.ssm_ckpt[j * n..(j + 1) * n]
    }

    /// The conv register at step `t` into `out` (`[d_inner * d_conv]`): the
    /// window's starting register shifted `t + 1` times, the `x_branch` rows
    /// of steps `0..=t` entering at its end.
    pub fn conv_window(&self, t: usize, out: &mut [f32]) {
        let (di, dc) = (self.dims.d_inner, self.dims.d_conv);
        for (d, reg) in out[..di * dc].chunks_exact_mut(dc).enumerate() {
            for (k, r) in reg.iter_mut().enumerate() {
                let from = k + t + 1;
                *r = if from < dc {
                    self.conv_init[d * dc + from]
                } else {
                    self.x_branch(from - dc)[d]
                };
            }
        }
    }

    // -- helpers --

    /// Base index for timestep `t`.
    #[inline(always)]
    fn base(&self, t: usize) -> usize {
        t * self.offsets.step_stride
    }

    // -----------------------------------------------------------------------
    // Read accessors
    // -----------------------------------------------------------------------

    /// `residual[t]`: `&[d_model]`.
    #[inline]
    pub fn residual(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.residual;
        &self.data[b..b + self.dims.d_model]
    }

    /// `rms_val[t]`: scalar.
    #[inline]
    pub fn rms_val(&self, t: usize) -> f32 {
        self.data[self.base(t) + self.offsets.rms_val]
    }

    /// `post_norm[t]`: `&[d_model]`.
    #[inline]
    pub fn post_norm(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.post_norm;
        &self.data[b..b + self.dims.d_model]
    }

    /// `x_branch[t]`: `&[d_inner]`.
    #[inline]
    pub fn x_branch(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.x_branch;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `post_conv[t]`: `&[d_inner]`.
    #[inline]
    pub fn post_conv(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.post_conv;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `u[t]`: `&[d_inner]`.
    #[inline]
    pub fn u(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.u;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `xdbl[t]`: `&[xdbl_dim]`.
    #[inline]
    pub fn xdbl(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.xdbl;
        &self.data[b..b + self.dims.xdbl_dim]
    }

    /// `delta_raw[t]`: `&[d_inner]`.
    #[inline]
    pub fn delta_raw(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.delta_raw;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `delta[t]`: `&[d_inner]`.
    #[inline]
    pub fn delta(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.delta;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `y[t]`: `&[d_inner]`.
    #[inline]
    pub fn y(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.y;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `gate_pre_silu[t]`: `&[d_inner]`.
    #[inline]
    pub fn gate_pre_silu(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.gate_pre_silu;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `gate_post_silu[t]`: `&[d_inner]`.
    #[inline]
    pub fn gate_post_silu(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.gate_post_silu;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `gated[t]`: `&[d_inner]`.
    #[inline]
    pub fn gated(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.gated;
        &self.data[b..b + self.dims.d_inner]
    }

    /// `bcdt_rstd[t]`: the dt, B and C norm scales, empty for Mamba-1.
    #[inline]
    pub fn bcdt_rstd(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.bcdt_rstd;
        let n = self.offsets.step_stride - self.offsets.bcdt_rstd;
        &self.data[b..b + n]
    }

    // -----------------------------------------------------------------------
    // Write accessors
    // -----------------------------------------------------------------------

    /// `residual_mut[t]`: `&mut [d_model]`.
    #[inline]
    pub fn residual_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.residual;
        let dm = self.dims.d_model;
        &mut self.data[b..b + dm]
    }

    /// Set `rms_val[t]`.
    #[inline]
    pub fn set_rms_val(&mut self, t: usize, val: f32) {
        let idx = self.base(t) + self.offsets.rms_val;
        self.data[idx] = val;
    }

    /// `post_norm_mut[t]`: `&mut [d_model]`.
    #[inline]
    pub fn post_norm_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.post_norm;
        let dm = self.dims.d_model;
        &mut self.data[b..b + dm]
    }

    /// `x_branch_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn x_branch_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.x_branch;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `post_conv_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn post_conv_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.post_conv;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `u_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn u_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.u;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `xdbl_mut[t]`: `&mut [xdbl_dim]`.
    #[inline]
    pub fn xdbl_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.xdbl;
        let xd = self.dims.xdbl_dim;
        &mut self.data[b..b + xd]
    }

    /// `delta_raw_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn delta_raw_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.delta_raw;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `delta_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn delta_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.delta;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `y_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn y_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.y;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `gate_pre_silu_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn gate_pre_silu_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.gate_pre_silu;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `gate_post_silu_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn gate_post_silu_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.gate_post_silu;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `gated_mut[t]`: `&mut [d_inner]`.
    #[inline]
    pub fn gated_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.gated;
        let di = self.dims.d_inner;
        &mut self.data[b..b + di]
    }

    /// `bcdt_rstd_mut[t]`: the dt, B and C norm scales, empty for Mamba-1.
    #[inline]
    pub fn bcdt_rstd_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.bcdt_rstd;
        let n = self.offsets.step_stride - self.offsets.bcdt_rstd;
        &mut self.data[b..b + n]
    }

    // -----------------------------------------------------------------------
    // Bulk copy methods — gather columns across timesteps for batched SGEMM
    // -----------------------------------------------------------------------

    /// Copy `post_norm` for all timesteps into `dst`: `[seq_len * d_model]`.
    ///
    /// Used by batched in_proj forward: `in_proj(post_norm_all)`.
    pub fn copy_post_norm_all(&self, dst: &mut [f32]) {
        let dm = self.dims.d_model;
        for t in 0..self.dims.seq_len {
            let src = self.post_norm(t);
            dst[t * dm..(t + 1) * dm].copy_from_slice(src);
        }
    }

    /// Copy `u` for all timesteps into `dst`: `[seq_len * d_inner]`.
    ///
    /// Used by batched x_proj forward: `x_proj(u_all)`.
    pub fn copy_u_all(&self, dst: &mut [f32]) {
        let di = self.dims.d_inner;
        for t in 0..self.dims.seq_len {
            let src = self.u(t);
            dst[t * di..(t + 1) * di].copy_from_slice(src);
        }
    }

    /// Copy `gated` for all timesteps into `dst`: `[seq_len * d_inner]`.
    ///
    /// Used by batched out_proj forward: `out_proj(gated_all)`.
    pub fn copy_gated_all(&self, dst: &mut [f32]) {
        let di = self.dims.d_inner;
        for t in 0..self.dims.seq_len {
            let src = self.gated(t);
            dst[t * di..(t + 1) * di].copy_from_slice(src);
        }
    }

    /// Copy `xdbl` delta_raw columns for all timesteps into `dst`:
    /// `[seq_len * dt_rank]`.
    ///
    /// Extracts the first `dt_rank` elements of `xdbl[t]` for batched dt_proj.
    pub fn copy_xdbl_dt_all(&self, dst: &mut [f32]) {
        let dr = self.dims.dt_rank;
        for t in 0..self.dims.seq_len {
            let src = self.xdbl(t);
            dst[t * dr..(t + 1) * dr].copy_from_slice(&src[..dr]);
        }
    }
}

// ---------------------------------------------------------------------------
// MambaBackboneFlat — all layers + input projection
// ---------------------------------------------------------------------------

/// Flat activation storage for the entire Mamba backbone.
///
/// Contiguous flat activation storage for the full backbone (all layers).
pub struct MambaBackboneFlat {
    /// Input projection inputs across all timesteps: `[seq_len * mamba_input_dim]`.
    pub input_proj_inputs: Vec<f32>,
    /// Per-layer flat activation buffers.
    pub layers: Vec<MambaLayerFlat>,
    /// Saved pre-norm_f input for backward: `[seq_len * d_model]`.
    pub norm_f_input: Vec<f32>,
    /// Saved RMS values per timestep for norm_f backward: `[seq_len]`.
    pub norm_f_rms: Vec<f32>,
    /// The rows entering each layer's gated MLP, `[seq_len * d_model]` per
    /// layer; empty for a model without one.
    pub mlp_in: Vec<Vec<f32>>,
}

impl MambaBackboneFlat {
    /// Allocate all zero-filled flat buffers, every layer checkpointing
    /// its SSM state every [`TAPE_STRIDE`] steps.
    pub fn zeros(dims: MambaDims) -> Self {
        Self::zeros_strided(dims, TAPE_STRIDE)
    }

    /// [`Self::zeros`] checkpointing every `stride` steps.
    pub fn zeros_strided(dims: MambaDims, stride: usize) -> Self {
        Self {
            input_proj_inputs: vec![0.0; dims.seq_len * dims.mamba_input_dim],
            layers: (0..dims.n_layers)
                .map(|_| MambaLayerFlat::zeros_strided(dims, stride))
                .collect(),
            norm_f_input: vec![0.0; dims.seq_len * dims.d_model],
            norm_f_rms: vec![0.0; dims.seq_len],
            mlp_in: (0..dims.n_layers)
                .filter(|_| dims.mlp_hidden > 0)
                .map(|_| vec![0.0; dims.seq_len * dims.d_model])
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Default dims matching MambaConfig::default() with seq_len=33.
    fn default_dims() -> MambaDims {
        MambaDims::new((128, 256, 16, 4, 8, 33, 346, 3))
    }

    #[test]
    fn test_field_offsets_no_overlap() {
        let dims = default_dims();
        let o = FieldOffsets::new(&dims);

        // Collect (offset, size) for each field.
        let dm = dims.d_model;
        let di = dims.d_inner;
        let xd = dims.xdbl_dim;

        let fields: Vec<(usize, usize)> = vec![
            (o.residual, dm),
            (o.rms_val, 1),
            (o.post_norm, dm),
            (o.x_branch, di),
            (o.post_conv, di),
            (o.u, di),
            (o.xdbl, xd),
            (o.delta_raw, di),
            (o.delta, di),
            (o.y, di),
            (o.gate_pre_silu, di),
            (o.gate_post_silu, di),
            (o.gated, di),
        ];

        // Verify each field ends exactly where the next begins (no gaps, no overlaps).
        for i in 0..fields.len() - 1 {
            let end_i = fields[i].0 + fields[i].1;
            let start_next = fields[i + 1].0;
            assert_eq!(
                end_i,
                start_next,
                "field {} ends at {end_i} but field {} starts at {start_next}",
                i,
                i + 1,
            );
        }

        // Last field ends exactly at step_stride.
        let last = fields.last().unwrap();
        assert_eq!(
            last.0 + last.1,
            o.step_stride,
            "last field end ({}) != step_stride ({})",
            last.0 + last.1,
            o.step_stride,
        );

        // Verify step_stride value for default dims:
        // 128+1+128+256+256+256+40+256+256+256+256+256+256 = 2601
        assert_eq!(o.step_stride, 2601);
    }

    #[test]
    fn test_field_accessor_roundtrip() {
        let dims = default_dims();
        let mut layer = MambaLayerFlat::zeros(dims);

        // Write sentinel values to each field at t=0 and t=dims.seq_len-1.
        for &t in &[0usize, dims.seq_len - 1] {
            // residual
            layer.residual_mut(t).iter_mut().for_each(|v| *v = 1.0);
            assert!(layer.residual(t).iter().all(|&v| v == 1.0));
            assert_eq!(layer.residual(t).len(), dims.d_model);

            // rms_val
            layer.set_rms_val(t, 42.0);
            assert_eq!(layer.rms_val(t), 42.0);

            // post_norm
            layer.post_norm_mut(t).iter_mut().for_each(|v| *v = 2.0);
            assert!(layer.post_norm(t).iter().all(|&v| v == 2.0));
            assert_eq!(layer.post_norm(t).len(), dims.d_model);

            // x_branch
            layer.x_branch_mut(t).iter_mut().for_each(|v| *v = 3.0);
            assert!(layer.x_branch(t).iter().all(|&v| v == 3.0));
            assert_eq!(layer.x_branch(t).len(), dims.d_inner);

            // post_conv
            layer.post_conv_mut(t).iter_mut().for_each(|v| *v = 5.0);
            assert!(layer.post_conv(t).iter().all(|&v| v == 5.0));
            assert_eq!(layer.post_conv(t).len(), dims.d_inner);

            // u
            layer.u_mut(t).iter_mut().for_each(|v| *v = 6.0);
            assert!(layer.u(t).iter().all(|&v| v == 6.0));
            assert_eq!(layer.u(t).len(), dims.d_inner);

            // xdbl
            layer.xdbl_mut(t).iter_mut().for_each(|v| *v = 7.0);
            assert!(layer.xdbl(t).iter().all(|&v| v == 7.0));
            assert_eq!(layer.xdbl(t).len(), dims.xdbl_dim);

            // delta_raw
            layer.delta_raw_mut(t).iter_mut().for_each(|v| *v = 8.0);
            assert!(layer.delta_raw(t).iter().all(|&v| v == 8.0));
            assert_eq!(layer.delta_raw(t).len(), dims.d_inner);

            // delta
            layer.delta_mut(t).iter_mut().for_each(|v| *v = 9.0);
            assert!(layer.delta(t).iter().all(|&v| v == 9.0));
            assert_eq!(layer.delta(t).len(), dims.d_inner);

            // y
            layer.y_mut(t).iter_mut().for_each(|v| *v = 13.0);
            assert!(layer.y(t).iter().all(|&v| v == 13.0));
            assert_eq!(layer.y(t).len(), dims.d_inner);

            // gate_pre_silu
            layer
                .gate_pre_silu_mut(t)
                .iter_mut()
                .for_each(|v| *v = 14.0);
            assert!(layer.gate_pre_silu(t).iter().all(|&v| v == 14.0));
            assert_eq!(layer.gate_pre_silu(t).len(), dims.d_inner);

            // gate_post_silu
            layer
                .gate_post_silu_mut(t)
                .iter_mut()
                .for_each(|v| *v = 15.0);
            assert!(layer.gate_post_silu(t).iter().all(|&v| v == 15.0));
            assert_eq!(layer.gate_post_silu(t).len(), dims.d_inner);

            // gated
            layer.gated_mut(t).iter_mut().for_each(|v| *v = 16.0);
            assert!(layer.gated(t).iter().all(|&v| v == 16.0));
            assert_eq!(layer.gated(t).len(), dims.d_inner);
        }

        // Verify t=0 and t=last don't bleed into each other:
        // t=1 should still be all zeros (untouched).
        if dims.seq_len > 2 {
            assert!(layer.residual(1).iter().all(|&v| v == 0.0));
            assert_eq!(layer.rms_val(1), 0.0);
            assert!(layer.gated(1).iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_bulk_copy_post_norm_all() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 4, 346, 3));
        let mut layer = MambaLayerFlat::zeros(dims);

        // Fill post_norm for each timestep with distinct values.
        for t in 0..dims.seq_len {
            let val = (t + 1) as f32;
            layer.post_norm_mut(t).iter_mut().for_each(|v| *v = val);
        }

        let mut dst = vec![0.0f32; dims.seq_len * dims.d_model];
        layer.copy_post_norm_all(&mut dst);

        for t in 0..dims.seq_len {
            let expected = (t + 1) as f32;
            let chunk = &dst[t * dims.d_model..(t + 1) * dims.d_model];
            assert!(
                chunk.iter().all(|&v| v == expected),
                "post_norm_all mismatch at t={t}",
            );
        }
    }

    #[test]
    fn test_bulk_copy_u_all() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 4, 346, 3));
        let mut layer = MambaLayerFlat::zeros(dims);

        for t in 0..dims.seq_len {
            let val = (t + 10) as f32;
            layer.u_mut(t).iter_mut().for_each(|v| *v = val);
        }

        let mut dst = vec![0.0f32; dims.seq_len * dims.d_inner];
        layer.copy_u_all(&mut dst);

        for t in 0..dims.seq_len {
            let expected = (t + 10) as f32;
            let chunk = &dst[t * dims.d_inner..(t + 1) * dims.d_inner];
            assert!(
                chunk.iter().all(|&v| v == expected),
                "u_all mismatch at t={t}",
            );
        }
    }

    #[test]
    fn test_bulk_copy_gated_all() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 4, 346, 3));
        let mut layer = MambaLayerFlat::zeros(dims);

        for t in 0..dims.seq_len {
            let val = (t + 20) as f32;
            layer.gated_mut(t).iter_mut().for_each(|v| *v = val);
        }

        let mut dst = vec![0.0f32; dims.seq_len * dims.d_inner];
        layer.copy_gated_all(&mut dst);

        for t in 0..dims.seq_len {
            let expected = (t + 20) as f32;
            let chunk = &dst[t * dims.d_inner..(t + 1) * dims.d_inner];
            assert!(
                chunk.iter().all(|&v| v == expected),
                "gated_all mismatch at t={t}",
            );
        }
    }

    #[test]
    fn test_bulk_copy_xdbl_dt_all() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 4, 346, 3));
        let mut layer = MambaLayerFlat::zeros(dims);

        for t in 0..dims.seq_len {
            let xdbl = layer.xdbl_mut(t);
            // Fill first dt_rank elements with sentinel, rest with -1.
            for (i, v) in xdbl.iter_mut().enumerate() {
                if i < dims.dt_rank {
                    *v = (t * 100 + i) as f32;
                } else {
                    *v = -1.0;
                }
            }
        }

        let mut dst = vec![0.0f32; dims.seq_len * dims.dt_rank];
        layer.copy_xdbl_dt_all(&mut dst);

        for t in 0..dims.seq_len {
            for i in 0..dims.dt_rank {
                let expected = (t * 100 + i) as f32;
                assert_eq!(
                    dst[t * dims.dt_rank + i],
                    expected,
                    "xdbl_dt_all mismatch at t={t}, i={i}",
                );
            }
        }
    }

    #[test]
    fn test_backbone_flat_allocation() {
        let dims = default_dims();
        let backbone = MambaBackboneFlat::zeros(dims);

        assert_eq!(
            backbone.input_proj_inputs.len(),
            dims.seq_len * dims.mamba_input_dim
        );
        assert_eq!(backbone.layers.len(), dims.n_layers);

        for layer in &backbone.layers {
            let expected_len = dims.seq_len * layer.offsets.step_stride;
            assert_eq!(layer.data.len(), expected_len);
            assert_eq!(layer.stride, TAPE_STRIDE);
            let ckpts = dims.seq_len.div_ceil(TAPE_STRIDE);
            assert_eq!(layer.ssm_ckpt.len(), ckpts * dims.d_inner * dims.d_state);
            assert_eq!(layer.conv_init.len(), dims.d_inner * dims.d_conv);
        }
    }

    /// The register rebuilt at a step is the forward's shift register at
    /// that step: the starting register's tail, then the rows in order.
    #[test]
    fn conv_window_is_the_shift_register() {
        let dims = MambaDims::new((8, 6, 2, 4, 1, 7, 5, 1));
        let (di, dc) = (dims.d_inner, dims.d_conv);
        let mut layer = MambaLayerFlat::zeros(dims);
        for (i, v) in layer.conv_init.iter_mut().enumerate() {
            *v = -(i as f32) - 1.0;
        }
        for t in 0..dims.seq_len {
            for (d, v) in layer.x_branch_mut(t).iter_mut().enumerate() {
                *v = (t * 100 + d) as f32;
            }
        }
        let mut reg = layer.conv_init.clone();
        let mut got = vec![0.0f32; di * dc];
        for t in 0..dims.seq_len {
            for d in 0..di {
                reg.copy_within(d * dc + 1..(d + 1) * dc, d * dc);
                reg[d * dc + dc - 1] = layer.x_branch(t)[d];
            }
            layer.conv_window(t, &mut got);
            assert_eq!(got, reg, "step {t}");
        }
    }
}
