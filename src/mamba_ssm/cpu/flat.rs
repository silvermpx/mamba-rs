//! O5 contiguous activation buffers for Mamba forward/backward.
//!
//! Replaces `Vec<MambaStepSaved>` (18 Vecs per step, ~600 heap allocations per
//! sample) with a single flat `Vec<f32>` per layer. All fields for all timesteps
//! are packed contiguously in memory: `data[t * step_stride + field_offset .. +len]`.
//!
//! Layout per timestep (17 fields, no `pre_norm` -- was duplicate of `residual`):
//!
//! | # | Field           | Size                         |
//! |---|-----------------|------------------------------|
//! | 0 | residual        | d_model                      |
//! | 1 | rms_val         | 1                            |
//! | 2 | post_norm       | d_model                      |
//! | 3 | x_branch        | d_inner                      |
//! | 4 | conv_state      | d_inner * d_conv             |
//! | 5 | post_conv       | d_inner                      |
//! | 6 | u               | d_inner                      |
//! | 7 | xdbl            | xdbl_dim (dt_rank+2*d_state) |
//! | 8 | delta_raw       | d_inner                      |
//! | 9 | delta           | d_inner                      |
//! |10 | h_prev          | d_inner * d_state            |
//! |11 | h_curr          | d_inner * d_state            |
//! |12 | da_exp          | d_inner * d_state            |
//! |13 | y               | d_inner                      |
//! |14 | gate_pre_silu   | d_inner                      |
//! |15 | gate_post_silu  | d_inner                      |
//! |16 | gated           | d_inner                      |

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
    pub conv_state: usize,
    pub post_conv: usize,
    pub u: usize,
    pub xdbl: usize,
    pub delta_raw: usize,
    pub delta: usize,
    pub h_prev: usize,
    pub h_curr: usize,
    pub da_exp: usize,
    pub y: usize,
    pub gate_pre_silu: usize,
    pub gate_post_silu: usize,
    pub gated: usize,
    /// Total f32 elements per timestep.
    pub step_stride: usize,
}

impl FieldOffsets {
    /// Compute field offsets from Mamba dimensions.
    pub fn new(dims: &MambaDims) -> Self {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let dc = dims.d_conv;
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

        let conv_state = off;
        off += di * dc;

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

        let h_prev = off;
        off += di * ds;

        let h_curr = off;
        off += di * ds;

        let da_exp = off;
        off += di * ds;

        let y = off;
        off += di;

        let gate_pre_silu = off;
        off += di;

        let gate_post_silu = off;
        off += di;

        let gated = off;
        off += di;

        Self {
            residual,
            rms_val,
            post_norm,
            x_branch,
            conv_state,
            post_conv,
            u,
            xdbl: xdbl_off,
            delta_raw,
            delta,
            h_prev,
            h_curr,
            da_exp,
            y,
            gate_pre_silu,
            gate_post_silu,
            gated,
            step_stride: off,
        }
    }
}

// ---------------------------------------------------------------------------
// MambaLayerFlat — one layer, all timesteps in a single Vec<f32>
// ---------------------------------------------------------------------------

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
}

impl MambaLayerFlat {
    /// Allocate a zero-filled flat buffer for `dims.seq_len` timesteps.
    pub fn zeros(dims: MambaDims) -> Self {
        let offsets = FieldOffsets::new(&dims);
        let total = dims.seq_len * offsets.step_stride;
        Self {
            data: vec![0.0; total],
            offsets,
            dims,
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

    /// `conv_state[t]`: `&[d_inner * d_conv]`.
    #[inline]
    pub fn conv_state(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.conv_state;
        &self.data[b..b + self.dims.d_inner * self.dims.d_conv]
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

    /// `h_prev[t]`: `&[d_inner * d_state]`.
    #[inline]
    pub fn h_prev(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.h_prev;
        &self.data[b..b + self.dims.d_inner * self.dims.d_state]
    }

    /// `h_curr[t]`: `&[d_inner * d_state]`.
    #[inline]
    pub fn h_curr(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.h_curr;
        &self.data[b..b + self.dims.d_inner * self.dims.d_state]
    }

    /// `da_exp[t]`: `&[d_inner * d_state]`.
    #[inline]
    pub fn da_exp(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.da_exp;
        &self.data[b..b + self.dims.d_inner * self.dims.d_state]
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

    /// `conv_state_mut[t]`: `&mut [d_inner * d_conv]`.
    #[inline]
    pub fn conv_state_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.conv_state;
        let len = self.dims.d_inner * self.dims.d_conv;
        &mut self.data[b..b + len]
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

    /// `h_prev_mut[t]`: `&mut [d_inner * d_state]`.
    #[inline]
    pub fn h_prev_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.h_prev;
        let len = self.dims.d_inner * self.dims.d_state;
        &mut self.data[b..b + len]
    }

    /// `h_curr_mut[t]`: `&mut [d_inner * d_state]`.
    #[inline]
    pub fn h_curr_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.h_curr;
        let len = self.dims.d_inner * self.dims.d_state;
        &mut self.data[b..b + len]
    }

    /// `da_exp_mut[t]`: `&mut [d_inner * d_state]`.
    #[inline]
    pub fn da_exp_mut(&mut self, t: usize) -> &mut [f32] {
        let b = self.base(t) + self.offsets.da_exp;
        let len = self.dims.d_inner * self.dims.d_state;
        &mut self.data[b..b + len]
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
    /// Input projection outputs across all timesteps: `[seq_len * d_model]`.
    pub input_proj_outputs: Vec<f32>,
    /// Per-layer flat activation buffers.
    pub layers: Vec<MambaLayerFlat>,
    /// Saved pre-norm_f input for backward: `[seq_len * d_model]`.
    pub norm_f_input: Vec<f32>,
    /// Saved RMS values per timestep for norm_f backward: `[seq_len]`.
    pub norm_f_rms: Vec<f32>,
}

impl MambaBackboneFlat {
    /// Allocate all zero-filled flat buffers.
    pub fn zeros(dims: MambaDims) -> Self {
        Self {
            input_proj_inputs: vec![0.0; dims.seq_len * dims.mamba_input_dim],
            input_proj_outputs: vec![0.0; dims.seq_len * dims.d_model],
            layers: (0..dims.n_layers)
                .map(|_| MambaLayerFlat::zeros(dims))
                .collect(),
            norm_f_input: vec![0.0; dims.seq_len * dims.d_model],
            norm_f_rms: vec![0.0; dims.seq_len],
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
        let ds = dims.d_state;
        let dc = dims.d_conv;
        let xd = dims.xdbl_dim;

        let fields: Vec<(usize, usize)> = vec![
            (o.residual, dm),
            (o.rms_val, 1),
            (o.post_norm, dm),
            (o.x_branch, di),
            (o.conv_state, di * dc),
            (o.post_conv, di),
            (o.u, di),
            (o.xdbl, xd),
            (o.delta_raw, di),
            (o.delta, di),
            (o.h_prev, di * ds),
            (o.h_curr, di * ds),
            (o.da_exp, di * ds),
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
        // 128+1+128+256+1024+256+256+40+256+256+4096+4096+4096+256+256+256+256 = 15913
        assert_eq!(o.step_stride, 15913);
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

            // conv_state
            layer.conv_state_mut(t).iter_mut().for_each(|v| *v = 4.0);
            assert!(layer.conv_state(t).iter().all(|&v| v == 4.0));
            assert_eq!(layer.conv_state(t).len(), dims.d_inner * dims.d_conv);

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

            // h_prev
            layer.h_prev_mut(t).iter_mut().for_each(|v| *v = 10.0);
            assert!(layer.h_prev(t).iter().all(|&v| v == 10.0));
            assert_eq!(layer.h_prev(t).len(), dims.d_inner * dims.d_state);

            // h_curr
            layer.h_curr_mut(t).iter_mut().for_each(|v| *v = 11.0);
            assert!(layer.h_curr(t).iter().all(|&v| v == 11.0));
            assert_eq!(layer.h_curr(t).len(), dims.d_inner * dims.d_state);

            // da_exp
            layer.da_exp_mut(t).iter_mut().for_each(|v| *v = 12.0);
            assert!(layer.da_exp(t).iter().all(|&v| v == 12.0));
            assert_eq!(layer.da_exp(t).len(), dims.d_inner * dims.d_state);

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
        assert_eq!(
            backbone.input_proj_outputs.len(),
            dims.seq_len * dims.d_model
        );
        assert_eq!(backbone.layers.len(), dims.n_layers);

        for layer in &backbone.layers {
            let expected_len = dims.seq_len * layer.offsets.step_stride;
            assert_eq!(layer.data.len(), expected_len);
        }
    }
}
