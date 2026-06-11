//! Contiguous activation buffers for Mamba-3 SISO forward/backward.
//!
//! 28 fields per timestep. All intermediate values needed for backward
//! are saved in a single contiguous `Vec<f32>` per layer.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use super::dims::Mamba3Dims;

/// Memory layout descriptor — f32-element offsets for each saved field within one timestep.
#[derive(Debug, Clone, Copy)]
pub struct Mamba3FieldOffsets {
    pub residual: usize,     // [d_model]
    pub rms_val: usize,      // [1]
    pub post_norm: usize,    // [d_model]
    pub z: usize,            // [d_inner]
    pub x: usize,            // [d_inner]
    pub b_raw: usize,        // [ng * ds]
    pub c_raw: usize,        // [ng * ds]
    pub b_normed: usize,     // [ng * ds]
    pub c_normed: usize,     // [ng * ds]
    pub bcnorm_rms_b: usize, // [ng]
    pub bcnorm_rms_c: usize, // [ng]
    pub dd_dt_raw: usize,    // [nh]
    pub dd_a_raw: usize,     // [nh]
    pub trap_raw: usize,     // [nh]
    pub angles_raw: usize,   // [n_angles.max(1)]
    pub angle_cumsum: usize, // [n_angles.max(1)]
    pub alpha: usize,        // [nh]
    pub beta: usize,         // [nh]
    pub gamma: usize,        // [nh]
    pub dt_val: usize,       // [nh]
    pub a_val: usize,        // [nh]
    pub h_prev: usize,       // [nh * hd * ds]
    pub h_curr: usize,       // [nh * hd * ds]
    pub k_prev: usize,       // [nh * ds]
    pub v_prev: usize,       // [nh * hd]
    pub y: usize,            // [d_inner]
    /// UNUSED on CPU (backward recomputes the group rstd from `y`, matching
    /// forward). Kept in the layout for offset-table stability; the GPU
    /// mixed path saves its per-head rstd in a separate buffer.
    pub gated_rms_val: usize, // [1]
    pub gated: usize,        // [d_inner]
    pub step_stride: usize,  // total floats per timestep
}

impl Mamba3FieldOffsets {
    pub fn new(dims: &Mamba3Dims) -> Self {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ng = dims.ngroups;
        let na = dims.num_rope_angles;

        let mut off = 0usize;
        let residual = off;
        off += dm;
        let rms_val = off;
        off += 1;
        let post_norm = off;
        off += dm;
        let z = off;
        off += di;
        let x = off;
        off += di;
        let b_raw = off;
        off += ng * ds;
        let c_raw = off;
        off += ng * ds;
        let b_normed = off;
        off += ng * ds;
        let c_normed = off;
        off += ng * ds;
        let bcnorm_rms_b = off;
        off += ng;
        let bcnorm_rms_c = off;
        off += ng;
        let dd_dt_raw = off;
        off += nh;
        let dd_a_raw = off;
        off += nh;
        let trap_raw = off;
        off += nh;
        let angles_raw = off;
        off += na.max(1);
        let angle_cumsum = off;
        off += na.max(1);
        let alpha = off;
        off += nh;
        let beta = off;
        off += nh;
        let gamma = off;
        off += nh;
        let dt_val = off;
        off += nh;
        let a_val = off;
        off += nh;
        let h_prev = off;
        off += nh * hd * ds;
        let h_curr = off;
        off += nh * hd * ds;
        let k_prev = off;
        off += nh * ds;
        let v_prev = off;
        off += nh * hd;
        let y = off;
        off += di;
        let gated_rms_val = off;
        off += 1;
        let gated = off;
        off += di;

        Self {
            residual,
            rms_val,
            post_norm,
            z,
            x,
            b_raw,
            c_raw,
            b_normed,
            c_normed,
            bcnorm_rms_b,
            bcnorm_rms_c,
            dd_dt_raw,
            dd_a_raw,
            trap_raw,
            angles_raw,
            angle_cumsum,
            alpha,
            beta,
            gamma,
            dt_val,
            a_val,
            h_prev,
            h_curr,
            k_prev,
            v_prev,
            y,
            gated_rms_val,
            gated,
            step_stride: off,
        }
    }
}

/// Contiguous activation buffer for one Mamba-3 layer across all timesteps.
pub struct Mamba3LayerFlat {
    pub data: Vec<f32>,
    pub offsets: Mamba3FieldOffsets,
    pub dims: Mamba3Dims,
}

impl Mamba3LayerFlat {
    /// Allocate zeroed buffer for `seq_len` timesteps.
    pub fn zeros(dims: Mamba3Dims) -> Self {
        let offsets = Mamba3FieldOffsets::new(&dims);
        let total = dims.seq_len * offsets.step_stride;
        Self {
            data: vec![0.0; total],
            offsets,
            dims,
        }
    }

    /// Base offset for timestep `t`.
    #[inline(always)]
    pub fn base(&self, t: usize) -> usize {
        t * self.offsets.step_stride
    }

    // ── Read accessors ──

    pub fn z(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.z;
        &self.data[b..b + self.dims.d_inner]
    }
    pub fn x(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.x;
        &self.data[b..b + self.dims.d_inner]
    }
    pub fn b_normed(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.b_normed;
        &self.data[b..b + self.dims.ngroups * self.dims.d_state]
    }
    pub fn c_normed(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.c_normed;
        &self.data[b..b + self.dims.ngroups * self.dims.d_state]
    }
    pub fn y(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.y;
        &self.data[b..b + self.dims.d_inner]
    }
    pub fn h_prev(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.h_prev;
        let len = self.dims.nheads * self.dims.headdim * self.dims.d_state;
        &self.data[b..b + len]
    }
    pub fn k_prev(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.k_prev;
        &self.data[b..b + self.dims.nheads * self.dims.d_state]
    }
    pub fn v_prev(&self, t: usize) -> &[f32] {
        let b = self.base(t) + self.offsets.v_prev;
        &self.data[b..b + self.dims.nheads * self.dims.headdim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba3_siso::config::Mamba3Config;

    fn test_dims() -> Mamba3Dims {
        let cfg = Mamba3Config {
            d_model: 16,
            d_state: 8,
            expand: 2,
            headdim: 4,
            ngroups: 1,
            n_layers: 1,
            rope_fraction: 0.5,
            a_floor: 1e-4,
            is_outproj_norm: false,
        };
        Mamba3Dims::from_config(&cfg, 33)
    }

    #[test]
    fn test_offsets_monotonic() {
        let dims = test_dims();
        let o = Mamba3FieldOffsets::new(&dims);
        let fields = [
            o.residual,
            o.rms_val,
            o.post_norm,
            o.z,
            o.x,
            o.b_raw,
            o.c_raw,
            o.b_normed,
            o.c_normed,
            o.bcnorm_rms_b,
            o.bcnorm_rms_c,
            o.dd_dt_raw,
            o.dd_a_raw,
            o.trap_raw,
            o.angles_raw,
            o.angle_cumsum,
            o.alpha,
            o.beta,
            o.gamma,
            o.dt_val,
            o.a_val,
            o.h_prev,
            o.h_curr,
            o.k_prev,
            o.v_prev,
            o.y,
            o.gated_rms_val,
            o.gated,
        ];
        for i in 1..fields.len() {
            assert!(fields[i] > fields[i - 1], "offset[{i}] not monotonic");
        }
    }

    #[test]
    fn test_flat_allocation() {
        let dims = test_dims();
        let flat = Mamba3LayerFlat::zeros(dims);
        assert_eq!(flat.data.len(), dims.seq_len * flat.offsets.step_stride);
    }
}
