//! Trainable weight storage for Mamba backbone.
//!
//! Same layout as inference `MambaWeights` but all fields are owned `Vec<f32>`
//! for in-place gradient accumulation during backward pass.

use crate::ops::dims::MambaDims;

/// Trainable weights for a single Mamba layer.
///
/// Each field can be used as both weight storage and gradient accumulator.
/// The forward pass reads weights; the backward pass accumulates gradients
/// into a separate instance with the same layout.
#[derive(Clone)]
pub struct TrainMambaLayerWeights {
    /// RMSNorm scale `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// Input projection `[d_model, 2 * d_inner]`.
    pub in_proj_w: Vec<f32>,
    /// Depthwise conv1d filter `[d_inner * d_conv]`.
    pub conv1d_weight: Vec<f32>,
    /// Depthwise conv1d bias `[d_inner]`.
    pub conv1d_bias: Vec<f32>,
    /// x-to-(delta, B, C) projection `[d_inner, dt_rank + 2*d_state]`.
    pub x_proj_w: Vec<f32>,
    /// Delta bottleneck projection `[dt_rank, d_inner]`.
    pub dt_proj_w: Vec<f32>,
    /// Delta projection bias `[d_inner]`.
    pub dt_proj_b: Vec<f32>,
    /// Log-space SSM decay `[d_inner * d_state]`.
    pub a_log: Vec<f32>,
    /// SSM skip/D parameter `[d_inner]`.
    pub d_param: Vec<f32>,
    /// Output projection `[d_inner, d_model]`.
    pub out_proj_w: Vec<f32>,
}

/// Trainable weights for the full Mamba backbone.
#[derive(Clone)]
pub struct TrainMambaWeights {
    /// Input projection weight `[mamba_input_dim, d_model]`.
    pub input_proj_w: Vec<f32>,
    /// Input projection bias `[d_model]`.
    pub input_proj_b: Vec<f32>,
    /// Per-layer weights (one per Mamba layer).
    pub layers: Vec<TrainMambaLayerWeights>,
    /// Final RMSNorm scale `[d_model]`.
    pub norm_f_weight: Vec<f32>,
}

impl TrainMambaWeights {
    /// Allocate zeroed weights matching the given dimensions.
    pub fn zeros_from_dims(dims: &MambaDims) -> Self {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let dc = dims.d_conv;
        let dr = dims.dt_rank;
        let xdbl = dims.xdbl_dim;
        let mid = dims.mamba_input_dim;

        Self {
            input_proj_w: vec![0.0; mid * dm],
            input_proj_b: vec![0.0; dm],
            layers: (0..dims.n_layers)
                .map(|_| TrainMambaLayerWeights {
                    norm_weight: vec![0.0; dm],
                    in_proj_w: vec![0.0; dm * 2 * di],
                    conv1d_weight: vec![0.0; di * dc],
                    conv1d_bias: vec![0.0; di],
                    x_proj_w: vec![0.0; di * xdbl],
                    dt_proj_w: vec![0.0; dr * di],
                    dt_proj_b: vec![0.0; di],
                    a_log: vec![0.0; di * ds],
                    d_param: vec![0.0; di],
                    out_proj_w: vec![0.0; di * dm],
                })
                .collect(),
            norm_f_weight: vec![0.0; dm],
        }
    }

    /// Zero all weight/gradient values in-place.
    pub fn zero(&mut self) {
        self.input_proj_w.fill(0.0);
        self.input_proj_b.fill(0.0);
        for l in &mut self.layers {
            l.norm_weight.fill(0.0);
            l.in_proj_w.fill(0.0);
            l.conv1d_weight.fill(0.0);
            l.conv1d_bias.fill(0.0);
            l.x_proj_w.fill(0.0);
            l.dt_proj_w.fill(0.0);
            l.dt_proj_b.fill(0.0);
            l.a_log.fill(0.0);
            l.d_param.fill(0.0);
            l.out_proj_w.fill(0.0);
        }
        self.norm_f_weight.fill(0.0);
    }

    /// Accumulate another set of weights/gradients into this one.
    pub fn add_inplace(&mut self, other: &Self) {
        add_vecs(&mut self.input_proj_w, &other.input_proj_w);
        add_vecs(&mut self.input_proj_b, &other.input_proj_b);
        for (sl, ol) in self.layers.iter_mut().zip(other.layers.iter()) {
            add_vecs(&mut sl.norm_weight, &ol.norm_weight);
            add_vecs(&mut sl.in_proj_w, &ol.in_proj_w);
            add_vecs(&mut sl.conv1d_weight, &ol.conv1d_weight);
            add_vecs(&mut sl.conv1d_bias, &ol.conv1d_bias);
            add_vecs(&mut sl.x_proj_w, &ol.x_proj_w);
            add_vecs(&mut sl.dt_proj_w, &ol.dt_proj_w);
            add_vecs(&mut sl.dt_proj_b, &ol.dt_proj_b);
            add_vecs(&mut sl.a_log, &ol.a_log);
            add_vecs(&mut sl.d_param, &ol.d_param);
            add_vecs(&mut sl.out_proj_w, &ol.out_proj_w);
        }
        add_vecs(&mut self.norm_f_weight, &other.norm_f_weight);
    }
}

/// Element-wise `a[i] += b[i]`.
#[inline]
fn add_vecs(a: &mut [f32], b: &[f32]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}
