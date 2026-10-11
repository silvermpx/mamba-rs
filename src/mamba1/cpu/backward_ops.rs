//! Backward operation primitives for Mamba training.
//!
//! Conv1d step backward and RMSNorm backward.

/// Dimensions for conv1d backward.
pub struct Conv1dDims {
    /// Expanded inner dimension (`expand * d_model`).
    pub d_inner: usize,
    /// Convolution kernel width.
    pub d_conv: usize,
}

/// Backward pass for a single conv1d depthwise step.
///
/// Given upstream gradient `d_out[d_inner]` (after SiLU backward),
/// computes gradients for conv weights, bias, and the input x_branch.
///
/// The conv1d shift register state at this timestep is used to
/// compute weight gradients. The input gradient flows back through
/// the last position of the filter only (current x_branch feeds
/// into position `d_conv - 1`).
pub fn backward_conv1d_step(
    d_x_branch: &mut [f32],    // [d_inner] output: gradient w.r.t. x_branch input
    d_conv_weight: &mut [f32], // [d_inner * d_conv] accumulated weight gradient
    d_conv_bias: &mut [f32],   // [d_inner] accumulated bias gradient
    d_conv_out: &[f32],        // [d_inner] upstream gradient (post SiLU backward)
    conv_state: &[f32],        // [d_inner * d_conv] saved conv state at this timestep
    conv_weight: &[f32],       // [d_inner * d_conv] conv weights
    dims: Conv1dDims,
) {
    let di = dims.d_inner;
    let dc = dims.d_conv;

    for d in 0..di {
        let base = d * dc;

        // Bias gradient: db[d] += d_conv_out[d]
        d_conv_bias[d] += d_conv_out[d];

        // Weight gradient: dw[d, k] += d_conv_out[d] * state[d, k]
        for k in 0..dc {
            d_conv_weight[base + k] += d_conv_out[d] * conv_state[base + k];
        }

        // Input gradient: x_branch entered at position dc-1 of the state.
        // dx_branch[d] = d_conv_out[d] * weight[d, dc-1]
        d_x_branch[d] = d_conv_out[d] * conv_weight[base + dc - 1];
    }
}

/// Batched backward pass for RMSNorm.
///
/// Processes `batch` samples of dimension `dim` in flat layout.
///
/// Given `y[b,i] = x[b,i] * inv_rms[b] * scale[i]`, computes:
/// - `dx[b,i]` (overwritten)
/// - `d_scale[i] += ...` (accumulated across batch)
///
/// Arguments:
/// - `scale_and_rms`: tuple of `(scale: &[dim], rms_vals: &[batch])`
pub fn backward_rms_norm(
    dx: &mut [f32],                  // [batch * dim] output gradient
    d_scale: &mut [f32],             // [dim] accumulated scale gradient
    dy: &[f32],                      // [batch * dim] upstream gradient
    x: &[f32],                       // [batch * dim] saved input
    scale_and_rms: (&[f32], &[f32]), // (scale[dim], rms_vals[batch])
    batch: usize,
    dim: usize,
) {
    let (scale, rms_vals) = scale_and_rms;
    let dim_f = dim as f32;

    for (b, &rms_b) in rms_vals.iter().enumerate().take(batch) {
        let off = b * dim;
        let inv_rms = 1.0 / rms_b;
        let inv_rms2 = inv_rms * inv_rms;

        // Dot product: sum(dy * scale * x) * inv_rms^2
        let mut dot = 0.0_f32;
        for i in 0..dim {
            dot += dy[off + i] * scale[i] * x[off + i];
        }
        dot *= inv_rms2;

        for i in 0..dim {
            // d_scale: accumulated across batch
            d_scale[i] += dy[off + i] * x[off + i] * inv_rms;
            // dx: chain rule through normalization
            dx[off + i] = inv_rms * (scale[i] * dy[off + i] - x[off + i] * dot / dim_f);
        }
    }
}
