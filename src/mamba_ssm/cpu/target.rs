//! Target network Mamba forward pass (no backward, no activation saves).
//!
//! Used for target network inference (no gradient tracking): T=1 cold-state forward, or
//! T=seq_len burn-in forward with state carry.

use super::weights::TrainMambaWeights;
use crate::ops::blas::matvec_forward;
use crate::ops::fast_math::fast_exp_scalar;

const RMS_NORM_EPS: f32 = 1e-5;

/// Scratch buffers for single-step (T=1) target Mamba forward.
pub struct MambaTargetScratch {
    /// In-proj output (x + gate concatenated) `[2 * d_inner]`.
    pub proj: Vec<f32>,
    /// x-branch after split `[d_inner]`.
    pub x: Vec<f32>,
    /// Gate branch after SiLU `[d_inner]`.
    pub gate_silu: Vec<f32>,
    /// RMSNorm intermediate `[d_model]`.
    pub norm_buf: Vec<f32>,
    /// Saved residual for skip connection `[d_model]`.
    pub residual: Vec<f32>,
    /// x_proj output (delta_raw, B, C concatenated) `[dt_rank + 2*d_state]`.
    pub xdbl: Vec<f32>,
    /// Discretized delta after softplus `[d_inner]`.
    pub delta: Vec<f32>,
    /// SSM output `[d_inner]`.
    pub y: Vec<f32>,
    /// SSM hidden state (zero-initialized per call) `[d_inner * d_state]`.
    pub h: Vec<f32>,
}

impl MambaTargetScratch {
    /// Allocate scratch buffers for single-step target forward.
    ///
    /// - `d_model`: model dimension
    /// - `d_inner`: expanded dimension (`expand * d_model`)
    /// - `d_state`: SSM state dimension
    /// - `dt_rank`: delta projection bottleneck rank
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, dt_rank: usize) -> Self {
        Self {
            proj: vec![0.0; 2 * d_inner],
            x: vec![0.0; d_inner],
            gate_silu: vec![0.0; d_inner],
            norm_buf: vec![0.0; d_model],
            residual: vec![0.0; d_model],
            xdbl: vec![0.0; dt_rank + 2 * d_state],
            delta: vec![0.0; d_inner],
            y: vec![0.0; d_inner],
            h: vec![0.0; d_inner * d_state],
        }
    }
}

/// Single-step (T=1) target Mamba forward with cold (zero) state.
///
/// Processes one timestep through all Mamba layers. No conv1d shift
/// register is needed (T=1 means conv output = bias + x * weight[dc-1]).
pub fn forward_mamba_target_step(
    output: &mut [f32], // [d_model] output
    input: &[f32],      // [mamba_input_dim] input
    w: &TrainMambaWeights,
    scratch: &mut MambaTargetScratch,
    dims: (usize, usize, usize, usize, usize, usize), // (dm, di, ds, dc, dr, mid)
) {
    let (dm, di, ds, dc, dr, mid) = dims;
    let xdbl_dim = dr + 2 * ds;

    // Input projection
    matvec_forward(
        &mut output[..dm],
        input,
        &w.input_proj_w,
        Some(&w.input_proj_b),
        mid,
        dm,
    );

    for lw in &w.layers {
        // Save residual
        scratch.residual[..dm].copy_from_slice(&output[..dm]);

        // RmsNorm
        let mean_sq: f32 = output[..dm].iter().map(|v| v * v).sum::<f32>() / dm as f32;
        let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
        for ((nb, &o), &nw) in scratch.norm_buf[..dm]
            .iter_mut()
            .zip(output[..dm].iter())
            .zip(lw.norm_weight[..dm].iter())
        {
            *nb = o * inv_rms * nw;
        }

        // in_proj
        matvec_forward(
            &mut scratch.proj[..2 * di],
            &scratch.norm_buf[..dm],
            &lw.in_proj_w,
            None,
            dm,
            2 * di,
        );

        // Split + gate SiLU
        scratch.x[..di].copy_from_slice(&scratch.proj[..di]);
        for (gs, &v) in scratch.gate_silu[..di]
            .iter_mut()
            .zip(scratch.proj[di..2 * di].iter())
        {
            let sig = 1.0 / (1.0 + fast_exp_scalar(-v));
            *gs = v * sig;
        }

        // Conv1d (T=1 cold state: just bias + x * weight[dc-1])
        for d in 0..di {
            let val = lw.conv1d_bias[d] + scratch.x[d] * lw.conv1d_weight[d * dc + dc - 1];
            let sig = 1.0 / (1.0 + fast_exp_scalar(-val));
            scratch.x[d] = val * sig; // fused SiLU
        }

        // x_proj
        matvec_forward(
            &mut scratch.xdbl[..xdbl_dim],
            &scratch.x[..di],
            &lw.x_proj_w,
            None,
            di,
            xdbl_dim,
        );

        // dt_proj + softplus
        matvec_forward(
            &mut scratch.delta[..di],
            &scratch.xdbl[..dr],
            &lw.dt_proj_w,
            Some(&lw.dt_proj_b),
            dr,
            di,
        );
        for d in 0..di {
            let raw = scratch.delta[d];
            scratch.delta[d] = if raw > 20.0 {
                raw
            } else {
                (1.0 + fast_exp_scalar(raw)).ln()
            };
        }

        // SSM (zero initial state)
        scratch.h[..di * ds].fill(0.0);
        let b_offset = dr;
        let c_offset = dr + ds;
        for d in 0..di {
            let delta_d = scratch.delta[d];
            let u_d = scratch.x[d];
            let delta_u_d = delta_d * u_d;
            let mut y_d = 0.0_f32;

            for n in 0..ds {
                let idx = d * ds + n;
                let a_dn = -fast_exp_scalar(lw.a_log[idx]);
                let da = fast_exp_scalar(delta_d * a_dn);
                let b_n = scratch.xdbl[b_offset + n];
                let c_n = scratch.xdbl[c_offset + n];

                scratch.h[idx] = da * scratch.h[idx] + delta_u_d * b_n;
                y_d += scratch.h[idx] * c_n;
            }

            y_d += lw.d_param[d] * u_d;
            scratch.y[d] = y_d;
        }

        // Gating
        for d in 0..di {
            scratch.y[d] *= scratch.gate_silu[d];
        }

        // out_proj
        matvec_forward(
            &mut output[..dm],
            &scratch.y[..di],
            &lw.out_proj_w,
            None,
            di,
            dm,
        );

        // Residual
        for (o, &r) in output[..dm].iter_mut().zip(scratch.residual[..dm].iter()) {
            *o += r;
        }
    }

    // Final RmsNorm (norm_f)
    let mean_sq: f32 = output[..dm].iter().map(|v| v * v).sum::<f32>() / dm as f32;
    let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
    for (o, &nfw) in output[..dm].iter_mut().zip(w.norm_f_weight[..dm].iter()) {
        *o *= inv_rms * nfw;
    }
}

/// Scratch buffers for multi-step target Mamba forward (burn-in).
pub struct MambaTargetSeqScratch {
    /// Current timestep output `[d_model]`.
    pub temporal: Vec<f32>,
    /// In-proj output (x + gate concatenated) `[2 * d_inner]`.
    pub proj: Vec<f32>,
    /// x-branch after split `[d_inner]`.
    pub x: Vec<f32>,
    /// Gate branch after SiLU `[d_inner]`.
    pub gate_silu: Vec<f32>,
    /// RMSNorm intermediate `[d_model]`.
    pub norm_buf: Vec<f32>,
    /// x_proj output (delta_raw, B, C concatenated) `[dt_rank + 2*d_state]`.
    pub xdbl: Vec<f32>,
    /// Discretized delta after softplus `[d_inner]`.
    pub delta: Vec<f32>,
    /// SSM output `[d_inner]`.
    pub y: Vec<f32>,
    /// Saved residual for skip connection `[d_model]`.
    pub residual: Vec<f32>,
    /// Conv1d shift register state across layers `[n_layers * d_inner * d_conv]`.
    pub conv_states: Vec<f32>,
    /// SSM hidden state across layers `[n_layers * d_inner * d_state]`.
    pub ssm_states: Vec<f32>,
    /// Sequence length (number of timesteps to process).
    pub seq_len: usize,
}

impl MambaTargetSeqScratch {
    /// Allocate scratch buffers for multi-step target forward (burn-in).
    ///
    /// - `dm`: d_model
    /// - `di`: d_inner
    /// - `ds`: d_state
    /// - `dc`: d_conv
    /// - `dr`: dt_rank
    /// - `nl`: number of layers
    /// - `seq_len`: burn-in sequence length
    pub fn new(
        dm: usize,
        di: usize,
        ds: usize,
        dc: usize,
        dr: usize,
        nl: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            temporal: vec![0.0; dm],
            proj: vec![0.0; 2 * di],
            x: vec![0.0; di],
            gate_silu: vec![0.0; di],
            norm_buf: vec![0.0; dm],
            xdbl: vec![0.0; dr + 2 * ds],
            delta: vec![0.0; di],
            y: vec![0.0; di],
            residual: vec![0.0; dm],
            conv_states: vec![0.0; nl * di * dc],
            ssm_states: vec![0.0; nl * di * ds],
            seq_len,
        }
    }

    /// Reset conv and SSM states for a new sample.
    pub fn reset_states(&mut self) {
        self.conv_states.fill(0.0);
        self.ssm_states.fill(0.0);
    }
}

/// Multi-step target forward with burn-in (standard burn-in approach for recurrent models).
///
/// Processes T = seq_len timesteps with state carry.
/// Input is pre-batched input_proj output: `[seq_len * d_model]`.
/// Writes the final timestep output into `output[d_model]`.
pub fn forward_mamba_target_sequence(
    output: &mut [f32],  // [d_model] output (last timestep)
    ip_out_flat: &[f32], // [seq_len * d_model] pre-batched input_proj output
    w: &TrainMambaWeights,
    scratch: &mut MambaTargetSeqScratch,
    dims: (usize, usize, usize, usize, usize, usize), // (dm, di, ds, dc, dr, seq_len)
) {
    let (dm, di, ds, dc, dr, seq_len) = dims;
    let xdbl_dim = dr + 2 * ds;
    let conv_per_layer = di * dc;
    let ssm_per_layer = di * ds;

    for t in 0..seq_len {
        // Load this timestep's input_proj output
        scratch.temporal[..dm].copy_from_slice(&ip_out_flat[t * dm..(t + 1) * dm]);

        for (layer_idx, lw) in w.layers.iter().enumerate() {
            let conv_start = layer_idx * conv_per_layer;
            let ssm_start = layer_idx * ssm_per_layer;

            // Save residual
            scratch.residual[..dm].copy_from_slice(&scratch.temporal[..dm]);

            // RmsNorm
            let mean_sq: f32 =
                scratch.temporal[..dm].iter().map(|v| v * v).sum::<f32>() / dm as f32;
            let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
            for ((nb, &t), &nw) in scratch.norm_buf[..dm]
                .iter_mut()
                .zip(scratch.temporal[..dm].iter())
                .zip(lw.norm_weight[..dm].iter())
            {
                *nb = t * inv_rms * nw;
            }

            // in_proj
            matvec_forward(
                &mut scratch.proj[..2 * di],
                &scratch.norm_buf[..dm],
                &lw.in_proj_w,
                None,
                dm,
                2 * di,
            );

            // Split + gate SiLU
            scratch.x[..di].copy_from_slice(&scratch.proj[..di]);
            for d in 0..di {
                let v = scratch.proj[di + d];
                let sig = 1.0 / (1.0 + fast_exp_scalar(-v));
                scratch.gate_silu[d] = v * sig;
            }

            // Conv1d with state carry
            {
                let cs = &mut scratch.conv_states[conv_start..conv_start + conv_per_layer];
                for d in 0..di {
                    let base = d * dc;
                    // Shift + insert
                    for k in 0..dc - 1 {
                        cs[base + k] = cs[base + k + 1];
                    }
                    cs[base + dc - 1] = scratch.x[d];
                    // Dot product
                    let mut val = lw.conv1d_bias[d];
                    for k in 0..dc {
                        val += cs[base + k] * lw.conv1d_weight[base + k];
                    }
                    let sig = 1.0 / (1.0 + fast_exp_scalar(-val));
                    scratch.x[d] = val * sig;
                }
            }

            // x_proj
            matvec_forward(
                &mut scratch.xdbl[..xdbl_dim],
                &scratch.x[..di],
                &lw.x_proj_w,
                None,
                di,
                xdbl_dim,
            );

            // dt_proj + softplus
            matvec_forward(
                &mut scratch.delta[..di],
                &scratch.xdbl[..dr],
                &lw.dt_proj_w,
                Some(&lw.dt_proj_b),
                dr,
                di,
            );
            for d in 0..di {
                let raw = scratch.delta[d];
                scratch.delta[d] = if raw > 20.0 {
                    raw
                } else {
                    (1.0 + fast_exp_scalar(raw)).ln()
                };
            }

            // SSM
            let b_offset = dr;
            let c_offset = dr + ds;
            let h = &mut scratch.ssm_states[ssm_start..ssm_start + ssm_per_layer];
            for d in 0..di {
                let delta_d = scratch.delta[d];
                let u_d = scratch.x[d];
                let delta_u_d = delta_d * u_d;
                let mut y_d = 0.0_f32;

                for n in 0..ds {
                    let idx = d * ds + n;
                    let a_dn = -fast_exp_scalar(lw.a_log[idx]);
                    let da = fast_exp_scalar(delta_d * a_dn);
                    let b_n = scratch.xdbl[b_offset + n];
                    let c_n = scratch.xdbl[c_offset + n];

                    h[idx] = da * h[idx] + delta_u_d * b_n;
                    y_d += h[idx] * c_n;
                }

                y_d += lw.d_param[d] * u_d;
                scratch.y[d] = y_d;
            }

            // Gating
            for d in 0..di {
                scratch.y[d] *= scratch.gate_silu[d];
            }

            // out_proj
            matvec_forward(
                &mut scratch.temporal[..dm],
                &scratch.y[..di],
                &lw.out_proj_w,
                None,
                di,
                dm,
            );

            // Residual
            for (t, &r) in scratch.temporal[..dm]
                .iter_mut()
                .zip(scratch.residual[..dm].iter())
            {
                *t += r;
            }
        }

        // norm_f after all layers (per timestep)
        let mean_sq: f32 = scratch.temporal[..dm].iter().map(|v| v * v).sum::<f32>() / dm as f32;
        let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
        for (t, &nfw) in scratch.temporal[..dm]
            .iter_mut()
            .zip(w.norm_f_weight[..dm].iter())
        {
            *t *= inv_rms * nfw;
        }
    }

    // Output = last timestep
    output[..dm].copy_from_slice(&scratch.temporal[..dm]);
}
