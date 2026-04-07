//! O2+O3+O4+O5 optimized Mamba forward and backward passes using flat buffers.
//!
//! Four optimization layers applied to the Mamba layer:
//!
//! - **O5**: Uses [`MambaLayerFlat`] (single contiguous `[T*stride]` buffer) instead of
//!   `Vec<Vec<f32>>` (18 heap allocations per timestep).
//! - **O3**: Pre-computed `a_neg[idx] = -exp(a_log[idx])` passed in, avoiding
//!   `d_inner * d_state` transcendentals per timestep.
//! - **O4**: Fused conv1d depthwise dot product + SiLU into a single loop
//!   (eliminates separate SiLU pass over `d_inner` elements).
//! - **O2**: Batched SGEMM for in_proj/out_proj across all T timesteps.
//!
//! Input/output is `&mut [f32]` flat `[T * d_model]` instead of `&mut [Vec<f32>]`.
//! The `pre_norm` field is removed (was duplicate of `residual`).
//!
//! ## Module structure
//!
//! - Batched SGEMM: `forward_mamba_layer_batched` / `backward_mamba_layer_batched`
//! - Full backbone: `forward_mamba_backbone_batched` / `backward_mamba_backbone_batched`

use super::backward_ops::{Conv1dDims, backward_conv1d_step, backward_rms_norm};
use super::flat::{MambaBackboneFlat, MambaLayerFlat};
use super::scratch::BackwardPhaseScratch;
use super::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use crate::ops::blas::sgemm_backward;
use crate::ops::dims::MambaDims;
use crate::ops::fast_math::fast_exp_scalar;

pub fn backward_mamba_layer_batched(
    d_temporal_flat: &mut [f32],
    d_layer: &mut TrainMambaLayerWeights,
    acts: &MambaLayerFlat,
    w: &TrainMambaLayerWeights,
    a_neg: &[f32],
    scratch: &mut BackwardPhaseScratch,
    dims: &MambaDims,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let seq_len = dims.seq_len;

    // ===================================================================
    // Phase B1: Batch out_proj backward
    //   sgemm_backward(d_gated_flat, d_out_proj_w, None,
    //                  d_temporal_flat, gated_all, out_proj_w,
    //                  (T, d_inner, d_model))
    // ===================================================================
    acts.copy_gated_all(&mut scratch.gated_buf);
    sgemm_backward(
        &mut scratch.d_gated_flat,
        &mut d_layer.out_proj_w,
        None,
        d_temporal_flat,
        &scratch.gated_buf,
        &w.out_proj_w,
        (seq_len, di, dm),
    );

    // ===================================================================
    // Phase B2: Batch gating backward (element-wise, all T at once)
    //   gated = y * SiLU(gate)
    //   d_y = d_gated * gate_post_silu
    //   d_gate = d_gated * y * silu_grad(gate_pre_silu)
    // ===================================================================
    for t in 0..seq_len {
        let t_off = t * di;
        let gate_post_silu = acts.gate_post_silu(t);
        let gate_pre_silu = acts.gate_pre_silu(t);
        let y_vals = acts.y(t);
        for d in 0..di {
            let dg = scratch.d_gated_flat[t_off + d];
            scratch.d_y_flat[t_off + d] = dg * gate_post_silu[d];
            let x = gate_pre_silu[d];
            let sigma = 1.0 / (1.0 + fast_exp_scalar(-x));
            let silu_grad = sigma * (1.0 + x * (1.0 - sigma));
            scratch.d_gate_flat[t_off + d] = dg * y_vals[d] * silu_grad;
        }
    }

    // ===================================================================
    // Phase B3: Sequential SSM BPTT (reverse)
    // ===================================================================
    scratch.d_h.fill(0.0);
    scratch.d_conv_carry.fill(0.0);

    let b_offset = dt_rank;
    let c_offset = dt_rank + ds;

    for t in (0..seq_len).rev() {
        let t_off_di = t * di;
        let t_off_xdbl = t * xdbl_dim;

        // Zero per-timestep gradient slices
        scratch.d_delta_flat[t_off_di..t_off_di + di].fill(0.0);
        scratch.d_u_flat[t_off_di..t_off_di + di].fill(0.0);
        scratch.d_xdbl_flat[t_off_xdbl..t_off_xdbl + xdbl_dim].fill(0.0);

        // SSM backward
        {
            let acts_delta = acts.delta(t);
            let acts_u = acts.u(t);
            let acts_da_exp = acts.da_exp(t);
            let acts_h_prev = acts.h_prev(t);
            let acts_h_curr = acts.h_curr(t);
            let acts_xdbl = acts.xdbl(t);

            for d in 0..di {
                let delta_d = acts_delta[d];
                let u_d = acts_u[d];
                let dy_d = scratch.d_y_flat[t_off_di + d];

                // d_D[d] += d_y[d] * u[d]
                d_layer.d_param[d] += dy_d * u_d;

                // d_u[d] += d_y[d] * D[d] (from skip connection)
                scratch.d_u_flat[t_off_di + d] += dy_d * w.d_param[d];

                for n in 0..ds {
                    let idx = d * ds + n;
                    let a_dn = a_neg[idx];
                    let da = acts_da_exp[idx];
                    let h_prev = acts_h_prev[idx];
                    let b_n = acts_xdbl[b_offset + n];
                    let c_n = acts_xdbl[c_offset + n];
                    let h_curr = acts_h_curr[idx];

                    // d_h[d,n] += d_y[d] * C[n]
                    scratch.d_h[idx] += dy_d * c_n;
                    let dh = scratch.d_h[idx];

                    // d_delta[d] += dh * (A[d,n] * da * h_prev + u * B[n])
                    scratch.d_delta_flat[t_off_di + d] += dh * (a_dn * da * h_prev + u_d * b_n);

                    // d_u[d] += dh * delta * B[n]
                    scratch.d_u_flat[t_off_di + d] += dh * delta_d * b_n;

                    // d_B[n] += dh * delta * u
                    scratch.d_xdbl_flat[t_off_xdbl + b_offset + n] += dh * delta_d * u_d;

                    // d_C[n] += d_y[d] * h[t,d,n]
                    scratch.d_xdbl_flat[t_off_xdbl + c_offset + n] += dy_d * h_curr;

                    // d_a_log[d,n] += dh * da * delta * A[d,n] * h_prev
                    d_layer.a_log[idx] += dh * da * delta_d * a_dn * h_prev;

                    // BPTT: propagate d_h backwards through time
                    scratch.d_h[idx] = da * dh;
                }
            }
        }

        // Softplus backward
        {
            let acts_delta_raw = acts.delta_raw(t);
            for (d, &raw) in acts_delta_raw.iter().enumerate().take(di) {
                scratch.d_delta_raw_flat[t_off_di + d] = if raw > 20.0 {
                    scratch.d_delta_flat[t_off_di + d]
                } else {
                    let sig = 1.0 / (1.0 + fast_exp_scalar(-raw));
                    scratch.d_delta_flat[t_off_di + d] * sig
                };
            }
        }
    }

    // ===================================================================
    // Phase B4: Batch dt_proj backward
    //   sgemm_backward(d_dt_input_flat, d_dt_proj_w, Some(d_dt_proj_b),
    //                  d_delta_raw_flat, xdbl_dt_buf, dt_proj_w,
    //                  (T, dt_rank, d_inner))
    //   Then accumulate d_dt_input into d_xdbl per timestep.
    // ===================================================================
    acts.copy_xdbl_dt_all(&mut scratch.xdbl_dt_buf);
    sgemm_backward(
        &mut scratch.d_dt_input_flat,
        &mut d_layer.dt_proj_w,
        Some(&mut d_layer.dt_proj_b),
        &scratch.d_delta_raw_flat,
        &scratch.xdbl_dt_buf,
        &w.dt_proj_w,
        (seq_len, dt_rank, di),
    );

    // Accumulate d_dt_input into d_xdbl[0..dt_rank] per timestep
    for t in 0..seq_len {
        let xdbl_off = t * xdbl_dim;
        let dt_off = t * dt_rank;
        for i in 0..dt_rank {
            scratch.d_xdbl_flat[xdbl_off + i] += scratch.d_dt_input_flat[dt_off + i];
        }
    }

    // ===================================================================
    // Phase B5: Batch x_proj backward
    //   sgemm_backward(d_u_xproj_flat, d_x_proj_w, None,
    //                  d_xdbl_flat, u_buf, x_proj_w,
    //                  (T, d_inner, xdbl_dim))
    //   Then accumulate: d_u_flat[i] += d_u_xproj_flat[i]
    // ===================================================================
    acts.copy_u_all(&mut scratch.u_buf);
    sgemm_backward(
        &mut scratch.d_u_xproj_flat,
        &mut d_layer.x_proj_w,
        None,
        &scratch.d_xdbl_flat,
        &scratch.u_buf,
        &w.x_proj_w,
        (seq_len, di, xdbl_dim),
    );

    // Accumulate d_u_xproj into d_u
    for (du, &du_xp) in scratch
        .d_u_flat
        .iter_mut()
        .zip(scratch.d_u_xproj_flat.iter())
    {
        *du += du_xp;
    }

    // ===================================================================
    // Phase B6: SiLU backward (batch) + conv1d backward with carry (SEQUENTIAL)
    // ===================================================================

    // SiLU backward (batch)
    for t in 0..seq_len {
        let t_off = t * di;
        let post_conv = acts.post_conv(t);
        for (d, &x) in post_conv.iter().enumerate().take(di) {
            let sig = 1.0 / (1.0 + fast_exp_scalar(-x));
            scratch.d_conv_out_flat[t_off + d] =
                scratch.d_u_flat[t_off + d] * sig * (1.0 + x * (1.0 - sig));
        }
    }

    // Conv1d backward: SEQUENTIAL, reverse. Includes shift-register gradient carry fix.
    for t in (0..seq_len).rev() {
        let t_off = t * di;

        backward_conv1d_step(
            &mut scratch.d_x_branch_flat[t_off..t_off + di],
            &mut d_layer.conv1d_weight,
            &mut d_layer.conv1d_bias,
            &scratch.d_conv_out_flat[t_off..t_off + di],
            acts.conv_state(t),
            &w.conv1d_weight,
            Conv1dDims {
                d_inner: di,
                d_conv: dc,
            },
        );

        // Propagate gradient through shift register positions 0..d_conv-2.
        if dc > 1 {
            let carry_stride = dc - 1;
            for d in 0..di {
                let carry_base = d * carry_stride;
                let w_base = d * dc;
                // Add accumulated carry from future timesteps
                scratch.d_x_branch_flat[t_off + d] += scratch.d_conv_carry[carry_base];
                // Shift carry left
                for k in 0..carry_stride - 1 {
                    scratch.d_conv_carry[carry_base + k] = scratch.d_conv_carry[carry_base + k + 1]
                        + scratch.d_conv_out_flat[t_off + d] * w.conv1d_weight[w_base + dc - 2 - k];
                }
                // Last carry position
                scratch.d_conv_carry[carry_base + carry_stride - 1] =
                    scratch.d_conv_out_flat[t_off + d] * w.conv1d_weight[w_base];
            }
        }
    }

    // ===================================================================
    // Phase B7: Batch in_proj backward
    //   Reconstruct d_proj from d_x_branch + d_gate (concatenate per timestep).
    //   sgemm_backward(d_norm_flat, d_in_proj_w, None,
    //                  d_proj_flat, post_norm_buf, in_proj_w,
    //                  (T, d_model, 2*d_inner))
    // ===================================================================
    for t in 0..seq_len {
        let t_off_di = t * di;
        let proj_off = t * 2 * di;
        scratch.d_proj_flat[proj_off..proj_off + di]
            .copy_from_slice(&scratch.d_x_branch_flat[t_off_di..t_off_di + di]);
        scratch.d_proj_flat[proj_off + di..proj_off + 2 * di]
            .copy_from_slice(&scratch.d_gate_flat[t_off_di..t_off_di + di]);
    }

    acts.copy_post_norm_all(&mut scratch.post_norm_buf);
    sgemm_backward(
        &mut scratch.d_norm_flat,
        &mut d_layer.in_proj_w,
        None,
        &scratch.d_proj_flat,
        &scratch.post_norm_buf,
        &w.in_proj_w,
        (seq_len, dm, 2 * di),
    );

    // ===================================================================
    // Phase B8: Batch RmsNorm backward + residual
    // ===================================================================
    for t in 0..seq_len {
        let off = t * dm;
        let rms_slice = [acts.rms_val(t)];
        scratch.d_pre_norm_flat[off..off + dm].fill(0.0);
        backward_rms_norm(
            &mut scratch.d_pre_norm_flat[off..off + dm],
            &mut d_layer.norm_weight,
            &scratch.d_norm_flat[off..off + dm],
            acts.residual(t),
            (&w.norm_weight, &rms_slice),
            1,
            dm,
        );

        // Residual gradient: d_temporal[t] += d_pre_norm[t]
        for d in 0..dm {
            d_temporal_flat[off + d] += scratch.d_pre_norm_flat[off + d];
        }
    }
}

// ---------------------------------------------------------------------------

/// O2 optimized full Mamba backbone backward pass with batched SGEMM.
///
/// Batches the input projection backward into a single
/// `sgemm_backward(T, mamba_input_dim, d_model)` call, and calls
/// [`backward_mamba_layer_batched`] for each layer in reverse.
///
/// On exit `d_temporal_flat` contains the gradient w.r.t. the input projection
/// output (the input embedding is detached — dx is computed and discarded).
///
/// # Arguments
///
/// - `d_temporal_flat`: `[T * d_model]` — upstream gradient. Modified in-place.
/// - `d_mamba`: gradient accumulators for all Mamba weights.
/// - `acts`: saved activations from forward pass.
/// - `mamba_w`: Mamba weights (read-only).
/// - `a_neg_all`: `[n_layers * d_inner * d_state]` — pre-computed `-exp(a_log)`.
/// - `scratch`: pre-allocated backward phase scratch buffers.
/// - `dims`: collected Mamba dimensions.
pub fn backward_mamba_backbone_batched(
    d_temporal_flat: &mut [f32],
    d_mamba: &mut TrainMambaWeights,
    acts: &MambaBackboneFlat,
    mamba_w: &TrainMambaWeights,
    a_neg_all: &[f32],
    scratch: &mut BackwardPhaseScratch,
    dims: &MambaDims,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let mid = dims.mamba_input_dim;
    let seq_len = dims.seq_len;
    let n_layers = dims.n_layers;

    let a_neg_per_layer = di * ds;

    // --- norm_f backward (before layer loop, since layers are in reverse) ---
    // d_temporal_flat arrives with upstream gradient. Apply norm_f backward first.
    {
        let norm_f_dx = &mut scratch.d_input_proj_scratch[..seq_len * dm];
        norm_f_dx.fill(0.0);
        scratch.d_norm_f_weight_local.fill(0.0);
        backward_rms_norm(
            norm_f_dx,
            &mut scratch.d_norm_f_weight_local,
            &d_temporal_flat[..seq_len * dm],
            &acts.norm_f_input[..seq_len * dm],
            (&mamba_w.norm_f_weight, &acts.norm_f_rms[..seq_len]),
            seq_len,
            dm,
        );
        // Copy norm_f dx back into d_temporal_flat for the layer backward passes
        d_temporal_flat[..seq_len * dm].copy_from_slice(&norm_f_dx[..seq_len * dm]);
        // Accumulate d_norm_f_weight gradient
        for (a, b) in d_mamba
            .norm_f_weight
            .iter_mut()
            .zip(&scratch.d_norm_f_weight_local)
        {
            *a += b;
        }
    }

    // --- Mamba layers in reverse (batched) ---
    for layer_idx in (0..n_layers).rev() {
        let a_neg_start = layer_idx * a_neg_per_layer;

        backward_mamba_layer_batched(
            d_temporal_flat,
            &mut d_mamba.layers[layer_idx],
            &acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            &a_neg_all[a_neg_start..a_neg_start + a_neg_per_layer],
            scratch,
            dims,
        );
    }

    // --- Batched input projection backward ---
    // dx is discarded (input embedding is detached from the Mamba graph).
    // Use d_input_proj_scratch from BackwardPhaseScratch for the discarded dx.
    sgemm_backward(
        &mut scratch.d_input_proj_scratch,
        &mut d_mamba.input_proj_w,
        Some(&mut d_mamba.input_proj_b),
        &d_temporal_flat[..seq_len * dm],
        &acts.input_proj_inputs[..seq_len * mid],
        &mamba_w.input_proj_w,
        (seq_len, mid, dm),
    );
    // d_input_proj_scratch is intentionally discarded (input embedding detached)
}
