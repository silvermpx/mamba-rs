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

use super::flat::{MambaBackboneFlat, MambaLayerFlat};
use super::scratch::PhaseScratch;
use super::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use crate::ops::blas::{matvec_forward, sgemm_forward};
use crate::ops::dims::{MambaDims, MambaRecurrentState};
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_scalar};

/// O2 optimized single-layer Mamba forward pass: 7-phase pipeline with batched SGEMM.
///
/// Restructures the per-timestep forward into phases where the two biggest SGEMM
/// calls (in_proj and out_proj) are batched across ALL timesteps in a single call.
/// The SSM core (phase F4) remains sequential since SSM recurrence is inherently
/// sequential.
///
/// Amortises SGEMM dispatch overhead across `T` timesteps.
///
/// # 7 Phases
///
/// - **F1**: Batch RmsNorm (loop over T)
/// - **F2**: Batch in_proj — ONE SGEMM call `[T * d_model] -> [T * 2*d_inner]`
/// - **F3**: Split x/gate + batch SiLU (loop over T)
/// - **F4**: Sequential SSM core (per-timestep: conv1d, x_proj, dt_proj, SSM, gating)
/// - **F5**: Batch out_proj — ONE SGEMM call `[T * d_inner] -> [T * d_model]`
/// - **F6**: Residual add (all T)
///
/// # Arguments
///
/// - `temporal_flat`: `[T * d_model]` — input on entry, output on exit (in-place).
/// - `acts`: flat activation storage for this layer.
/// - `layer_w`: this layer's weights.
/// - `conv_state`: `[d_inner * d_conv]` — persistent conv1d shift register state.
/// - `ssm_state`: `[d_inner * d_state]` — persistent SSM hidden state.
/// - `a_neg`: `[d_inner * d_state]` — pre-computed `-exp(a_log[idx])`.
/// - `scratch`: per-phase scratch buffers, allocated once and reused.
/// - `dims`: collected Mamba dimensions.
pub fn forward_mamba_layer_batched(
    temporal_flat: &mut [f32],
    acts: &mut MambaLayerFlat,
    layer_w: &TrainMambaLayerWeights,
    state: &mut MambaRecurrentState<'_>,
    scratch: &mut PhaseScratch,
    dims: &MambaDims,
) {
    let conv_state = &mut *state.conv;
    let ssm_state = &mut *state.ssm;
    let a_neg = state.a_neg;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let seq_len = dims.seq_len;

    // ===================================================================
    // Phase F1: Batch RmsNorm (loop over T)
    // ===================================================================
    for t in 0..seq_len {
        let off = t * dm;
        let src = &temporal_flat[off..off + dm];

        // Save residual
        acts.residual_mut(t).copy_from_slice(src);

        // Compute RMS
        let mut sum_sq = 0.0_f32;
        for &v in &src[..dm] {
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / dm as f32;
        let rms = (mean_sq + RMS_NORM_EPS).sqrt();
        acts.set_rms_val(t, rms);
        let inv_rms = 1.0 / rms;

        // Write post_norm into scratch buffer for batched in_proj
        let pn = &mut scratch.post_norm_flat[off..off + dm];
        for d in 0..dm {
            pn[d] = src[d] * inv_rms * layer_w.norm_weight[d];
        }

        // Also save post_norm into acts (needed for backward)
        acts.post_norm_mut(t).copy_from_slice(pn);
    }

    // ===================================================================
    // Phase F2: Batch in_proj -- ONE SGEMM call
    //           [T * d_model] -> [T * 2*d_inner]
    // ===================================================================
    sgemm_forward(
        &mut scratch.proj_flat,
        &scratch.post_norm_flat,
        &layer_w.in_proj_w,
        None,
        seq_len,
        dm,
        2 * di,
    );

    // ===================================================================
    // Phase F3: Split x/gate + batch SiLU (loop over T)
    // ===================================================================
    for t in 0..seq_len {
        let proj_off = t * 2 * di;

        // x_branch = proj[..d_inner]
        acts.x_branch_mut(t)
            .copy_from_slice(&scratch.proj_flat[proj_off..proj_off + di]);

        // gate_pre_silu = proj[d_inner..2*d_inner]
        let gate = &scratch.proj_flat[proj_off + di..proj_off + 2 * di];
        acts.gate_pre_silu_mut(t).copy_from_slice(gate);

        // SiLU(gate) -> scratch.gate_silu_flat
        let gs_off = t * di;
        for (d, &g) in gate.iter().enumerate().take(di) {
            let sig = 1.0 / (1.0 + fast_exp_scalar(-g));
            scratch.gate_silu_flat[gs_off + d] = g * sig;
        }
        acts.gate_post_silu_mut(t)
            .copy_from_slice(&scratch.gate_silu_flat[gs_off..gs_off + di]);
    }

    // ===================================================================
    // Phase F4: Sequential SSM core (per-timestep)
    //           conv1d -> x_proj -> dt_proj -> SSM recurrence -> gating
    // ===================================================================
    let b_offset = dt_rank;
    let c_offset = dt_rank + ds;

    for t in 0..seq_len {
        // -- F4a: conv1d shift + fused SiLU --
        {
            let x_branch = acts.x_branch(t);
            for (d, &xb) in x_branch.iter().enumerate().take(di) {
                let base = d * dc;
                for k in 0..dc - 1 {
                    conv_state[base + k] = conv_state[base + k + 1];
                }
                conv_state[base + dc - 1] = xb;
            }
        }
        acts.conv_state_mut(t)
            .copy_from_slice(&conv_state[..di * dc]);

        // Fused depthwise conv1d dot product + SiLU
        {
            let step_base = t * acts.offsets.step_stride;
            let pc_start = step_base + acts.offsets.post_conv;
            let u_start = step_base + acts.offsets.u;
            for d in 0..di {
                let base = d * dc;
                let mut val = layer_w.conv1d_bias[d];
                for k in 0..dc {
                    val += conv_state[base + k] * layer_w.conv1d_weight[base + k];
                }
                acts.data[pc_start + d] = val;
                acts.data[u_start + d] = val / (1.0 + fast_exp_scalar(-val));
            }
        }

        // -- F4b: x_proj: u -> xdbl --
        // Copy u out to avoid aliasing (u and xdbl in same flat buffer).
        let gs_off = t * di;
        scratch.gate_silu_flat[..di].copy_from_slice(acts.u(t));
        matvec_forward(
            acts.xdbl_mut(t),
            &scratch.gate_silu_flat[..di],
            &layer_w.x_proj_w,
            None,
            di,
            xdbl_dim,
        );

        // -- F4c: dt_proj + softplus --
        // Copy dt portion out (xdbl and delta_raw alias in same flat buffer).
        scratch.gate_silu_flat[..dt_rank].copy_from_slice(&acts.xdbl(t)[..dt_rank]);
        matvec_forward(
            acts.delta_raw_mut(t),
            &scratch.gate_silu_flat[..dt_rank],
            &layer_w.dt_proj_w,
            Some(&layer_w.dt_proj_b),
            dt_rank,
            di,
        );

        // Softplus
        {
            let step_base = t * acts.offsets.step_stride;
            let dr_start = step_base + acts.offsets.delta_raw;
            let d_start = step_base + acts.offsets.delta;
            for d in 0..di {
                let raw = acts.data[dr_start + d];
                acts.data[d_start + d] = if raw > 20.0 {
                    raw
                } else {
                    (1.0_f32 + fast_exp_scalar(raw)).ln()
                };
            }
        }

        // -- F4d: SSM recurrence --
        acts.h_prev_mut(t).copy_from_slice(&ssm_state[..di * ds]);

        {
            let step_base = t * acts.offsets.step_stride;
            let delta_start = step_base + acts.offsets.delta;
            let u_start = step_base + acts.offsets.u;
            let xdbl_start = step_base + acts.offsets.xdbl;
            let da_start = step_base + acts.offsets.da_exp;
            let y_start = step_base + acts.offsets.y;

            for d in 0..di {
                let delta_d = acts.data[delta_start + d];
                let u_d = acts.data[u_start + d];
                let delta_u_d = delta_d * u_d; // hoisted from inner loop (C2/T6)
                let mut y_d = 0.0_f32;

                for n in 0..ds {
                    let idx = d * ds + n;
                    let a_dn = a_neg[idx];
                    let b_n = acts.data[xdbl_start + b_offset + n];
                    let c_n = acts.data[xdbl_start + c_offset + n];

                    let da = fast_exp_scalar(delta_d * a_dn);
                    acts.data[da_start + idx] = da;

                    let h_prev = ssm_state[idx];
                    ssm_state[idx] = da * h_prev + delta_u_d * b_n;

                    y_d += ssm_state[idx] * c_n;
                }

                y_d += layer_w.d_param[d] * u_d;
                acts.data[y_start + d] = y_d;
            }
        }

        acts.h_curr_mut(t).copy_from_slice(&ssm_state[..di * ds]);

        // -- F4e: gating: gated = y * gate_post_silu --
        {
            let step_base = t * acts.offsets.step_stride;
            let y_start = step_base + acts.offsets.y;
            let gpost_start = step_base + acts.offsets.gate_post_silu;
            let gated_start = step_base + acts.offsets.gated;
            for d in 0..di {
                scratch.gated_flat[gs_off + d] =
                    acts.data[y_start + d] * acts.data[gpost_start + d];
                acts.data[gated_start + d] = scratch.gated_flat[gs_off + d];
            }
        }
    }

    // ===================================================================
    // Phase F5: Batch out_proj -- ONE SGEMM call
    //           [T * d_inner] -> [T * d_model]
    // ===================================================================
    sgemm_forward(
        &mut scratch.out_flat,
        &scratch.gated_flat,
        &layer_w.out_proj_w,
        None,
        seq_len,
        di,
        dm,
    );

    // ===================================================================
    // Phase F6: Residual add (all T)
    // ===================================================================
    for t in 0..seq_len {
        let off = t * dm;
        let residual = acts.residual(t);
        for d in 0..dm {
            temporal_flat[off + d] = residual[d] + scratch.out_flat[off + d];
        }
    }
}

// ---------------------------------------------------------------------------
// Full Mamba backbone — O2 batched SGEMM
// ---------------------------------------------------------------------------

/// O2 optimized full Mamba backbone forward pass with batched SGEMM.
///
/// Batches the input projection into a single `sgemm_forward(T, mamba_input_dim,
/// d_model)` call, and calls [`forward_mamba_layer_batched`] for each layer.
///
/// # Arguments
///
/// - `temporal_flat`: `[T * d_model]` — written by input projection, then
///   modified in-place by each Mamba layer.
/// - `acts`: flat activation storage for the entire backbone.
/// - `mamba_w`: all Mamba weights (input_proj + layers).
/// - `mamba_input_flat`: `[T * mamba_input_dim]` — input sequence.
/// - `conv_states`: `[n_layers * d_inner * d_conv]` — persistent conv states.
/// - `ssm_states`: `[n_layers * d_inner * d_state]` — persistent SSM states.
/// - `a_neg_all`: `[n_layers * d_inner * d_state]` — pre-computed `-exp(a_log)`.
/// - `scratch`: per-phase scratch buffers (allocated once, reused).
/// - `dims`: collected Mamba dimensions.
pub fn forward_mamba_backbone_batched(
    temporal_flat: &mut [f32],
    acts: &mut MambaBackboneFlat,
    mamba_w: &TrainMambaWeights,
    mamba_input_flat: &[f32],
    state: &mut MambaRecurrentState<'_>,
    scratch: &mut PhaseScratch,
    dims: &MambaDims,
) {
    let conv_states = &mut *state.conv;
    let ssm_states = &mut *state.ssm;
    let a_neg_all = state.a_neg;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let mid = dims.mamba_input_dim;
    let seq_len = dims.seq_len;
    let n_layers = dims.n_layers;

    // --- Batched input projection: [T * mid] -> [T * dm] ---
    // Save inputs for backward
    acts.input_proj_inputs
        .copy_from_slice(&mamba_input_flat[..seq_len * mid]);

    // Single batched SGEMM call
    sgemm_forward(
        &mut temporal_flat[..seq_len * dm],
        &mamba_input_flat[..seq_len * mid],
        &mamba_w.input_proj_w,
        Some(&mamba_w.input_proj_b),
        seq_len,
        mid,
        dm,
    );

    // Save outputs for backward
    acts.input_proj_outputs[..seq_len * dm].copy_from_slice(&temporal_flat[..seq_len * dm]);

    // --- Mamba layers (batched) ---
    let conv_per_layer = di * dc;
    let ssm_per_layer = di * ds;
    let a_neg_per_layer = di * ds;

    for layer_idx in 0..n_layers {
        let conv_start = layer_idx * conv_per_layer;
        let ssm_start = layer_idx * ssm_per_layer;
        let a_neg_start = layer_idx * a_neg_per_layer;

        forward_mamba_layer_batched(
            temporal_flat,
            &mut acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            &mut MambaRecurrentState {
                conv: &mut conv_states[conv_start..conv_start + conv_per_layer],
                ssm: &mut ssm_states[ssm_start..ssm_start + ssm_per_layer],
                a_neg: &a_neg_all[a_neg_start..a_neg_start + a_neg_per_layer],
            },
            scratch,
            dims,
        );
    }

    // Final RmsNorm (norm_f) after all Mamba layers.
    // Save pre-norm input for backward, then apply norm in-place.
    acts.norm_f_input[..seq_len * dm].copy_from_slice(&temporal_flat[..seq_len * dm]);
    for t in 0..seq_len {
        let off = t * dm;
        let mean_sq: f32 = temporal_flat[off..off + dm]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / dm as f32;
        let rms = (mean_sq + RMS_NORM_EPS).sqrt();
        acts.norm_f_rms[t] = rms;
        let inv_rms = 1.0 / rms;
        for d in 0..dm {
            temporal_flat[off + d] = temporal_flat[off + d] * inv_rms * mamba_w.norm_f_weight[d];
        }
    }
}
