//! Mamba-3 SISO T=1 recurrent inference step.
//!
//! Single-step forward pass for auto-regressive generation.
//! Operates on persistent state (SSM + K + V + angle).
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).
//! Reference: `mamba3_siso_step.py` lines 128-222.

use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::state::Mamba3LayerState;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_scalar};
use crate::ops::norms::{bcnorm, rms_norm_weighted, rmsnorm_gated};

/// Pre-allocated scratch buffers for Mamba-3 T=1 step.
pub struct Mamba3StepScratch {
    pub proj: Vec<f32>,       // [in_proj_dim]
    pub z: Vec<f32>,          // [d_inner]
    pub x: Vec<f32>,          // [d_inner]
    pub b_raw: Vec<f32>,      // [ngroups * d_state]
    pub c_raw: Vec<f32>,      // [ngroups * d_state]
    pub b_normed: Vec<f32>,   // [ngroups * d_state]
    pub c_normed: Vec<f32>,   // [ngroups * d_state]
    pub y: Vec<f32>,          // [d_inner]
    pub gated: Vec<f32>,      // [d_inner]
    pub norm_buf: Vec<f32>,   // [d_model]
    pub residual: Vec<f32>,   // [d_model]
    pub bc_inv_rms: Vec<f32>, // [ngroups] (BCNorm inv_rms scratch)
}

impl Mamba3StepScratch {
    /// Allocate scratch buffers from config.
    pub fn new(cfg: &Mamba3Config) -> Self {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let ng = cfg.ngroups;
        let ip = cfg.in_proj_out_dim();
        Self {
            proj: vec![0.0; ip],
            z: vec![0.0; di],
            x: vec![0.0; di],
            b_raw: vec![0.0; ng * ds],
            c_raw: vec![0.0; ng * ds],
            b_normed: vec![0.0; ng * ds],
            c_normed: vec![0.0; ng * ds],
            y: vec![0.0; di],
            gated: vec![0.0; di],
            norm_buf: vec![0.0; dm],
            residual: vec![0.0; dm],
            bc_inv_rms: vec![0.0; ng],
        }
    }
}

/// Mamba-3 SISO single-layer T=1 step.
///
/// Processes one timestep through one Mamba-3 layer, mutating persistent state.
/// `temporal` is both input and output: `[d_model]`.
///
/// ## Algorithm
/// ```text
/// temporal → RMSNorm → in_proj → 8-way split
///   → BCNorm(B,C) → per-head: +bias → RoPE → trapezoidal SSM
///   → y = h@C + D*x → gating (SiLU or RMSNormGated)
///   → out_proj + residual → temporal
/// ```
pub fn mamba3_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba3StepScratch,
    lw: &Mamba3LayerWeights,
    state: &mut Mamba3LayerState,
    cfg: &Mamba3Config,
) {
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let hd = cfg.headdim;
    let ng = cfg.ngroups;
    let ip = cfg.in_proj_out_dim();
    let n_rope = cfg.num_rope_angles();
    let a_floor = cfg.a_floor;

    // 1. Save residual
    scratch.residual[..dm].copy_from_slice(&temporal[..dm]);

    // 2. RMSNorm
    rms_norm_weighted(
        &mut scratch.norm_buf[..dm],
        &temporal[..dm],
        &lw.norm_weight,
        RMS_NORM_EPS,
    );

    // 3. in_proj: [d_model] → [in_proj_dim] (BLAS matvec, no bias)
    crate::ops::blas::matvec_forward(
        &mut scratch.proj[..ip],
        &scratch.norm_buf[..dm],
        &lw.in_proj_w,
        None,
        dm,
        ip,
    );

    // 4. 8-way split: z, x, B, C, dd_dt, dd_A, trap, angles
    let mut off = 0;
    scratch.z[..di].copy_from_slice(&scratch.proj[off..off + di]);
    off += di;
    scratch.x[..di].copy_from_slice(&scratch.proj[off..off + di]);
    off += di;
    scratch.b_raw[..ng * ds].copy_from_slice(&scratch.proj[off..off + ng * ds]);
    off += ng * ds;
    scratch.c_raw[..ng * ds].copy_from_slice(&scratch.proj[off..off + ng * ds]);
    off += ng * ds;
    let dd_dt_off = off;
    off += nh;
    let dd_a_off = off;
    off += nh;
    let trap_off = off;
    off += nh;
    let angles_off = off;

    // 5. BCNorm: per-group RMSNorm(B) * weight, RMSNorm(C) * weight
    bcnorm(
        &mut scratch.b_normed[..ng * ds],
        &scratch.b_raw[..ng * ds],
        &lw.b_norm_weight,
        ng,
        ds,
        RMS_NORM_EPS,
        &mut scratch.bc_inv_rms,
    );
    bcnorm(
        &mut scratch.c_normed[..ng * ds],
        &scratch.c_raw[..ng * ds],
        &lw.c_norm_weight,
        ng,
        ds,
        RMS_NORM_EPS,
        &mut scratch.bc_inv_rms,
    );

    // 6. Per-head: bias + RoPE + A/DT + trapezoidal SSM
    for h in 0..nh {
        let g = h / (nh / ng);

        // Input-dependent A: A = -softplus(dd_A), clamp max=-a_floor
        let a_val = (-super::forward::softplus(scratch.proj[dd_a_off + h])).min(-a_floor);

        // DT = softplus(dd_dt + dt_bias)
        let dt_val = super::forward::softplus(scratch.proj[dd_dt_off + h] + lw.dt_bias[h]);

        // Per-head B/C with bias
        let mut k_local = [0.0_f32; 64];
        let mut q_local = [0.0_f32; 64];
        for n in 0..ds {
            k_local[n] = scratch.b_normed[g * ds + n] + lw.b_bias[h * ds + n];
            q_local[n] = scratch.c_normed[g * ds + n] + lw.c_bias[h * ds + n];
        }

        // RoPE: per-head angle accumulation and rotation
        if n_rope > 0 {
            let angle_base = h * n_rope;
            let pi = std::f32::consts::PI;
            for a in 0..n_rope {
                let raw = scratch.proj[angles_off + a];
                let delta = raw.tanh() * pi * dt_val;
                let mut acc = state.angle_state[angle_base + a] as f64 + delta as f64;
                let two_pi_64 = 2.0 * std::f64::consts::PI;
                acc -= two_pi_64 * (acc / two_pi_64).floor();
                state.angle_state[angle_base + a] = acc as f32;
                let (sin_a, cos_a) = state.angle_state[angle_base + a].sin_cos();

                let i0 = 2 * a;
                let i1 = 2 * a + 1;
                let b0 = k_local[i0];
                let b1 = k_local[i1];
                k_local[i0] = cos_a * b0 - sin_a * b1;
                k_local[i1] = sin_a * b0 + cos_a * b1;
                let c0 = q_local[i0];
                let c1 = q_local[i1];
                q_local[i0] = cos_a * c0 - sin_a * c1;
                q_local[i1] = sin_a * c0 + cos_a * c1;
            }
        }

        // Trapezoidal: alpha, beta, gamma
        let alpha = fast_exp_scalar(a_val * dt_val);
        let trap = 1.0 / (1.0 + fast_exp_scalar(-scratch.proj[trap_off + h]));
        let beta = alpha * dt_val * (1.0 - trap);
        let gamma = trap * dt_val;

        // SSM recurrence (SIMD): h = alpha*h + beta*v_prev*k_prev + gamma*x*k_cur
        let kp = &state.k_state[h * ds..h * ds + ds];
        for p in 0..hd {
            let x_val = scratch.x[h * hd + p];
            let v_prev = state.v_state[h * hd + p];
            let s_off = (h * hd + p) * ds;
            let y_val = super::forward::simd_ssm_recurrence(
                &mut state.ssm_state[s_off..s_off + ds],
                kp,
                &k_local[..ds],
                &q_local[..ds],
                alpha,
                beta * v_prev,
                gamma * x_val,
            );
            scratch.y[h * hd + p] = lw.d_param[h].mul_add(x_val, y_val);
        }

        // Update K and V state for next step
        state.k_state[h * ds..h * ds + ds].copy_from_slice(&k_local[..ds]);
        for p in 0..hd {
            state.v_state[h * hd + p] = scratch.x[h * hd + p];
        }
    }

    // 7. Output gating
    if cfg.is_outproj_norm {
        rmsnorm_gated(
            &mut scratch.gated[..di],
            &scratch.y[..di],
            &scratch.z[..di],
            &lw.norm_gate_weight,
            hd,
            RMS_NORM_EPS,
        );
    } else {
        for i in 0..di {
            let z = scratch.z[i];
            let sig = 1.0 / (1.0 + fast_exp_scalar(-z));
            scratch.gated[i] = scratch.y[i] * z * sig;
        }
    }

    // 8. out_proj: [d_inner] → [d_model] (BLAS matvec) + residual
    crate::ops::blas::matvec_forward(
        &mut temporal[..dm],
        &scratch.gated[..di],
        &lw.out_proj_w,
        None,
        di,
        dm,
    );
    for j in 0..dm {
        temporal[j] += scratch.residual[j];
    }
}

/// Full Mamba-3 SISO backbone step: input_proj → all layers → norm_f.
///
/// `input`: `[input_dim]` observation vector.
/// `temporal`: `[d_model]` working buffer (output after step).
pub fn mamba3_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba3StepScratch,
    weights: &Mamba3Weights,
    states: &mut [Mamba3LayerState],
    cfg: &Mamba3Config,
) {
    let dm = cfg.d_model;
    let input_dim = input.len();

    // Input projection: [input_dim] → [d_model] (BLAS matvec + bias)
    crate::ops::blas::matvec_forward(
        &mut temporal[..dm],
        &input[..input_dim],
        &weights.input_proj_w,
        Some(&weights.input_proj_b),
        input_dim,
        dm,
    );

    // Process each layer
    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        mamba3_layer_step(temporal, scratch, lw, &mut states[layer_idx], cfg);
    }

    // Final RMSNorm (reuse scratch.norm_buf to avoid heap allocation)
    rms_norm_weighted(
        &mut scratch.norm_buf[..dm],
        &temporal[..dm],
        &weights.norm_f_weight,
        RMS_NORM_EPS,
    );
    temporal[..dm].copy_from_slice(&scratch.norm_buf[..dm]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba3_siso::state::Mamba3State;

    fn test_cfg() -> Mamba3Config {
        Mamba3Config {
            d_model: 32,
            d_state: 8,
            expand: 2,
            headdim: 8,
            ngroups: 1,
            n_layers: 2,
            rope_fraction: 0.5,
            a_floor: 0.0625,
            is_outproj_norm: false,
        }
    }

    #[test]
    fn test_m3_single_step_nonzero() {
        let cfg = test_cfg();
        cfg.validate();
        let w = Mamba3Weights::init(&cfg, 16, 42);
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3StepScratch::new(&cfg);
        let input = vec![1.0_f32; 16];
        let mut temporal = vec![0.0; cfg.d_model];

        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );

        assert!(
            temporal.iter().any(|&v| v != 0.0),
            "output should be non-zero"
        );
        assert!(
            temporal.iter().all(|v| v.is_finite()),
            "output should be finite"
        );
    }

    #[test]
    fn test_m3_state_carries() {
        let cfg = test_cfg();
        let w = Mamba3Weights::init(&cfg, 16, 42);
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3StepScratch::new(&cfg);
        let input = vec![1.0_f32; 16];
        let mut temporal = vec![0.0; cfg.d_model];

        // Step 1
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        let out1 = temporal.clone();

        // Step 2 — different output because state changed
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        let out2 = temporal.clone();

        assert_ne!(out1, out2, "state should cause different outputs");
    }

    #[test]
    fn test_m3_state_reset_reproduces() {
        let cfg = test_cfg();
        let w = Mamba3Weights::init(&cfg, 16, 42);
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3StepScratch::new(&cfg);
        let input = vec![1.0_f32; 16];
        let mut temporal = vec![0.0; cfg.d_model];

        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        let out1 = temporal.clone();

        // Reset and redo
        state.reset();
        temporal.fill(0.0);
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        let out2 = temporal.clone();

        assert_eq!(out1, out2, "reset should reproduce original output");
    }

    #[test]
    fn test_m3_deterministic() {
        let cfg = test_cfg();
        let w = Mamba3Weights::init(&cfg, 16, 42);

        let run = || {
            let mut state = Mamba3State::zeros(&cfg);
            let mut scratch = Mamba3StepScratch::new(&cfg);
            let input = vec![0.5_f32; 16];
            let mut temporal = vec![0.0; cfg.d_model];
            for _ in 0..5 {
                mamba3_step(
                    &mut temporal,
                    &input,
                    &mut scratch,
                    &w,
                    &mut state.layers,
                    &cfg,
                );
            }
            temporal
        };

        assert_eq!(run(), run(), "must be deterministic");
    }

    #[test]
    fn test_m3_rope_angles_accumulate() {
        let cfg = test_cfg();
        let w = Mamba3Weights::init(&cfg, 16, 42);
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3StepScratch::new(&cfg);
        let input = vec![1.0_f32; 16];
        let mut temporal = vec![0.0; cfg.d_model];

        // After 10 steps, angle state should be non-zero and in [0, 2pi)
        for _ in 0..10 {
            mamba3_step(
                &mut temporal,
                &input,
                &mut scratch,
                &w,
                &mut state.layers,
                &cfg,
            );
        }

        let two_pi = 2.0 * std::f32::consts::PI;
        for layer in &state.layers {
            assert!(
                layer.angle_state.iter().any(|&v| v != 0.0),
                "angles should accumulate"
            );
            assert!(
                layer.angle_state.iter().all(|&v| v >= 0.0 && v < two_pi),
                "angles should be in [0, 2pi)"
            );
        }
    }

    #[test]
    fn test_m3_bcnorm_unit_weight() {
        // With weight=1, BCNorm should produce values with unit RMS
        let cfg = test_cfg();
        let ds = cfg.d_state;
        let x = vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0]; // ds=8
        let w = vec![1.0; ds];
        let mut out = vec![0.0; ds];
        let mut inv_rms = vec![0.0; 1];
        bcnorm(&mut out, &x, &w, 1, ds, 1e-5, &mut inv_rms);

        let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / ds as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "BCNorm with unit weight should produce ~unit RMS, got {rms}"
        );
    }
}
