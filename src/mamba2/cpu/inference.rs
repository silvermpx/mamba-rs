//! Mamba-2 decode: one token through the backbone, mutating the state.
//!
//! ```text
//! temporal -> RMSNorm -> in_proj -> split [z0 | x0 | z | xBC | dt]
//!   -> conv over the last d_conv inputs + SiLU -> x | B | C
//!   -> per head: dt, dA, scan along the head's channels, + D x
//!   -> [silu(z0) x0 | gated RMSNorm(y, z)] -> out_proj + residual
//!   [-> gated MLP] -> ... -> final RMSNorm
//! ```

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::{Mamba2LayerWeights, Mamba2Weights};
use super::core::{conv_hist_push, conv_silu_token, gate_token, rms_row, ssm_token};
use crate::mlp::cpu::{GatedMlpPacked, GatedMlpStepScratch, gated_mlp_step, gated_mlp_step_packed};
use crate::ops::packed::{PackedLinear, linear_row};

/// One layer's projections packed for decode.
#[derive(Clone, Debug)]
pub struct Mamba2LayerPacked {
    pub in_proj: PackedLinear,
    pub out_proj: PackedLinear,
}

/// Every projection of a Mamba-2 backbone packed for decode: a copy of the
/// weights' projections, so it must be packed again after they change.
#[derive(Clone, Debug)]
pub struct Mamba2PackedLinears {
    pub layers: Vec<Mamba2LayerPacked>,
    /// One per layer with an MLP.
    pub mlp: Vec<GatedMlpPacked>,
    /// The input projection, absent for a backbone without one.
    pub input_proj: Option<PackedLinear>,
}

impl Mamba2PackedLinears {
    /// Pack every projection of `weights`.
    pub fn pack(weights: &Mamba2Weights, cfg: &Mamba2Config, input_dim: usize) -> Self {
        let (dm, di) = (cfg.d_model, cfg.d_inner());
        Self {
            layers: weights
                .layers
                .iter()
                .map(|lw| Mamba2LayerPacked {
                    in_proj: PackedLinear::pack(&lw.in_proj_w, dm, cfg.in_proj_out_dim()),
                    out_proj: PackedLinear::pack(&lw.out_proj_w, di, dm),
                })
                .collect(),
            mlp: weights.mlp.iter().map(GatedMlpPacked::pack).collect(),
            input_proj: (!weights.input_proj_w.is_empty())
                .then(|| PackedLinear::pack(&weights.input_proj_w, input_dim, dm)),
        }
    }
}

/// Reusable buffers for [`mamba2_step`].
pub struct Mamba2StepScratch {
    residual: Vec<f32>,
    norm_buf: Vec<f32>,
    proj: Vec<f32>,
    xbc: Vec<f32>,
    y: Vec<f32>,
    gated: Vec<f32>,
    mlp: GatedMlpStepScratch,
}

impl Mamba2StepScratch {
    pub fn new(cfg: &Mamba2Config) -> Self {
        let dm = cfg.d_model;
        Self {
            residual: vec![0.0; dm],
            norm_buf: vec![0.0; dm],
            proj: vec![0.0; cfg.in_proj_out_dim()],
            xbc: vec![0.0; cfg.conv_dim()],
            y: vec![0.0; cfg.ssm_dim()],
            gated: vec![0.0; cfg.d_inner()],
            mlp: GatedMlpStepScratch::new(dm, cfg.mlp_hidden()),
        }
    }
}

/// `y = x W (+ b)` for one row, the bias added by the GEMM, from the
/// packed panels when there are some.
fn project(y: &mut [f32], x: &[f32], w: (&[f32], Option<&PackedLinear>), b: &[f32]) {
    let bias = if b.is_empty() { None } else { Some(b) };
    let (n_in, n_out) = (x.len(), y.len());
    linear_row(y, x, w, bias, (n_in, n_out));
}

/// One token through one Mamba-2 mixer; `temporal` (`[d_model]`) is both
/// the input and the output, residual included.
pub fn mamba2_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    lw: &Mamba2LayerWeights,
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    layer_step(temporal, scratch, (lw, None), state, cfg);
}

/// [`mamba2_layer_step`] with the projections packed when `lp` holds them.
fn layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    (lw, lp): (&Mamba2LayerWeights, Option<&Mamba2LayerPacked>),
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    let dm = cfg.d_model;
    let (d_mlp, d_ssm, cd) = (cfg.d_mlp(), cfg.ssm_dim(), cfg.conv_dim());
    scratch.residual.copy_from_slice(&temporal[..dm]);
    rms_row(
        &mut scratch.norm_buf,
        &temporal[..dm],
        &lw.norm_weight,
        cfg.rms_norm_eps,
    );
    project(
        &mut scratch.proj,
        &scratch.norm_buf,
        (&lw.in_proj_w, lp.map(|p| &p.in_proj)),
        &lw.in_proj_b,
    );
    let xbc_raw = &scratch.proj[2 * d_mlp + d_ssm..2 * d_mlp + d_ssm + cd];
    let dt_raw = &scratch.proj[2 * d_mlp + d_ssm + cd..];
    conv_silu_token(&mut scratch.xbc, xbc_raw, &state.conv_hist, lw);
    conv_hist_push(&mut state.conv_hist, xbc_raw);
    ssm_token(
        (&mut state.ssm, &mut scratch.y),
        &scratch.xbc,
        dt_raw,
        lw,
        cfg,
    );
    gate_token(&mut scratch.gated, &scratch.y, &scratch.proj, lw, cfg);
    project(
        &mut temporal[..dm],
        &scratch.gated,
        (&lw.out_proj_w, lp.map(|p| &p.out_proj)),
        &lw.out_proj_b,
    );
    for (t, &r) in temporal[..dm].iter_mut().zip(&scratch.residual) {
        *t += r;
    }
}

/// One token through the whole Mamba-2 backbone: input projection, every
/// layer (mixer, then the MLP when the model has one), final norm.
/// `input` is `[input_dim]`; `temporal` (`[d_model]`) receives the output.
pub fn mamba2_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    weights: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    backbone_step(temporal, input, scratch, (weights, None), states, cfg);
}

/// [`mamba2_step`] with every projection taken from `packed` (packed from
/// these `weights` by [`Mamba2PackedLinears::pack`]).
///
/// # Panics
///
/// When `packed` holds another number of layers or MLPs than `weights`.
pub fn mamba2_step_packed(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    (weights, packed): (&Mamba2Weights, &Mamba2PackedLinears),
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    assert!(
        packed.layers.len() == weights.layers.len() && packed.mlp.len() == weights.mlp.len(),
        "packed projections of another backbone"
    );
    backbone_step(
        temporal,
        input,
        scratch,
        (weights, Some(packed)),
        states,
        cfg,
    );
}

fn backbone_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    (weights, packed): (&Mamba2Weights, Option<&Mamba2PackedLinears>),
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    let dm = cfg.d_model;
    if weights.input_proj_w.is_empty() {
        temporal[..dm].copy_from_slice(&input[..dm]);
    } else {
        project(
            &mut temporal[..dm],
            input,
            (
                &weights.input_proj_w,
                packed.and_then(|p| p.input_proj.as_ref()),
            ),
            &weights.input_proj_b,
        );
    }
    for (l, (lw, st)) in weights.layers.iter().zip(states.iter_mut()).enumerate() {
        layer_step(
            temporal,
            scratch,
            (lw, packed.map(|p| &p.layers[l])),
            st,
            cfg,
        );
        if let Some(mlp) = weights.mlp.get(l) {
            let t = &mut temporal[..dm];
            match packed {
                Some(p) => {
                    gated_mlp_step_packed(t, (mlp, &p.mlp[l]), &mut scratch.mlp, cfg.rms_norm_eps)
                }
                None => gated_mlp_step(t, mlp, &mut scratch.mlp, cfg.rms_norm_eps),
            }
        }
    }
    scratch.norm_buf.copy_from_slice(&temporal[..dm]);
    rms_row(
        &mut temporal[..dm],
        &scratch.norm_buf,
        &weights.norm_f_weight,
        cfg.rms_norm_eps,
    );
}
