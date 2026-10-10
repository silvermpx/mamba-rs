//! Mamba-3 MIMO decode: one token through the backbone, mutating the
//! recurrent state.
//!
//! ```text
//! temporal -> RMSNorm -> in_proj -> split [z | x | B | C | dd_dt | dd_A | trap | angles]
//!   -> B/C RMSNorm per (rank, group) row
//!   -> per head: discretise, advance angles, bias + half-split rotate R keys and queries,
//!      u_r = mimo_x_r * x, state rows, y_r += D u_r, gate and mix the ranks through mimo_o
//!   -> out_proj + residual [-> gated MLP] -> ... -> final RMSNorm
//! ```

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use super::core::{HeadScratch, HeadState, mimo_head_step, prepare_token, row_norm};
use crate::mlp::cpu::{GatedMlpPacked, GatedMlpStepScratch, gated_mlp_step, gated_mlp_step_packed};
use crate::ops::packed::{PackedLinear, linear_row};

/// One layer's projections packed for decode.
#[derive(Clone, Debug)]
pub struct Mamba3MimoLayerPacked {
    pub in_proj: PackedLinear,
    pub out_proj: PackedLinear,
}

/// Every projection of a MIMO backbone packed for decode: a copy of the
/// weights' projections, so it must be packed again after they change.
#[derive(Clone, Debug)]
pub struct Mamba3MimoPackedLinears {
    pub layers: Vec<Mamba3MimoLayerPacked>,
    /// One per layer with an MLP.
    pub mlp: Vec<GatedMlpPacked>,
    /// The input projection, absent for a backbone without one.
    pub input_proj: Option<PackedLinear>,
}

impl Mamba3MimoPackedLinears {
    /// Pack every projection of `weights`.
    pub fn pack(weights: &Mamba3MimoWeights, cfg: &Mamba3MimoConfig, input_dim: usize) -> Self {
        let (dm, di) = (cfg.d_model, cfg.d_inner());
        Self {
            layers: weights
                .layers
                .iter()
                .map(|lw| Mamba3MimoLayerPacked {
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

/// Reusable buffers for [`mamba3_mimo_step`].
pub struct Mamba3MimoStepScratch {
    residual: Vec<f32>,
    norm_buf: Vec<f32>,
    proj: Vec<f32>,
    /// Normalised B and C, `[R * G * N]` each.
    bn: Vec<f32>,
    cn: Vec<f32>,
    /// The gated output of every head, `[d_inner]`.
    o: Vec<f32>,
    tanh_pi: Vec<f32>,
    head: HeadScratch,
    mlp: GatedMlpStepScratch,
}

impl Mamba3MimoStepScratch {
    pub fn new(cfg: &Mamba3MimoConfig) -> Self {
        let dm = cfg.d_model;
        Self {
            residual: vec![0.0; dm],
            norm_buf: vec![0.0; dm],
            proj: vec![0.0; cfg.in_proj_out_dim()],
            bn: vec![0.0; cfg.bc_width()],
            cn: vec![0.0; cfg.bc_width()],
            o: vec![0.0; cfg.d_inner()],
            tanh_pi: vec![0.0; cfg.num_rope_angles()],
            head: HeadScratch::new(cfg),
            mlp: GatedMlpStepScratch::new(dm, cfg.mlp_hidden()),
        }
    }
}

/// One token through one MIMO mixer; `temporal` (`[d_model]`) is both the
/// input and the output, residual included.
pub fn mamba3_mimo_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba3MimoStepScratch,
    lw: &Mamba3MimoLayerWeights,
    state: &mut Mamba3MimoLayerState,
    cfg: &Mamba3MimoConfig,
) {
    layer_step(temporal, scratch, (lw, None), state, cfg);
}

/// [`mamba3_mimo_layer_step`] with the projections packed when `lp` holds
/// them.
fn layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba3MimoStepScratch,
    (lw, lp): (&Mamba3MimoLayerWeights, Option<&Mamba3MimoLayerPacked>),
    state: &mut Mamba3MimoLayerState,
    cfg: &Mamba3MimoConfig,
) {
    let (dm, di, n, p_len, r) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
    );
    let na = cfg.num_rope_angles();

    scratch.residual.copy_from_slice(&temporal[..dm]);
    row_norm(
        &mut scratch.norm_buf,
        &temporal[..dm],
        &lw.norm_weight,
        cfg.rms_norm_eps,
    );
    linear_row(
        &mut scratch.proj,
        &scratch.norm_buf,
        (&lw.in_proj_w, lp.map(|p| &p.in_proj)),
        None,
        (dm, cfg.in_proj_out_dim()),
    );
    let tok = prepare_token(
        cfg,
        lw,
        &scratch.proj,
        (&mut scratch.bn, &mut scratch.cn, &mut scratch.tanh_pi),
    );
    for h in 0..cfg.nheads() {
        mimo_head_step(
            (cfg, lw, h),
            &tok,
            HeadState {
                ssm: &mut state.ssm_state[h * p_len * n..(h + 1) * p_len * n],
                k: &mut state.k_state[h * r * n..(h + 1) * r * n],
                v: &mut state.v_state[h * p_len..(h + 1) * p_len],
                angle: &mut state.angle_state[h * na..(h + 1) * na],
            },
            &mut scratch.head,
            &mut scratch.o[h * p_len..(h + 1) * p_len],
        );
    }

    linear_row(
        &mut temporal[..dm],
        &scratch.o,
        (&lw.out_proj_w, lp.map(|p| &p.out_proj)),
        None,
        (di, dm),
    );
    for (t, &res) in temporal[..dm].iter_mut().zip(&scratch.residual) {
        *t += res;
    }
}

/// One token through the whole MIMO backbone: input projection, every
/// layer (mixer, then the MLP when the model has one) and the final norm.
/// `input` is `[input_dim]`; `temporal` (`[d_model]`) receives the output.
pub fn mamba3_mimo_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba3MimoStepScratch,
    weights: &Mamba3MimoWeights,
    states: &mut [Mamba3MimoLayerState],
    cfg: &Mamba3MimoConfig,
) {
    backbone_step(temporal, input, scratch, (weights, None), states, cfg);
}

/// [`mamba3_mimo_step`] with every projection taken from `packed` (packed
/// from these `weights` by [`Mamba3MimoPackedLinears::pack`]).
///
/// # Panics
///
/// When `packed` holds another number of layers or MLPs than `weights`.
pub fn mamba3_mimo_step_packed(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba3MimoStepScratch,
    (weights, packed): (&Mamba3MimoWeights, &Mamba3MimoPackedLinears),
    states: &mut [Mamba3MimoLayerState],
    cfg: &Mamba3MimoConfig,
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
    scratch: &mut Mamba3MimoStepScratch,
    (weights, packed): (&Mamba3MimoWeights, Option<&Mamba3MimoPackedLinears>),
    states: &mut [Mamba3MimoLayerState],
    cfg: &Mamba3MimoConfig,
) {
    let dm = cfg.d_model;
    if weights.input_proj_w.is_empty() {
        temporal[..dm].copy_from_slice(&input[..dm]);
    } else {
        linear_row(
            &mut temporal[..dm],
            input,
            (
                &weights.input_proj_w,
                packed.and_then(|p| p.input_proj.as_ref()),
            ),
            Some(&weights.input_proj_b),
            (input.len(), dm),
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
    row_norm(
        &mut temporal[..dm],
        &scratch.norm_buf,
        &weights.norm_f_weight,
        cfg.rms_norm_eps,
    );
}
