//! The Mamba-1 host trainer: the shared CPU trainer over this family's
//! backbone forward and backward ([`forward_mamba_backbone_batched`],
//! [`backward_mamba_backbone_batched`]).

use super::backward::backward_mamba_backbone_batched;
use super::flat::MambaBackboneFlat;
use super::forward::forward_mamba_backbone_batched;
use super::scratch::{BackwardPhaseScratch, PhaseScratch};
use super::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use crate::config::ModelFamily;
use crate::mamba1::config::MambaConfig;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::module::cpu_trainer::{Control, CpuCore, CpuFamily, TensorRole, cpu_trainer_type};
use crate::ops::dims::{MambaDims, MambaRecurrentState};
use crate::optim::session::TrainSessionCfg;
use crate::optim::tensors::tensor_list;
use crate::precision::{CpuOptions, WeightDtype};

tensor_list!(
    TrainMambaWeights,
    pre: [input_proj_w, input_proj_b],
    layers: layers [
        norm_weight,
        norm_bias,
        in_proj_w,
        in_proj_b,
        conv1d_weight,
        conv1d_bias,
        x_proj_w,
        dt_proj_w,
        dt_proj_b,
        a_log,
        d_param,
        out_proj_w,
        out_proj_b
    ],
    mlp: mlp [norm_weight, norm_bias, fc1_w, fc2_w],
    post: [norm_f_weight, norm_f_bias],
);

pub(crate) struct MambaCpu {
    cfg: MambaConfig,
    dims: MambaDims,
    /// `-exp(a_log)` of every layer, which the scans read; refreshed after
    /// every optimizer step.
    a_neg: Vec<f32>,
    dtype: WeightDtype,
}

/// One sequence's carried conv and SSM states, activations and scratch.
pub(crate) struct MambaCpuSeq {
    acts: MambaBackboneFlat,
    conv: Vec<f32>,
    ssm: Vec<f32>,
    fwd: PhaseScratch,
    bwd: BackwardPhaseScratch,
}

impl CpuFamily for MambaCpu {
    type Weights = TrainMambaWeights;
    type Seq = MambaCpuSeq;
    const FAMILY: ModelFamily = ModelFamily::Mamba1;
    const RECURRENCE: &'static [&'static str] = &["conv", "ssm"];

    fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The GPU trainer's mirror: every projection, the input one too.
    fn round_weights(&self, w: &mut TrainMambaWeights) {
        w.round_bulk(self.dtype);
    }

    fn new_seq(&self, _w: &TrainMambaWeights, _seq_len: usize) -> Result<MambaCpuSeq, String> {
        let mut fwd = PhaseScratch::zeros(&self.dims);
        let mut bwd = BackwardPhaseScratch::zeros(&self.dims);
        (fwd.dtype, bwd.dtype) = (self.dtype, self.dtype);
        Ok(MambaCpuSeq {
            acts: MambaBackboneFlat::zeros(self.dims),
            conv: vec![0.0; self.cfg.n_layers * self.recurrence_len(0)],
            ssm: vec![0.0; self.cfg.n_layers * self.recurrence_len(1)],
            fwd,
            bwd,
        })
    }

    fn reset_seq(&self, seq: &mut MambaCpuSeq) {
        seq.conv.fill(0.0);
        seq.ssm.fill(0.0);
    }

    fn forward(
        &self,
        w: &TrainMambaWeights,
        seq: &mut MambaCpuSeq,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        // A bidirectional model reads whole sequences: every window starts
        // from the empty state.
        if self.cfg.bidirectional {
            self.reset_seq(seq);
        }
        out.fill(0.0);
        forward_mamba_backbone_batched(
            out,
            &mut seq.acts,
            w,
            input,
            &mut MambaRecurrentState {
                conv: &mut seq.conv,
                ssm: &mut seq.ssm,
                a_neg: &self.a_neg,
            },
            &mut seq.fwd,
            &self.dims,
        );
        Ok(())
    }

    fn backward(
        &self,
        w: &TrainMambaWeights,
        seq: &mut MambaCpuSeq,
        grads: &mut TrainMambaWeights,
        d_out: &mut [f32],
        d_input: &mut [f32],
    ) -> Result<(), String> {
        backward_mamba_backbone_batched(
            d_out,
            grads,
            &seq.acts,
            w,
            &self.a_neg,
            &mut seq.bwd,
            &self.dims,
        );
        // Without an input projection the rows the layers read are the
        // input, so their gradient is the input's; with one, the
        // projection's backward left it in the scratch.
        let n = d_input.len();
        if w.input_proj_w.is_empty() {
            d_input.copy_from_slice(&d_out[..n]);
        } else {
            d_input.copy_from_slice(&seq.bwd.d_input_proj_scratch[..n]);
        }
        Ok(())
    }

    fn zeros_like(w: &TrainMambaWeights) -> TrainMambaWeights {
        let mut g = w.clone();
        crate::optim::tensors::TensorList::zero(&mut g);
        g
    }

    /// The matrices decay; the reference training's no-decay group does
    /// not: what the mixer marks `_no_weight_decay` (A_log, D), every
    /// bias and every norm scale, as the Mamba-2 and Mamba-3 walks group
    /// them. The control
    /// channels are the step-size columns of x_proj, dt_proj and A_log, as
    /// on the GPU. A bias the config lacks is an empty tensor with its
    /// role, so every other tensor keeps its place. The gated MLPs' norms
    /// do not decay and their projections do.
    fn roles(&self, w: &TrainMambaWeights) -> Vec<TensorRole> {
        let (dt_rank, xdbl) = (self.cfg.dt_rank(), self.cfg.xdbl_dim());
        let role = |decays, control| TensorRole { decays, control };
        let mut roles = vec![role(true, Control::None), role(false, Control::None)];
        for _ in &w.layers {
            roles.extend([
                role(false, Control::None),
                role(false, Control::None),
                role(true, Control::None),
                role(false, Control::None),
                role(true, Control::None),
                role(false, Control::None),
                role(
                    true,
                    Control::Columns {
                        row_len: xdbl,
                        cols: 0..dt_rank,
                    },
                ),
                role(true, Control::All),
                role(false, Control::All),
                role(false, Control::All),
                role(false, Control::None),
                role(true, Control::None),
                role(false, Control::None),
            ]);
        }
        for _ in &w.mlp {
            roles.extend([
                role(false, Control::None),
                role(false, Control::None),
                role(true, Control::None),
                role(true, Control::None),
            ]);
        }
        roles.extend([role(false, Control::None), role(false, Control::None)]);
        roles
    }

    fn after_update(&mut self, w: &TrainMambaWeights) {
        let mut at = 0;
        for lw in &w.layers {
            for (a, &log) in self.a_neg[at..at + lw.a_log.len()]
                .iter_mut()
                .zip(&lw.a_log)
            {
                *a = -log.exp();
            }
            at += lw.a_log.len();
        }
    }

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    fn recurrence_len(&self, i: usize) -> usize {
        let c = &self.cfg;
        match i {
            0 => c.d_inner() * c.d_conv,
            _ => c.d_inner() * c.d_state,
        }
    }

    /// The host keeps both states as the GPU does, layer after layer.
    fn export_recurrence(&self, seq: &MambaCpuSeq, i: usize, l: usize, out: &mut Vec<f32>) {
        let n = self.recurrence_len(i);
        let all = if i == 0 { &seq.conv } else { &seq.ssm };
        out.extend_from_slice(&all[l * n..(l + 1) * n]);
    }

    fn import_recurrence(&self, seq: &mut MambaCpuSeq, i: usize, l: usize, values: &[f32]) {
        let n = self.recurrence_len(i);
        let all = if i == 0 { &mut seq.conv } else { &mut seq.ssm };
        all[l * n..(l + 1) * n].copy_from_slice(values);
    }
}

cpu_trainer_type!(
    /// A Mamba-1 model trained on the CPU: the batch's sequences side by
    /// side, the accumulation window, the clips, the spike skip and AdamW,
    /// with the surface of [`crate::module::CpuTrainer`]. The gradients of
    /// the batch sum in a fixed number of chunks reduced in a fixed tree,
    /// so the bits do not depend on the thread pool.
    MambaCpuTrainer,
    MambaCpu
);

impl MambaCpuTrainer {
    /// A trainer of `weights` (copied as the f32 master) for the session's
    /// batch and window, with its AdamW settings.
    pub fn new(
        weights: &MambaWeights,
        cfg: MambaConfig,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        Self::new_with(weights, cfg, session, CpuOptions::default())
    }

    /// [`MambaCpuTrainer::new`] in the precision `opts` names: f32
    /// masters, the projections and the values between the kernels
    /// rounded as the GPU trainer of that precision rounds them, and in
    /// f16 the dynamic loss scaler.
    pub fn new_with(
        weights: &MambaWeights,
        cfg: MambaConfig,
        session: TrainSessionCfg,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        cfg.validate()?;
        weights.validate(&cfg, session.input_dim)?;
        let train = TrainMambaWeights {
            input_proj_w: weights.input_proj_w.clone(),
            input_proj_b: weights.input_proj_b.clone(),
            layers: weights
                .layers
                .iter()
                .map(|lw| TrainMambaLayerWeights {
                    norm_weight: lw.norm_weight.clone(),
                    norm_bias: lw.norm_bias.clone(),
                    in_proj_w: lw.in_proj_w.clone(),
                    in_proj_b: lw.in_proj_b.clone(),
                    conv1d_weight: lw.conv1d_weight.clone(),
                    conv1d_bias: lw.conv1d_bias.clone(),
                    x_proj_w: lw.x_proj_w.clone(),
                    dt_proj_w: lw.dt_proj_w.clone(),
                    dt_proj_b: lw.dt_proj_b.clone(),
                    a_log: lw.a_log.clone(),
                    d_param: lw.d_param.clone(),
                    out_proj_w: lw.out_proj_w.clone(),
                    out_proj_b: lw.out_proj_b.clone(),
                })
                .collect(),
            norm_f_weight: weights.norm_f_weight.clone(),
            norm_f_bias: weights.norm_f_bias.clone(),
            mlp: weights.mlp.clone(),
        };
        let dims = MambaDims::from_config(&cfg, session.seq_len, session.input_dim);
        let mut fam = MambaCpu {
            cfg,
            dims,
            a_neg: vec![0.0; cfg.n_layers * cfg.d_inner() * cfg.d_state],
            dtype: opts.dtype,
        };
        fam.after_update(&train);
        Ok(Self {
            core: CpuCore::new(fam, train, session, cfg.d_model)?,
        })
    }

    /// The master weights, for a checkpoint or the inference model.
    pub fn snapshot_master(&self) -> MambaWeights {
        let w = &self.core.weights;
        MambaWeights {
            input_proj_w: w.input_proj_w.clone(),
            input_proj_b: w.input_proj_b.clone(),
            layers: w
                .layers
                .iter()
                .map(|lw| {
                    let mut out = MambaLayerWeights {
                        norm_weight: lw.norm_weight.clone(),
                        norm_bias: lw.norm_bias.clone(),
                        in_proj_w: lw.in_proj_w.clone(),
                        in_proj_b: lw.in_proj_b.clone(),
                        conv1d_weight: lw.conv1d_weight.clone(),
                        conv1d_bias: lw.conv1d_bias.clone(),
                        x_proj_w: lw.x_proj_w.clone(),
                        dt_proj_w: lw.dt_proj_w.clone(),
                        dt_proj_b: lw.dt_proj_b.clone(),
                        a_log: lw.a_log.clone(),
                        a_neg: vec![0.0; lw.a_log.len()],
                        d_param: lw.d_param.clone(),
                        out_proj_w: lw.out_proj_w.clone(),
                        out_proj_b: lw.out_proj_b.clone(),
                    };
                    out.compute_a_neg();
                    out
                })
                .collect(),
            norm_f_weight: w.norm_f_weight.clone(),
            norm_f_bias: w.norm_f_bias.clone(),
            mlp: w.mlp.clone(),
        }
    }
}
