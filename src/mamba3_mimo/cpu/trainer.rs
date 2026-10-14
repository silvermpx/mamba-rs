//! The Mamba-3 MIMO host trainer: the shared CPU trainer over this
//! family's window forward and backward ([`forward_mamba3_mimo_train`],
//! [`backward_mamba3_mimo_train`]).

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::Mamba3MimoWeights;
use super::train::{
    Mamba3MimoTape, Mamba3MimoTrainScratch, backward_mamba3_mimo_train, forward_mamba3_mimo_train,
};
use crate::config::ModelFamily;
use crate::module::cpu_trainer::{Control, CpuCore, CpuFamily, TensorRole, cpu_trainer_type};
use crate::optim::session::TrainSessionCfg;
use crate::optim::tensors::tensor_list;
use crate::precision::{CpuOptions, WeightDtype};

/// Positions between the tape's state checkpoints: the backward replays a
/// head's recurrence over at most this many steps from a stored state.
const TAPE_STRIDE: usize = 8;

tensor_list!(
    Mamba3MimoWeights,
    pre: [input_proj_w, input_proj_b],
    layers: layers [
        norm_weight,
        norm_bias,
        in_proj_w,
        dt_bias,
        b_norm_weight,
        c_norm_weight,
        b_bias,
        c_bias,
        d_param,
        mimo_x,
        mimo_z,
        mimo_o,
        out_proj_w,
        norm_gate_weight
    ],
    mlp: mlp [norm_weight, norm_bias, fc1_w, fc2_w],
    post: [norm_f_weight, norm_f_bias],
);

pub(crate) struct Mamba3MimoCpu {
    cfg: Mamba3MimoConfig,
    input_dim: usize,
    dtype: WeightDtype,
}

/// One sequence's carried states, tape and scratch.
pub(crate) struct Mamba3MimoCpuSeq {
    states: Vec<Mamba3MimoLayerState>,
    tape: Mamba3MimoTape,
    scratch: Mamba3MimoTrainScratch,
}

impl Mamba3MimoCpu {
    /// One layer's recurrence tensor `i` (ssm, k, v, angle).
    fn part(st: &Mamba3MimoLayerState, i: usize) -> &[f32] {
        match i {
            0 => &st.ssm_state,
            1 => &st.k_state,
            2 => &st.v_state,
            _ => &st.angle_state,
        }
    }

    fn part_mut(st: &mut Mamba3MimoLayerState, i: usize) -> &mut [f32] {
        match i {
            0 => &mut st.ssm_state,
            1 => &mut st.k_state,
            2 => &mut st.v_state,
            _ => &mut st.angle_state,
        }
    }
}

impl CpuFamily for Mamba3MimoCpu {
    type Weights = Mamba3MimoWeights;
    type Seq = Mamba3MimoCpuSeq;
    const FAMILY: ModelFamily = ModelFamily::Mamba3Mimo;
    const RECURRENCE: &'static [&'static str] = &["ssm", "k", "v", "angle"];

    fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The GPU trainer's half mirror: in_proj, out_proj and the MLP's two
    /// projections; TF32 rounds the input projection's operands too.
    fn round_weights(&self, w: &mut Mamba3MimoWeights) {
        w.round_bulk(self.dtype);
    }

    fn new_seq(&self, _w: &Mamba3MimoWeights, seq_len: usize) -> Result<Mamba3MimoCpuSeq, String> {
        Ok(Mamba3MimoCpuSeq {
            states: (0..self.cfg.n_layers)
                .map(|_| Mamba3MimoLayerState::zeros(&self.cfg))
                .collect(),
            tape: Mamba3MimoTape::new(&self.cfg, seq_len, self.input_dim, TAPE_STRIDE)
                .with_dtype(self.dtype),
            scratch: Mamba3MimoTrainScratch::default(),
        })
    }

    fn reset_seq(&self, seq: &mut Mamba3MimoCpuSeq) {
        seq.states.iter_mut().for_each(Mamba3MimoLayerState::reset);
    }

    fn set_varlen(&self, seq: &mut Mamba3MimoCpuSeq, ids: Option<&[i32]>) -> Result<(), String> {
        seq.tape.set_varlen(ids);
        Ok(())
    }

    fn forward(
        &self,
        w: &Mamba3MimoWeights,
        seq: &mut Mamba3MimoCpuSeq,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        forward_mamba3_mimo_train(
            out,
            input,
            w,
            &mut seq.states,
            &mut seq.tape,
            (&self.cfg, &mut seq.scratch),
        )
    }

    fn backward(
        &self,
        w: &Mamba3MimoWeights,
        seq: &mut Mamba3MimoCpuSeq,
        grads: &mut Mamba3MimoWeights,
        d_out: &mut [f32],
        d_input: &mut [f32],
    ) -> Result<(), String> {
        backward_mamba3_mimo_train(
            d_out,
            d_input,
            (w, grads),
            &seq.tape,
            (&self.cfg, &mut seq.scratch),
        )
    }

    fn zeros_like(w: &Mamba3MimoWeights) -> Mamba3MimoWeights {
        w.zeros_like()
    }

    /// The matrices decay; the control channels are SISO's: the
    /// dt/A/trapezoid/angle columns of in_proj and dt_bias, as on the GPU.
    fn roles(&self, w: &Mamba3MimoWeights) -> Vec<TensorRole> {
        let ip = self.cfg.in_proj_out_dim();
        let ctrl = 3 * self.cfg.nheads() + self.cfg.num_rope_angles();
        let plain = |decays| TensorRole {
            decays,
            control: Control::None,
        };
        let mut roles = vec![plain(true), plain(false)];
        for _ in &w.layers {
            roles.extend([
                plain(false),
                plain(false),
                TensorRole {
                    decays: true,
                    control: Control::Columns {
                        row_len: ip,
                        cols: ip - ctrl..ip,
                    },
                },
                TensorRole {
                    decays: false,
                    control: Control::All,
                },
                plain(false),
                plain(false),
                plain(false),
                plain(false),
                plain(false),
                plain(true),
                plain(true),
                plain(true),
                plain(true),
                plain(false),
            ]);
        }
        for _ in &w.mlp {
            roles.extend([plain(false), plain(false), plain(true), plain(true)]);
        }
        roles.extend([plain(false), plain(false)]);
        roles
    }

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    fn recurrence_len(&self, i: usize) -> usize {
        let c = &self.cfg;
        let (nh, p, n) = (c.nheads(), c.headdim, c.d_state);
        match i {
            0 => nh * p * n,
            1 => nh * c.mimo_rank * n,
            2 => nh * p,
            _ => nh * c.num_rope_angles(),
        }
    }

    /// The host keeps each layer's four states as the GPU does.
    fn export_recurrence(&self, seq: &Mamba3MimoCpuSeq, i: usize, l: usize, out: &mut Vec<f32>) {
        out.extend_from_slice(Self::part(&seq.states[l], i));
    }

    fn import_recurrence(&self, seq: &mut Mamba3MimoCpuSeq, i: usize, l: usize, values: &[f32]) {
        Self::part_mut(&mut seq.states[l], i).copy_from_slice(values);
    }
}

cpu_trainer_type!(
    /// A Mamba-3 MIMO model trained on the CPU: the batch's sequences side
    /// by side, the accumulation window, the clips, the spike skip and
    /// AdamW, with the surface of [`crate::module::CpuTrainer`]. The
    /// gradients of the batch sum in a fixed number of chunks reduced in a
    /// fixed tree, so the bits do not depend on the thread pool.
    Mamba3MimoCpuTrainer,
    Mamba3MimoCpu
);

impl Mamba3MimoCpuTrainer {
    /// A trainer of `weights` (copied as the f32 master) for the session's
    /// batch and window, with its AdamW settings.
    pub fn new(
        weights: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        Self::new_with(weights, cfg, session, CpuOptions::default())
    }

    /// [`Mamba3MimoCpuTrainer::new`] in the precision `opts` names: f32
    /// masters, the bulk matrices and the values between the kernels
    /// rounded as the GPU trainer of that precision rounds them, and in
    /// f16 the dynamic loss scaler.
    pub fn new_with(
        weights: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        session: TrainSessionCfg,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        cfg.validate()?;
        weights.validate(&cfg, session.input_dim)?;
        let fam = Mamba3MimoCpu {
            cfg,
            input_dim: session.input_dim,
            dtype: opts.dtype,
        };
        Ok(Self {
            core: CpuCore::new(fam, weights.clone(), session, cfg.d_model)?,
        })
    }

    /// The master weights, for a checkpoint or the inference model.
    pub fn snapshot_master(&self) -> Mamba3MimoWeights {
        self.core.weights.clone()
    }
}
