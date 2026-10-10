//! Mamba-2 backbone wrapper (CPU), the Mamba-2 peer of
//! [`crate::module::MambaBackbone`]: inference, single steps and windows.

use crate::mamba2::config::Mamba2Config;
use crate::mamba2::cpu::inference::{
    Mamba2PackedLinears, Mamba2StepScratch, mamba2_step, mamba2_step_packed,
};
use crate::mamba2::cpu::prefill::{Mamba2PrefillScratch, forward_mamba2_backbone_prefill};
use crate::mamba2::state::{Mamba2LayerState, Mamba2State};
use crate::mamba2::weights::Mamba2Weights;

/// Mamba-2 backbone for CPU inference. The recurrent state is owned by the
/// caller, so several sessions can share one set of weights.
pub struct Mamba2Backbone {
    weights: Mamba2Weights,
    cfg: Mamba2Config,
    /// The projections packed for decode by [`Self::pack_for_decode`]; the
    /// weights never change after construction, so the copy stays current.
    packed: Option<Mamba2PackedLinears>,
}

impl Mamba2Backbone {
    /// Fresh weights with the reference init.
    pub fn init(cfg: Mamba2Config, input_dim: usize, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            weights: Mamba2Weights::init(&cfg, input_dim, seed),
            cfg,
            packed: None,
        })
    }

    /// Wrap loaded weights after checking every tensor against `cfg`.
    pub fn from_weights(
        weights: Mamba2Weights,
        cfg: Mamba2Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        Ok(Self {
            weights,
            cfg,
            packed: None,
        })
    }

    pub fn weights(&self) -> &Mamba2Weights {
        &self.weights
    }

    pub fn config(&self) -> &Mamba2Config {
        &self.cfg
    }

    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// Pack the projections in panels for [`Self::forward_step`]: the same
    /// outputs as the row-major product without a BLAS, faster on large
    /// layers, at the memory of a second copy of the projections.
    pub fn pack_for_decode(&mut self) {
        let input_dim = self.weights.input_proj_w.len() / self.cfg.d_model.max(1);
        self.packed = Some(Mamba2PackedLinears::pack(
            &self.weights,
            &self.cfg,
            input_dim,
        ));
    }

    /// Whether the projections are packed for decode.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    /// One token: reads `input`, writes `temporal[..d_model]` and advances
    /// every layer's state.
    pub fn forward_step(
        &self,
        input: &[f32],
        temporal: &mut [f32],
        scratch: &mut Mamba2StepScratch,
        states: &mut [Mamba2LayerState],
    ) {
        match &self.packed {
            Some(p) => mamba2_step_packed(
                temporal,
                input,
                scratch,
                (&self.weights, p),
                states,
                &self.cfg,
            ),
            None => mamba2_step(temporal, input, scratch, &self.weights, states, &self.cfg),
        }
    }

    /// A window of `seq_len` tokens in one pass, the decode chain's bits on
    /// every row.
    pub fn forward_sequence(
        &self,
        (inputs, outputs, seq_len): (&[f32], &mut [f32], usize),
        scratch: &mut Mamba2PrefillScratch,
        states: &mut [Mamba2LayerState],
    ) {
        forward_mamba2_backbone_prefill(
            outputs,
            inputs,
            &self.weights,
            states,
            scratch,
            (&self.cfg, seq_len),
        );
    }

    pub fn alloc_state(&self) -> Mamba2State {
        Mamba2State::zeros(&self.cfg)
    }

    /// Buffers for single steps and for windows.
    pub fn alloc_scratch(&self) -> Mamba2Scratch {
        Mamba2Scratch {
            step: Mamba2StepScratch::new(&self.cfg),
            prefill: Mamba2PrefillScratch::new(),
        }
    }
}

/// The step and window buffers of one Mamba-2 session.
pub struct Mamba2Scratch {
    pub step: Mamba2StepScratch,
    pub prefill: Mamba2PrefillScratch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
            })
            .collect()
    }

    const NO_BLAS: bool = cfg!(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )));

    /// Packed projections (the mixers', the MLPs' and the input's) give
    /// the row-major step's bits over a chain of steps, without a BLAS.
    #[test]
    fn packed_decode_keeps_the_bits() {
        let cfg = Mamba2Config {
            d_model: 96,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            d_intermediate: 200,
            ..Mamba2Config::default()
        };
        let input_dim = 40;
        let plain = Mamba2Backbone::init(cfg, input_dim, 7).unwrap();
        let mut packed = Mamba2Backbone::init(cfg, input_dim, 7).unwrap();
        packed.pack_for_decode();
        assert!(packed.is_packed());
        let (mut s0, mut s1) = (plain.alloc_state(), packed.alloc_state());
        let (mut c0, mut c1) = (plain.alloc_scratch(), packed.alloc_scratch());
        let (mut o0, mut o1) = (vec![0f32; cfg.d_model], vec![0f32; cfg.d_model]);
        for t in 0..12u64 {
            let x = det(input_dim, 100 + t);
            plain.forward_step(&x, &mut o0, &mut c0.step, &mut s0.layers);
            packed.forward_step(&x, &mut o1, &mut c1.step, &mut s1.layers);
            if NO_BLAS {
                assert_eq!(o0, o1, "step {t}");
            }
        }
    }
}
