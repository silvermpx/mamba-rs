//! Mamba-3 MIMO backbone wrapper (CPU), the MIMO peer of
//! [`crate::module::Mamba3Backbone`]: inference, single steps and windows.

use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::cpu::inference::{
    Mamba3MimoPackedLinears, Mamba3MimoStepScratch, mamba3_mimo_step, mamba3_mimo_step_packed,
};
use crate::mamba3_mimo::cpu::prefill::{
    Mamba3MimoPrefillScratch, forward_mamba3_mimo_backbone_prefill,
};
use crate::mamba3_mimo::state::Mamba3MimoState;
use crate::mamba3_mimo::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use crate::precision::{CpuOptions, WeightDtype};

/// Mamba-3 MIMO backbone for CPU inference. The recurrent state is owned
/// by the caller, so several sessions can share one set of weights.
pub struct Mamba3MimoBackbone {
    weights: Mamba3MimoWeights,
    cfg: Mamba3MimoConfig,
    /// The projections packed for decode by [`Self::pack_for_decode`]: a
    /// copy of `weights`, dropped by every mutable access to them so a step
    /// never reads stale panels.
    packed: Option<Mamba3MimoPackedLinears>,
    /// Width of an input row.
    input_dim: usize,
    /// The precision the backbone computes in.
    dtype: WeightDtype,
}

impl Mamba3MimoBackbone {
    /// Fresh weights with the reference init.
    pub fn init(cfg: Mamba3MimoConfig, input_dim: usize, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            weights: Mamba3MimoWeights::init(&cfg, input_dim, seed),
            cfg,
            packed: None,
            input_dim,
            dtype: WeightDtype::F32,
        })
    }

    /// Wrap loaded weights for input rows `input_dim` wide, after checking
    /// every tensor against `cfg`.
    pub fn new(
        weights: Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
    ) -> Result<Self, String> {
        Self::new_with(weights, cfg, input_dim, CpuOptions::default())
    }

    /// [`Mamba3MimoBackbone::new`] in the precision `opts` names: the bulk
    /// matrices rounded once here as the GPU engine of that precision
    /// holds them, and the values between the kernels rounded where it
    /// stores them. [`Mamba3MimoBackbone::weights`] then gives the rounded
    /// weights, the ones the backbone computes with.
    pub fn new_with(
        mut weights: Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        weights.round_bulk(opts.dtype);
        Ok(Self {
            weights,
            cfg,
            packed: None,
            input_dim,
            dtype: opts.dtype,
        })
    }

    /// The precision the backbone computes in.
    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The weights.
    pub fn weights(&self) -> &Mamba3MimoWeights {
        &self.weights
    }

    /// The weights, consuming the backbone.
    pub fn into_weights(self) -> Mamba3MimoWeights {
        self.weights
    }

    /// The weights, to change in place. Drops the packed projections; pack
    /// again with [`Self::pack_for_decode`]. In a reduced precision the
    /// backbone computes with the values written here as they are;
    /// [`Self::new_with`] rounds a set of weights as the GPU engine of that
    /// precision holds them.
    pub fn weights_mut(&mut self) -> &mut Mamba3MimoWeights {
        self.packed = None;
        &mut self.weights
    }

    /// One layer's weights.
    pub fn layer(&self, index: usize) -> &Mamba3MimoLayerWeights {
        &self.weights.layers[index]
    }

    /// One layer's weights, to change in place, as with
    /// [`Self::weights_mut`].
    pub fn layer_mut(&mut self, index: usize) -> &mut Mamba3MimoLayerWeights {
        self.packed = None;
        &mut self.weights.layers[index]
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba3MimoConfig {
        &self.cfg
    }

    /// Width of an input row.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    /// Total parameter count: every tensor the family's checkpoint holds.
    pub fn param_count(&self) -> usize {
        // A built backbone's weights always pass the checkpoint's checks.
        crate::mamba3_mimo::serialize::checkpoint_parts(&self.weights, &self.cfg, self.input_dim)
            .map_or(0, |(list, _)| list.element_count())
    }

    /// Width of an output row.
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// Pack the projections in panels for [`Self::forward_step`]: the same
    /// outputs as the row-major product without a BLAS, faster on large
    /// layers, at the memory of a second copy of the projections.
    pub fn pack_for_decode(&mut self) {
        let input_dim = self.weights.input_proj_w.len() / self.cfg.d_model.max(1);
        self.packed = Some(Mamba3MimoPackedLinears::pack(
            &self.weights,
            &self.cfg,
            input_dim,
        ));
    }

    /// Whether the projections are packed for decode.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    /// One token: reads `input` (`input_dim` wide), writes `output`
    /// (`d_model` wide) and advances every layer's state.
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut Mamba3MimoState,
        scratch: &mut Mamba3MimoScratch,
    ) {
        let (states, scratch) = (&mut state.layers, &mut scratch.step);
        scratch.set_dtype(self.dtype);
        match &self.packed {
            Some(p) => mamba3_mimo_step_packed(
                output,
                input,
                scratch,
                (&self.weights, p),
                states,
                &self.cfg,
            ),
            None => mamba3_mimo_step(output, input, scratch, &self.weights, states, &self.cfg),
        }
    }

    /// A window of `seq_len` tokens in one pass (`inputs` is
    /// `[seq_len, input_dim]`, `outputs` `[seq_len, d_model]`), the decode
    /// chain's bits on every row.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut Mamba3MimoState,
        scratch: &mut Mamba3MimoScratch,
        seq_len: usize,
    ) {
        let (states, scratch) = (&mut state.layers, &mut scratch.prefill);
        scratch.set_dtype(self.dtype);
        forward_mamba3_mimo_backbone_prefill(
            outputs,
            inputs,
            &self.weights,
            states,
            scratch,
            (&self.cfg, seq_len),
        );
    }

    /// One token of `B` independent sequences with the same weights, each
    /// through [`Self::forward_step`]: `inputs` is `[B * input_dim]`,
    /// `outputs` `[B * d_model]`, one state and one scratch a sequence.
    pub fn forward_step_batch(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        states: &mut [Mamba3MimoState],
        scratches: &mut [Mamba3MimoScratch],
    ) {
        crate::ops::step_batch::step_batch_by(
            (inputs, outputs),
            (states, scratches),
            (self.cfg.d_model, self.input_dim),
            |inp, out, state, scratch| self.forward_step(inp, out, state, scratch),
        );
    }

    /// A zeroed recurrent state, the start of a sequence.
    pub fn alloc_state(&self) -> Mamba3MimoState {
        Mamba3MimoState::zeros(&self.cfg)
    }

    /// Buffers for single steps and for windows.
    pub fn alloc_scratch(&self) -> Mamba3MimoScratch {
        Mamba3MimoScratch {
            step: Mamba3MimoStepScratch::new(&self.cfg),
            prefill: Mamba3MimoPrefillScratch::new(&self.cfg),
        }
    }
}

/// The step and window buffers of one MIMO session.
pub struct Mamba3MimoScratch {
    step: Mamba3MimoStepScratch,
    prefill: Mamba3MimoPrefillScratch,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counts name the model: its layers, and every element its
    /// checkpoint holds, one more layer adding one layer's elements.
    #[test]
    fn the_counts_cover_every_checkpoint_tensor() {
        let count = |n_layers: usize| {
            let cfg = Mamba3MimoConfig {
                d_model: 64,
                d_state: 16,
                headdim: 16,
                n_layers,
                ..Mamba3MimoConfig::default()
            };
            let b = Mamba3MimoBackbone::init(cfg, 40, 7).unwrap();
            let (list, _) =
                crate::mamba3_mimo::serialize::checkpoint_parts(b.weights(), b.config(), 40)
                    .unwrap();
            assert_eq!(b.n_layers(), n_layers);
            assert_eq!(b.param_count(), list.element_count());
            b.param_count()
        };
        let (two, three, four) = (count(2), count(3), count(4));
        assert!(two > 0 && three > two);
        assert_eq!(four - three, three - two);
    }

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
        let cfg = Mamba3MimoConfig {
            d_model: 64,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            d_intermediate: 200,
            ..Mamba3MimoConfig::default()
        };
        let input_dim = 40;
        let plain = Mamba3MimoBackbone::init(cfg, input_dim, 7).unwrap();
        let mut packed = Mamba3MimoBackbone::init(cfg, input_dim, 7).unwrap();
        packed.pack_for_decode();
        assert!(packed.is_packed());
        let (mut s0, mut s1) = (plain.alloc_state(), packed.alloc_state());
        let (mut c0, mut c1) = (plain.alloc_scratch(), packed.alloc_scratch());
        let (mut o0, mut o1) = (vec![0f32; cfg.d_model], vec![0f32; cfg.d_model]);
        for t in 0..12u64 {
            let x = det(input_dim, 100 + t);
            plain.forward_step(&x, &mut o0, &mut s0, &mut c0);
            packed.forward_step(&x, &mut o1, &mut s1, &mut c1);
            if NO_BLAS {
                assert_eq!(o0, o1, "step {t}");
            }
        }
    }
}
