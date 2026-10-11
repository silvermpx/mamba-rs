//! Mamba-3 SISO backbone wrapper (CPU). Mirrors [`MambaBackbone`] for the
//! Mamba-3 SISO architecture (Lahoti et al., ICLR 2026): no conv1d,
//! input-dependent A, trapezoidal integration, RoPE, BCNorm, output gate.
//!
//! This is the **inference-only T=1** entrypoint. Training (forward,
//! BPTT, parallel rayon) lives in [`crate::mamba3_siso::cpu`] and operates
//! on `TrainMamba3Weights` instead.
//!
//! [`MambaBackbone`]: crate::module::MambaBackbone

use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::cpu::inference::{
    Mamba3PackedLinears, Mamba3StepScratch, mamba3_step, mamba3_step_packed,
};
use crate::mamba3_siso::state::Mamba3State;
use crate::mamba3_siso::weights::Mamba3Weights;

/// Mamba-3 SISO backbone for CPU inference. Holds the full set of
/// per-layer weights and the architecture config; the recurrent state
/// is owned by the caller (`Mamba3State`) so multiple sessions can run
/// concurrently against shared weights.
///
/// For GPU inference see `crate::mamba3_siso::gpu::Mamba3GpuBackbone`;
/// for text generation see `crate::module::lm::MambaLM` (feature `hf`)
/// and `crate::module::gpu_lm::MambaGpuLM` (features `cuda` and `hf`).
pub struct Mamba3Backbone {
    weights: Mamba3Weights,
    cfg: Mamba3Config,
    /// The projections packed for decode by [`Self::pack_for_decode`]; the
    /// weights never change after construction, so the copy stays current.
    packed: Option<Mamba3PackedLinears>,
    /// Width of an input row.
    input_dim: usize,
}

impl Mamba3Backbone {
    /// Initialize fresh weights from `cfg`. Uses the Mamba-3 init scheme
    /// (zeros for biases, ones for B/C bias and norm scales, the
    /// Mamba-paper dt_proj scheme for the discretization MLP).
    pub fn init(cfg: Mamba3Config, input_dim: usize, seed: u64) -> Self {
        let weights = Mamba3Weights::init(&cfg, input_dim, seed);
        Self {
            weights,
            cfg,
            packed: None,
            input_dim,
        }
    }

    /// Wrap loaded weights for input rows `input_dim` wide, from a
    /// checkpoint ([`crate::mamba3_siso::serialize::load_mamba3`]) or a
    /// trainer's `snapshot_master`. Every tensor length is checked against
    /// `cfg` first, so an undersized checkpoint is refused here instead of
    /// failing mid-inference.
    pub fn new(
        weights: Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        Ok(Self {
            weights,
            cfg,
            packed: None,
            input_dim,
        })
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `Mamba3Backbone::new(weights, cfg, input_dim)`"
    )]
    pub fn from_weights(
        weights: Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        Self::new(weights, cfg, input_dim)
    }

    /// Borrow the weight bundle (e.g. for serialization).
    pub fn weights(&self) -> &Mamba3Weights {
        &self.weights
    }

    /// Borrow the architecture config.
    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    /// Width of an input row.
    pub fn input_dim(&self) -> usize {
        self.input_dim
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
        self.packed = Some(Mamba3PackedLinears::pack(
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
        state: &mut Mamba3State,
        scratch: &mut Mamba3StepScratch,
    ) {
        let states = &mut state.layers;
        match &self.packed {
            Some(p) => mamba3_step_packed(
                output,
                input,
                scratch,
                (&self.weights, p),
                states,
                &self.cfg,
            ),
            None => mamba3_step(output, input, scratch, &self.weights, states, &self.cfg),
        }
    }

    /// `seq_len` tokens in a row, every one through [`Self::forward_step`]:
    /// `inputs` is `[seq_len, input_dim]`, `outputs` `[seq_len, d_model]`.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut Mamba3State,
        scratch: &mut Mamba3StepScratch,
        seq_len: usize,
    ) {
        let (din, dm) = (self.input_dim, self.cfg.d_model);
        for (x, y) in inputs.chunks(din).zip(outputs.chunks_mut(dm)).take(seq_len) {
            self.forward_step(x, y, state, scratch);
        }
    }

    /// Allocate a zero-initialized full-stack recurrent state (SSM + K +
    /// V + angle accumulators per layer). Reset between independent
    /// sequences.
    pub fn alloc_state(&self) -> Mamba3State {
        Mamba3State::zeros(&self.cfg)
    }

    /// Allocate a per-step scratch buffer (split, BCNorm, RoPE, gating,
    /// proj). Reusable across calls; never aliased across threads.
    pub fn alloc_scratch(&self) -> Mamba3StepScratch {
        Mamba3StepScratch::new(&self.cfg)
    }
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
        let cfg = Mamba3Config {
            d_model: 64,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            d_intermediate: 200,
            ..Mamba3Config::default()
        };
        let input_dim = 40;
        let plain = Mamba3Backbone::init(cfg, input_dim, 7);
        let mut packed = Mamba3Backbone::init(cfg, input_dim, 7);
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
