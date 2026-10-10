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
use crate::mamba3_siso::state::{Mamba3LayerState, Mamba3State};
use crate::mamba3_siso::weights::Mamba3Weights;

/// Mamba-3 SISO backbone for CPU inference. Holds the full set of
/// per-layer weights and the architecture config; the recurrent state
/// is owned by the caller (`Mamba3State`) so multiple sessions can run
/// concurrently against shared weights.
///
/// For GPU inference see `crate::mamba3_siso::gpu::GpuMamba3Backbone`;
/// for LM-style generation with sampling see
/// `crate::module::gpu_lm3::GpuMamba3LM` (feature `cuda` + `hf`).
pub struct Mamba3Backbone {
    weights: Mamba3Weights,
    cfg: Mamba3Config,
    /// The projections packed for decode by [`Self::pack_for_decode`]; the
    /// weights never change after construction, so the copy stays current.
    packed: Option<Mamba3PackedLinears>,
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
        }
    }

    /// Wrap pre-loaded weights — e.g. from
    /// [`crate::mamba3_siso::serialize::load_mamba3`] or a trainer's
    /// `snapshot_master`. Validates every tensor length against `cfg`
    /// before accepting, so undersized checkpoints are rejected here
    /// rather than panicking mid-inference.
    pub fn from_weights(
        weights: Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        Ok(Self {
            weights,
            cfg,
            packed: None,
        })
    }

    /// Borrow the weight bundle (e.g. for serialization).
    pub fn weights(&self) -> &Mamba3Weights {
        &self.weights
    }

    /// Borrow the architecture config.
    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    /// Backbone hidden width (`d_model` in the paper).
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

    /// Single-step (T=1) forward. Reads `input[..d_model]`, writes
    /// `temporal[..d_model]`, advances each layer's recurrent state in
    /// `states` in place. `scratch` is reusable across calls.
    pub fn forward_step(
        &self,
        input: &[f32],
        temporal: &mut [f32],
        scratch: &mut Mamba3StepScratch,
        states: &mut [Mamba3LayerState],
    ) {
        match &self.packed {
            Some(p) => mamba3_step_packed(
                temporal,
                input,
                scratch,
                (&self.weights, p),
                states,
                &self.cfg,
            ),
            None => mamba3_step(temporal, input, scratch, &self.weights, states, &self.cfg),
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
            plain.forward_step(&x, &mut o0, &mut c0, &mut s0.layers);
            packed.forward_step(&x, &mut o1, &mut c1, &mut s1.layers);
            if NO_BLAS {
                assert_eq!(o0, o1, "step {t}");
            }
        }
    }
}
