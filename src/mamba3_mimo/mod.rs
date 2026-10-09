//! Mamba-3 MIMO (multi-input, multi-output).
//!
//! A sibling of [`crate::mamba3_siso`]: each head keeps one state but takes
//! `mimo_rank` key/value pairs per token and reads `mimo_rank` outputs back,
//! mixed through per-head projections, and rotates keys and queries in
//! half-split pairs. The released checkpoints (state-spaces/mamba3-mimo-*)
//! use rank 4.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569); the
//! reference is state-spaces/mamba `mamba_ssm/modules/mamba3.py`.

pub mod config;
pub mod cpu;
pub mod state;
pub mod weights;

pub use config::Mamba3MimoConfig;
pub use cpu::{
    Mamba3MimoPrefillScratch, Mamba3MimoStepScratch, Mamba3MimoTape, Mamba3MimoTrainScratch,
    backward_mamba3_mimo_train, forward_mamba3_mimo_backbone_prefill,
    forward_mamba3_mimo_backbone_prefill_mode, forward_mamba3_mimo_train, mamba3_mimo_layer_step,
    mamba3_mimo_step,
};
pub use state::{Mamba3MimoLayerState, Mamba3MimoState};
pub use weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
