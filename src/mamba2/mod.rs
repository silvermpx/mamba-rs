//! Mamba-2, the structured state space duality (SSD).
//!
//! Each head has a scalar decay per token and a `d_state x headdim` state
//! fed through a causal conv over x, B and C; the output is gated by a
//! grouped RMSNorm. The released checkpoints are state-spaces/mamba2-*.
//!
//! Source: Dao & Gu, "Transformers are SSMs", ICML 2024 (arXiv
//! 2405.21060); the reference is state-spaces/mamba
//! `mamba_ssm/modules/mamba2.py`.

pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod route;
pub mod serialize;
pub mod state;
pub mod weights;

pub use config::Mamba2Config;
pub use cpu::{
    Mamba2PrefillScratch, Mamba2StepScratch, Mamba2Tape, Mamba2TrainScratch, backward_mamba2_train,
    forward_mamba2_backbone_prefill, forward_mamba2_backbone_prefill_mode, forward_mamba2_train,
    mamba2_layer_step, mamba2_step,
};
pub use state::{Mamba2LayerState, Mamba2State};
pub use weights::{Mamba2LayerWeights, Mamba2Weights};
