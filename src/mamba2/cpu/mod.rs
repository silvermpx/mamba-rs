//! Mamba-2 CPU backend.
//!
//! - `core` — conv, dt, the channel-wise scan and the gated norm
//! - `inference` — the T=1 decode step
//! - `prefill` — a prompt window in one pass, bit-equal to the decode chain
//! - `train` — the training forward with a checkpointed tape, and the backward

pub(crate) mod core;
pub mod inference;
pub mod prefill;
pub mod train;

pub use inference::{Mamba2StepScratch, mamba2_layer_step, mamba2_step};
pub use prefill::{
    Mamba2PrefillScratch, forward_mamba2_backbone_prefill, forward_mamba2_backbone_prefill_mode,
};
pub use train::{Mamba2Tape, Mamba2TrainScratch, backward_mamba2_train, forward_mamba2_train};
