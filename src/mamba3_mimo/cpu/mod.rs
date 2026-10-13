//! Mamba-3 MIMO CPU backend.
//!
//! - `core` — discretisation, rotation, the rank-generic SSM rows, the gate
//! - `inference` — the T=1 decode step
//! - `prefill` — a prompt window in one pass, bit-equal to the decode chain
//! - `train` — the training forward with a checkpointed tape, and the backward

pub(crate) mod core;
pub mod inference;
pub mod prefill;
pub mod train;
pub mod trainer;

pub use inference::{Mamba3MimoStepScratch, mamba3_mimo_layer_step, mamba3_mimo_step};
pub use prefill::{
    Mamba3MimoPrefillScratch, forward_mamba3_mimo_backbone_prefill,
    forward_mamba3_mimo_backbone_prefill_mode,
};
pub use train::{
    Mamba3MimoTape, Mamba3MimoTrainScratch, backward_mamba3_mimo_train, forward_mamba3_mimo_train,
};
