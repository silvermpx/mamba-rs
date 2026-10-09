//! Mamba-3 MIMO CPU backend.
//!
//! - `core` — discretisation, rotation, the rank-generic SSM rows, the gate
//! - `inference` — the T=1 decode step
//! - `prefill` — a prompt window in one pass, bit-equal to the decode chain

pub(crate) mod core;
pub mod inference;
pub mod prefill;

pub use inference::{Mamba3MimoStepScratch, mamba3_mimo_layer_step, mamba3_mimo_step};
pub use prefill::{
    Mamba3MimoPrefillScratch, forward_mamba3_mimo_backbone_prefill,
    forward_mamba3_mimo_backbone_prefill_mode,
};
