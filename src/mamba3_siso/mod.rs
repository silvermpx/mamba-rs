// Mamba-3 is a new module — allow some clippy noise during integration.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::approx_constant)]
#![allow(dead_code)]

//! Mamba-3 SISO (Single-Input Single-Output) implementation.
//!
//! A separate module alongside `mamba_ssm` (Mamba SSM). No conv1d, input-dependent A,
//! trapezoidal integration, RoPE, BCNorm.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).
//!
//! ## Module Structure
//!
//! - `config` — `Mamba3Config` with validation
//! - `cpu/` — CPU inference, training forward/backward
//! - `gpu/` — CUDA GPU inference and training (feature = "cuda")

pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod serialize;
pub mod state;
pub mod weights;

pub use config::Mamba3Config;
pub use cpu::backward::backward_mamba3_layer_batched;
pub use cpu::dims::Mamba3Dims;
pub use cpu::flat::{Mamba3FieldOffsets, Mamba3LayerFlat};
pub use cpu::forward::forward_mamba3_layer_batched;
pub use cpu::inference::{Mamba3StepScratch, mamba3_layer_step, mamba3_step};
pub use cpu::parallel::{
    invalidate_mamba3_scratch, parallel_mamba3_backward, parallel_mamba3_forward,
};
pub use cpu::scratch::Mamba3Scratch;
pub use cpu::weights::{TrainMamba3LayerWeights, TrainMamba3Weights};
pub use state::{Mamba3LayerState, Mamba3State};
pub use weights::{Mamba3LayerWeights, Mamba3Weights};
