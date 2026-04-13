//! # mamba-rs
//!
//! Mamba SSM (Selective State Space Model) implementation in Rust.
//!
//! Provides both CPU and GPU (CUDA) paths for inference and training,
//! including full backward pass with BPTT through recurrent state.
//!
//! Based on: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
//! Selective State Spaces" (NeurIPS 2024).
//!
//! ## Module Structure
//!
//! - [`mamba_ssm`] — Mamba-1 implementation (CPU + GPU)
//!   - `cpu/` — inference, forward, backward
//!   - `gpu/` — CUDA inference, forward, backward
//! - [`mamba3_siso`] — Mamba-3 SISO implementation (CPU + GPU)
//!   - `cpu/` — inference, forward, backward, parallel
//!   - `gpu/` — CUDA kernels (38 kernels)
//! - [`ops`] — shared operations (dims, BLAS, math, norms)
//! - [`module`] — high-level MambaBackbone API
//! - [`config`], [`state`], [`weights`], [`serialize`] — Mamba-1 data types

pub mod config;
#[cfg(feature = "hf")]
pub mod hf;
pub mod mamba3_siso;
pub mod mamba_ssm;
pub mod module;
pub mod ops;
pub mod serialize;
pub mod state;
pub mod weights;

// Re-export old paths for backward compatibility during transition.
// These will be removed once all external users migrate.
pub mod inference {
    pub use crate::mamba_ssm::cpu::inference::*;
}
pub mod train {
    pub use crate::mamba_ssm::cpu::backward;
    pub use crate::mamba_ssm::cpu::backward_ops;
    pub use crate::mamba_ssm::cpu::flat;
    pub use crate::mamba_ssm::cpu::forward;
    pub use crate::mamba_ssm::cpu::parallel;
    pub use crate::mamba_ssm::cpu::scratch;
    pub use crate::mamba_ssm::cpu::target;
    pub use crate::mamba_ssm::cpu::weights;

    // Re-export shared ops that were previously in train/
    pub use crate::ops::blas;
    pub use crate::ops::fast_math;
}

#[cfg(feature = "cuda")]
pub mod gpu {
    pub use crate::mamba_ssm::gpu::*;
}

pub use config::MambaConfig;
pub use mamba_ssm::cpu::inference::{
    MambaLayerScratch, MambaStepScratch, mamba_block_step, mamba_layer_step, mamba_step,
    mamba_step_no_proj,
};
pub use module::MambaBackbone;
pub use state::{MambaLayerState, MambaState};
pub use weights::{MambaLayerWeights, MambaWeights};

// Mamba-3 SISO re-exports
pub use mamba3_siso::{
    Mamba3Config, Mamba3Dims, Mamba3LayerState, Mamba3LayerWeights, Mamba3State, Mamba3StepScratch,
    Mamba3Weights,
};
