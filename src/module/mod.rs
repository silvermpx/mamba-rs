//! High-level Mamba wrappers.
//!
//! [`MambaBackbone`] is the primary user-facing API. It owns all weights
//! and provides both single-step inference and batched training methods.

mod backbone;
#[cfg(feature = "hf")]
pub mod backbone3;
#[cfg(all(feature = "hf", feature = "cuda"))]
pub mod gpu_lm;
#[cfg(feature = "hf")]
pub mod lm;
#[cfg(feature = "hf")]
pub mod sample;

pub use backbone::MambaBackbone;
