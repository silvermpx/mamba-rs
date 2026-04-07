//! High-level Mamba wrappers.
//!
//! [`MambaBackbone`] is the primary user-facing API. It owns all weights
//! and provides both single-step inference and batched training methods.

mod backbone;

pub use backbone::MambaBackbone;
