//! High-level Mamba wrappers.
//!
//! - [`MambaBackbone`] — the primary CPU API for Mamba SSM. Single-step
//!   inference + batched training methods.
//! - `Mamba3Backbone` (in submodule `backbone3`, under `hf` feature) —
//!   peer wrapper for Mamba-3 SISO, inference-only (training goes
//!   through `crate::mamba3_siso::cpu`).
//! - `gpu_lm` / `gpu_lm3` (under `cuda` + `hf` features) — LM-style
//!   generation (`generate`, `generate_streaming`, `generate_batch`) with
//!   sampling on top of the GPU backbones.

mod backbone;
// backbone3 and sample have no HF dependency — a default-feature build
// gets both CPU backbones and the sampling utilities.
pub mod backbone3;
#[cfg(all(feature = "hf", feature = "cuda"))]
pub mod gpu_lm;
#[cfg(all(feature = "hf", feature = "cuda"))]
pub mod gpu_lm3;
#[cfg(feature = "hf")]
pub mod lm;
pub mod sample;

pub use backbone::MambaBackbone;
pub use backbone3::Mamba3Backbone;
