//! Mamba-2 on the GPU.
//!
//! - `kernels` — the Mamba-2 CUDA module (its own module kind)
//! - `inference` — the decode engine and the chunked prefill
//! - `train` — the f32 trainer: forward, backward and AdamW

pub mod inference;
pub mod kernels;
pub mod train;

pub use inference::{Mamba2GpuInferenceEngine, Mamba2GpuPrefill, Mamba2GpuScratch, Mamba2GpuState};
pub use kernels::Mamba2Kernels;
pub use train::Mamba2GpuTrainer;
