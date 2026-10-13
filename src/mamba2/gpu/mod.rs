//! Mamba-2 on the GPU.
//!
//! - `kernels`: the Mamba-2 CUDA module, a module kind of its own
//! - `inference`: the decode engine and the chunked prefill
//! - `backbone`: the engine with its state and buffers, the surface every
//!   family's GPU backbone has
//! - `train`: the trainer, forward, backward and AdamW

pub mod backbone;
pub mod inference;
pub mod kernels;
pub mod train;

pub use backbone::Mamba2GpuBackbone;
pub use inference::{Mamba2GpuEngine, Mamba2GpuPrefill, Mamba2GpuScratch, Mamba2GpuState};
pub use kernels::Mamba2Kernels;
pub use train::{Mamba2GpuTrainer, Mamba2RecurrentStateBlob};
