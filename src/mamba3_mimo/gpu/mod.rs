//! Mamba-3 MIMO on the GPU.
//!
//! - `kernels` — the MIMO CUDA module (its own module kind)
//! - `inference` — the engine; a decode step is a window of one token
//! - `train` — the trainer, chunked or sequential: forward, backward and AdamW

pub mod inference;
pub mod kernels;
pub mod train;

pub use inference::{Mamba3MimoGpuEngine, Mamba3MimoGpuState, Mamba3MimoGpuWindow};
pub use kernels::Mamba3MimoKernels;
pub use train::Mamba3MimoGpuTrainer;
