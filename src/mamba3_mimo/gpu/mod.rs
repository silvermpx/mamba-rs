//! Mamba-3 MIMO on the GPU.
//!
//! - `kernels`: the MIMO CUDA module, a module kind of its own
//! - `inference`: the engine, where a decode step is a window of one token
//! - `backbone`: the engine with its state and windows, the surface every
//!   family's GPU backbone has
//! - `train`: the trainer, chunked or sequential, forward, backward and
//!   AdamW

pub mod backbone;
pub mod inference;
pub mod kernels;
pub mod train;

pub use backbone::Mamba3MimoGpuBackbone;
pub use inference::{Mamba3MimoGpuEngine, Mamba3MimoGpuState, Mamba3MimoGpuWindow};
pub use kernels::Mamba3MimoKernels;
pub use train::Mamba3MimoGpuTrainer;
