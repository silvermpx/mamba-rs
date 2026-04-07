//! CUDA GPU backend for Mamba-1 SSM.
//!
//! - `inference` — GPU T=1 step with CUDA Graphs
//! - `forward` — GPU training forward pass
//! - `backward` — GPU training backward pass
//! - `device` — CUDA context + cuBLAS
//! - `buffers` — GPU memory management
//! - `kernels` — NVRTC kernel compilation
//! - `blas` — cuBLAS SGEMM wrappers
//! - `launch` — kernel launch helpers

pub mod backward;
pub mod blas;
pub mod buffers;
pub mod context;
pub mod device;
pub mod forward;
pub mod inference;
pub mod kernels;
pub mod launch;
pub mod weights;
