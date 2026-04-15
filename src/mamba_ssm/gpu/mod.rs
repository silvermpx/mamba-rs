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

pub mod adamw;
pub mod backward;
pub mod backward_mixed;
pub mod blas;
pub mod buffers;
pub mod context;
pub mod device;
pub mod dtype;
pub mod forward;
pub mod forward_mixed;
pub mod graph_capture;
pub mod inference;
pub mod kernels;
pub mod launch;
pub mod loss_scaler;
pub mod prefill;
pub mod trainer;
pub mod training_graph;
pub mod weights;
pub mod weights_mixed_train;

pub use dtype::WeightDtype;
