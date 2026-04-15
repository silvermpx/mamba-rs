//! Mamba-3 SISO CUDA GPU backend.
//!
//! - `kernels` — NVRTC compilation + 43 kernel handles
//! - `mamba3_gpu` — GPU training forward + backward
//! - `inference` — GPU T=1 step + CUDA Graph
//! - `weights` — GPU weight/gradient storage

pub mod backward;
pub mod backward_mixed;
pub mod forward;
pub mod forward_mixed;
pub mod inference;
pub mod kernels;
pub mod mamba3_gpu;
pub mod state;
pub mod weights;
pub mod weights_mixed_train;

pub use inference::GpuMamba3Backbone;
pub use kernels::Mamba3Kernels;
pub use weights::{
    GpuMamba3Grads, GpuMamba3LayerGrads, GpuMamba3LayerWeights, GpuMamba3LayerWeightsInf,
    GpuMamba3Weights, GpuMamba3WeightsInf,
};
