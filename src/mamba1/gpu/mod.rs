//! Mamba-1 on the GPU, on the shared engine in [`crate::gpu`].
//!
//! - `inference`: the backbone: the decode step with CUDA Graph capture
//!   and the prompt prefill
//! - `prefill`: the GPU prompt prefill
//! - `forward`, `backward`, `forward_mixed`, `backward_mixed`: the f32 and
//!   the mixed-precision training passes
//! - `conv`: the depthwise conv launches they share, at any width
//! - `trainer`, `training_graph`: the training step and its captured graph
//! - `weights`, `weights_mixed_train`: device weights

pub mod backward;
pub mod backward_mixed;
pub(crate) mod conv;
pub mod forward;
pub mod forward_mixed;
pub mod inference;
pub mod prefill;
pub mod trainer;
pub mod training_graph;
pub mod weights;
pub mod weights_mixed_train;

pub use inference::{
    MambaGpuBackbone, MambaGpuEngine, MambaGpuEngineMixed, MambaGpuScratch, MambaGpuScratchMixed,
    MambaGpuState,
};
pub use trainer::MambaGpuTrainer;
