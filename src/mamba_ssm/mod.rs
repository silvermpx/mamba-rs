//! Mamba SSM (Selective State Space Model).
//!
//! Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).
//!
//! ## Structure
//! - `config` — the model configuration
//! - `weights`, `state` — the f32 weights and the recurrent state
//! - `serialize` — safetensors save and load of the weights
//! - `cpu/` — CPU inference + training (forward, backward)
//! - `gpu/` — GPU inference + training (CUDA kernels, CUDA Graphs)

pub mod config;
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod gpu;
pub mod serialize;
pub mod state;
pub mod weights;

pub use config::MambaConfig;
pub use state::{MambaLayerState, MambaState};
pub use weights::{MambaLayerWeights, MambaWeights};
