//! Mamba-3 SISO (Single-Input Single-Output) implementation.
//!
//! Compared with Mamba-1 and Mamba-2: no conv1d, an input-dependent A,
//! trapezoidal integration, RoPE and BCNorm.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).
//!
//! ## Module Structure
//!
//! - `config` — `Mamba3Config` with validation (the model envelope)
//! - `route` — which shapes the GPU kernels can run
//! - `cpu/` — CPU inference, training forward/backward
//! - `gpu/` — CUDA GPU inference and training (feature = "cuda")

pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod route;
pub mod serialize;
pub mod state;
pub mod weights;

pub use config::Mamba3Config;
pub use state::{Mamba3LayerState, Mamba3State};
pub use weights::{Mamba3LayerWeights, Mamba3Weights};
