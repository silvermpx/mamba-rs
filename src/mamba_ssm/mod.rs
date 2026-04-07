//! Mamba-1 SSM (Selective State Space Model).
//!
//! Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).
//!
//! ## Structure
//! - `cpu/` — CPU inference + training (forward, backward)
//! - `gpu/` — GPU inference + training (CUDA kernels, CUDA Graphs)

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod gpu;
