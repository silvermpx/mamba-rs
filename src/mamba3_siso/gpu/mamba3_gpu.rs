//! Compatibility shim for the legacy `mamba3_gpu` module path.
//!
//! The 2313-line single-file `mamba3_gpu.rs` was split (task #381) into:
//!   - [`super::state`]    — dimensions, saved acts, scratch, target scratch
//!   - [`super::forward`]  — gpu_forward_mamba3_layer / _backbone / _target_burnin
//!   - [`super::backward`] — gpu_backward_mamba3_layer / _backbone
//!
//! All public types and functions are re-exported here so existing
//! `use mamba3_siso::gpu::mamba3_gpu::Foo` paths continue to work.
//! New code should import directly from the split modules.

pub use super::backward::*;
pub use super::forward::*;
pub use super::state::*;
