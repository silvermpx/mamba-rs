//! Shared operations: dimensions, BLAS, math, normalization utilities.

pub mod blas;
pub mod dims;
pub mod fast_math;
pub(crate) mod init;
pub mod norms;

pub use dims::{MambaDims, MambaRecurrentState};
