//! Shared operations: dimensions, BLAS, math utilities.

pub mod blas;
pub mod dims;
pub mod fast_math;

pub use dims::{MambaDims, MambaRecurrentState};
