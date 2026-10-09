//! Shared operations: dimensions, BLAS, math, normalization utilities.

pub mod blas;
pub mod dims;
pub mod fast_math;
pub mod fold16;
pub(crate) mod init;
pub mod norms;
pub mod portable_math;

pub use dims::{MambaDims, MambaRecurrentState};
