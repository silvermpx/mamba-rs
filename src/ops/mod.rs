//! Shared operations: dimensions, BLAS, math, normalization utilities.

pub mod blas;
pub mod dims;
pub mod fast_math;
pub mod fold16;
pub(crate) mod init;
pub(crate) mod lane;
pub mod norms;
pub mod packed;
pub mod portable_math;
pub(crate) mod rounding;
pub(crate) mod st_io;
pub(crate) mod step_batch;

pub use dims::{MambaDims, MambaRecurrentState};

/// Gradient chunks of a CPU batch backward. The count is fixed, so the
/// samples each chunk sums and the tree that sums the chunks depend on the
/// batch alone, never on the thread pool: a run gives the same bits on any
/// machine. Sixteen keeps up to sixteen threads busy and bounds the chunks'
/// memory at sixteen copies of the gradient.
pub(crate) const BATCH_GRAD_CHUNKS: usize = 16;
