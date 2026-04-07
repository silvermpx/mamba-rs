//! CPU backend for Mamba-1 SSM.
//!
//! - `inference` — zero-alloc T=1 step (single + batch + sequence)
//! - `forward` — training forward pass (batched SGEMM)
//! - `backward` — training backward pass (BPTT)
//! - `target` — target network forward (no activation saves)

pub mod backward;
pub mod backward_ops;
pub mod flat;
pub mod forward;
pub mod inference;
pub mod parallel;
pub mod scratch;
pub mod target;
pub mod weights;
