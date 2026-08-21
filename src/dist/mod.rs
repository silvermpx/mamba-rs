//! Deterministic data-parallel training.
//!
//! One process per GPU, one logical rank per process. Gradients meet in
//! a single collective per optimizer step over the trainer's flat f32
//! arena; the default reduction contract folds the W addends per
//! element in strictly ascending logical-rank order, so the reduced
//! bits are independent of transport, topology, library version, and
//! physical GPU permutation — a run is bit-replayable for a fixed
//! logical world size.
//!
//! Everything here composes with every per-rank compute mode (the
//! deterministic house kernels, cuBLAS, TF32 — any GEMM tier, scan
//! mode, or dtype the trainers support): the reducer consumes finished
//! gradients and never participates in how they were computed. The
//! whole-run bit-identity claim holds when the per-rank route is itself
//! deterministic; on non-invariant tiers the reduction still adds zero
//! new nondeterminism of its own.
//!
//! The seed law keeps every random decision rank-free so the sample
//! stream and its noise are world-size-invariant; the fold plan module
//! is the reduction's numeric contract; [`EmulatedWorld`] proves the
//! sharded dataflow against the straight-line reference in one process,
//! with no communicator involved.

mod bootstrap;
mod config;
mod context;
mod error;
mod fold;
mod seed;

pub use bootstrap::{Bootstrap, SupervisorStatus, attach, bootstrap};
pub use config::{Devices, DistConfig, ReduceContract, Rendezvous};
pub use context::{DistContext, EmulatedWorld};
pub use error::DistError;
pub use fold::{Shard, reduce_mean_reference, shard_plan};
pub use seed::SeedLaw;
