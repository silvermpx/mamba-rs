//! The session, options and metrics every trainer shares, from
//! [`crate::optim`], where the CPU trainers find them too.

pub use crate::optim::session::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};
