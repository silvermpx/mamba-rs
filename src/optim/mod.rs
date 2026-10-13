//! What training needs whatever the device: the session, options and
//! metrics of a trainer, the AdamW state and the host AdamW, and the
//! trainer-state file.

pub mod adamw;
pub mod session;
pub(crate) mod tensors;
pub mod train_state;

pub use adamw::AdamWStateBlob;
pub use session::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};
pub use train_state::TrainerState;
