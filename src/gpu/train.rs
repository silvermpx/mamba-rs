//! The session, options and metrics every trainer shares, from
//! [`crate::optim`], where the CPU trainers find them too.

pub use crate::optim::session::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};

/// Where a window's rows come from: the host, uploaded by the call, or the
/// trainer's or backbone's own device buffer, which the caller wrote on its
/// stream (a hybrid's layer between two segments, with no host copy).
#[derive(Clone, Copy)]
pub(crate) enum RowSource<'a> {
    Host(&'a [f32]),
    Device,
}

impl<'a> RowSource<'a> {
    /// The host rows, None when they already sit on the device.
    pub(crate) fn host(self) -> Option<&'a [f32]> {
        match self {
            Self::Host(rows) => Some(rows),
            Self::Device => None,
        }
    }
}

/// Where the gradient of a window's input goes: nowhere, to the host, or
/// left in the trainer's device buffer for the caller's layer below.
pub(crate) enum InputGrad<'a> {
    None,
    Host(&'a mut [f32]),
    Device,
}
