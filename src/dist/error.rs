//! Error taxonomy for the distributed data-parallel layer.

use std::fmt;

/// Everything that can go wrong while setting up or running a
/// data-parallel world. Variants carry enough context to act on: config
/// errors are caller bugs, rendezvous/transport errors are environment
/// problems, and `RankFailed` is the fail-fast signal that some peer
/// died and the whole run must stop (a mid-run world-size change would
/// re-partition the data fold and silently change the numbers).
#[non_exhaustive]
#[derive(Debug)]
pub enum DistError {
    /// Invalid configuration (bad device list, world size zero, an
    /// accumulation window not divisible by the world size, ...).
    Config(String),
    /// The launch environment promised a rank layout it did not deliver
    /// (missing or contradictory environment variables).
    EnvContract(String),
    /// Rendezvous failed: peers did not show up within the deadline, or
    /// the rendezvous directory is unusable.
    Rendezvous(String),
    /// A collective or transport operation failed.
    Transport(String),
    /// A peer rank exited or signalled failure; the run must stop.
    RankFailed { rank: usize, detail: String },
    /// Replica-consistency check failed: ranks that must hold identical
    /// bytes diverged (wrong binary, libm drift, memory corruption).
    ReplicaDivergence { tag: String, detail: String },
}

impl fmt::Display for DistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistError::Config(m) => write!(f, "dist config: {m}"),
            DistError::EnvContract(m) => write!(f, "dist env contract: {m}"),
            DistError::Rendezvous(m) => write!(f, "dist rendezvous: {m}"),
            DistError::Transport(m) => write!(f, "dist transport: {m}"),
            DistError::RankFailed { rank, detail } => {
                write!(f, "dist rank {rank} failed: {detail}")
            }
            DistError::ReplicaDivergence { tag, detail } => {
                write!(f, "dist replica divergence at {tag}: {detail}")
            }
        }
    }
}

impl std::error::Error for DistError {}

impl From<DistError> for String {
    fn from(e: DistError) -> Self {
        e.to_string()
    }
}
