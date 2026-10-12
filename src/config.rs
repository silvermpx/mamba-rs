//! Configuration shared by the model families: the family itself and the
//! scan mode of the GPU recurrence.

// Mamba-1's configuration lived here up to 0.7. Code written then still
// finds it here for one release, its home is `mamba1::config`.
#[doc(hidden)]
pub use crate::mamba1::config::MambaConfig;

/// The model families this crate runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Mamba (Gu and Dao, 2023), the selective scan.
    Mamba1,
    /// Mamba-2 (Dao and Gu, 2024), the state space duality.
    Mamba2,
    /// Mamba-3 SISO (Lahoti et al., 2026), one input and one output channel
    /// per head.
    Mamba3,
    /// Mamba-3 MIMO, a rank of input and output channels per head.
    Mamba3Mimo,
}

/// Scan mode for GPU SSM recurrence.
///
/// Controls whether the SSM recurrence uses sequential O(T) scan
/// or parallel prefix scan O(T) work / O(log T) depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanMode {
    /// Sequential O(T) scan — zero overhead, optimal for short sequences.
    Sequential,
    /// Parallel prefix scan — warp shuffle + shared memory, optimal for long T.
    Parallel,
    /// Auto-select: Sequential for T <= 64, Parallel for T > 64
    /// (PARALLEL_SCAN_THRESHOLD, measured on RTX 6000 Ada).
    #[default]
    Auto,
}

impl ScanMode {
    /// Auto threshold: sequential at or below, parallel above. Single source
    /// of truth for the GPU dispatch (re-exported as
    /// `mamba1::gpu::forward::PARALLEL_SCAN_THRESHOLD`).
    pub const PARALLEL_SCAN_THRESHOLD: usize = 64;

    /// Resolve the effective scan choice for a sequence length and state dim.
    ///
    /// An explicit mode is honoured at every size: the sequential kernels
    /// are sized to the state at compile time. `Auto` above 64 states takes
    /// the parallel scan at any length, the faster path there, and the
    /// route existing `Auto` runs took.
    pub fn use_parallel(self, seq_len: usize, d_state: usize) -> bool {
        match self {
            Self::Sequential => false,
            Self::Parallel => true,
            Self::Auto => d_state > 64 || seq_len > Self::PARALLEL_SCAN_THRESHOLD,
        }
    }

    /// Resolve `Auto` for a shape. Delegates to [`Self::use_parallel`] so the
    /// two entry points can never disagree.
    pub fn resolve(self, seq_len: usize, d_state: usize) -> Self {
        if self.use_parallel(seq_len, d_state) {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }
}
