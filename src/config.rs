//! Configuration shared by the model families: the scan mode of the
//! GPU recurrence.

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
    /// `mamba_ssm::gpu::forward::PARALLEL_SCAN_THRESHOLD`).
    pub const PARALLEL_SCAN_THRESHOLD: usize = 64;

    /// Resolve the effective scan choice for a sequence length and state dim.
    ///
    /// `d_state > 64` ALWAYS takes the parallel path regardless of the
    /// requested mode. Historically the sequential kernels were hard-capped
    /// at 64; they are capacity-sized and guarded now, but the route stays —
    /// it is part of the numeric identity of existing runs, and the
    /// parallel scan is the faster path at large state sizes anyway.
    pub fn use_parallel(self, seq_len: usize, d_state: usize) -> bool {
        if d_state > 64 {
            return true;
        }
        match self {
            Self::Sequential => false,
            Self::Parallel => true,
            Self::Auto => seq_len > Self::PARALLEL_SCAN_THRESHOLD,
        }
    }

    /// Resolve Auto to a concrete mode based on sequence length.
    /// Resolve `Auto` for a shape. Delegates to [`Self::use_parallel`] so the
    /// `d_state > 64` sequential override can never diverge between the two
    /// entry points (this used to duplicate the
    /// threshold WITHOUT the d_state override - a safety-divergent copy).
    pub fn resolve(self, seq_len: usize, d_state: usize) -> Self {
        if self.use_parallel(seq_len, d_state) {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }
}
