/// Scan mode for GPU SSM recurrence.
///
/// Controls whether the SSM recurrence uses sequential O(T) scan
/// or parallel prefix scan O(T) work / O(log T) depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanMode {
    /// Sequential O(T) scan — optimal for T <= 128. Zero overhead.
    Sequential,
    /// Parallel prefix scan — optimal for T >= 256. Uses warp shuffle + shared memory.
    Parallel,
    /// Auto-select: Sequential for T <= 128, Parallel for T > 128.
    #[default]
    Auto,
}

impl ScanMode {
    /// Resolve Auto to a concrete mode based on sequence length.
    pub fn resolve(self, seq_len: usize) -> Self {
        match self {
            Self::Auto => {
                if seq_len <= 128 {
                    Self::Sequential
                } else {
                    Self::Parallel
                }
            }
            other => other,
        }
    }
}

/// Configuration for a Mamba SSM model.
///
/// Source: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
/// Paper: <https://arxiv.org/abs/2312.00752>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MambaConfig {
    /// Model dimension — input features projected to this size.
    pub d_model: usize,

    /// SSM state dimension. Controls memory capacity per channel.
    /// Paper default: 16.
    pub d_state: usize,

    /// Local convolution width before SSM. Paper default: 4.
    pub d_conv: usize,

    /// Expansion factor. `d_inner = expand * d_model`.
    /// Paper default: 2.
    pub expand: usize,

    /// Number of stacked Mamba layers. Paper default: varies by model size.
    pub n_layers: usize,

    /// GPU SSM scan mode. Default: Auto (sequential T<=128, parallel T>128).
    /// Only affects GPU training forward/backward. CPU always uses sequential.
    pub scan_mode: ScanMode,
}

impl MambaConfig {
    /// Inner dimension (expanded). Used for in_proj, SSM, conv1d.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// dt_rank = ceil(d_model / 16). Controls delta projection bottleneck.
    pub fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    /// x_proj output dimension: dt_rank + 2 * d_state (for delta, B, C).
    pub fn xdbl_dim(&self) -> usize {
        self.dt_rank() + 2 * self.d_state
    }
}

impl MambaConfig {
    /// Validate configuration constraints.
    ///
    /// Returns `Err` if any dimension would cause kernel failures
    /// (e.g., d_inner not divisible by 4 for vectorized CUDA kernels).
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 {
            return Err("d_model must be > 0".into());
        }
        if self.d_state == 0 {
            return Err("d_state must be > 0".into());
        }
        if self.d_conv == 0 {
            return Err("d_conv must be > 0".into());
        }
        if self.expand == 0 {
            return Err("expand must be > 0".into());
        }
        if self.n_layers == 0 {
            return Err("n_layers must be > 0".into());
        }
        // CUDA parallel scan SSM kernel supports d_state up to MAX_DSTATE=256.
        // Sequential SSM kernels are limited to d_state <= 64 (register arrays),
        // but the dispatch in forward.rs forces the parallel scan path when
        // d_state > 64, so the effective GPU limit is 256.
        if self.d_state > 256 {
            return Err(format!(
                "d_state ({}) must be <= 256 (CUDA parallel scan MAX_DSTATE limit)",
                self.d_state
            ));
        }
        // CUDA conv1d kernel unrolls d_conv — max 8
        if self.d_conv > 8 {
            return Err(format!(
                "d_conv ({}) must be <= 8 (CUDA kernel limit)",
                self.d_conv
            ));
        }
        // CUDA float4 kernels require d_inner divisible by 4
        if !self.d_inner().is_multiple_of(4) {
            return Err(format!(
                "d_inner ({}) must be divisible by 4 (d_model={} * expand={})",
                self.d_inner(),
                self.d_model,
                self.expand
            ));
        }
        Ok(())
    }
}

/// Paper defaults: d_model=128, d_state=16, d_conv=4, expand=2, n_layers=3.
impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_layers: 3,
            scan_mode: ScanMode::Auto,
        }
    }
}
