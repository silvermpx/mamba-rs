//! Mamba-1 model configuration.

use crate::config::ScanMode;

/// Configuration for a Mamba SSM model.
///
/// Source: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
/// Paper: <https://arxiv.org/abs/2312.00752>
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// GPU SSM scan mode. Default: Auto (see [`ScanMode`]).
    /// Only affects GPU training forward/backward. CPU always uses sequential.
    pub scan_mode: ScanMode,

    /// RMSNorm epsilon for every layer norm and the final norm_f.
    /// Default 1e-5 (state-spaces/mamba). FalconMamba checkpoints use 1e-6 —
    /// the HF loader plumbs the checkpoint's value through here.
    pub rms_norm_eps: f32,
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
        // Upper bounds. Configs can come from untrusted sources (safetensors
        // metadata, HF config.json): keep dimension products from wrapping
        // and keep grid.y (= d_inner for the CUDA parallel scan) under the
        // hardware limit of 65535.
        if self.d_model > (1 << 20) {
            return Err(format!("d_model ({}) must be <= 2^20", self.d_model));
        }
        if self.n_layers > 4096 {
            return Err(format!("n_layers ({}) must be <= 4096", self.n_layers));
        }
        let d_inner = self
            .expand
            .checked_mul(self.d_model)
            .ok_or("expand * d_model overflows usize")?;
        if d_inner > 65535 {
            return Err(format!(
                "d_inner ({d_inner}) must be <= 65535 (CUDA grid.y limit for the parallel scan)"
            ));
        }
        // 256 is the reference implementations' own maximum. The
        // sequential SSM kernels size their per-thread state arrays at
        // JIT time from the model config (past ~128 the compiler spills
        // to local memory — correct, slower); the training dispatch
        // still routes d_state > 64 to the parallel scan as its faster
        // numeric route.
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
            rms_norm_eps: crate::ops::fast_math::RMS_NORM_EPS,
        }
    }
}
