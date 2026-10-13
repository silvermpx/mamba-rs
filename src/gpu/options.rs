//! The options every family's GPU backbone and trainer constructor takes.

use super::dtype::WeightDtype;
use super::gemm_mode::GemmMode;
use crate::config::ScanMode;

/// Where a GPU model runs and in what precision: the device, the storage
/// precision of its weights and activations, and the GEMM mode.
///
/// The default is device 0, f32 storage and the GEMM mode the environment
/// variable `MAMBA_RS_GEMM_MODE` names, the deterministic kernels when it
/// is unset. Change a field with the builder methods, for example
/// `GpuOptions::default().dtype(WeightDtype::Bf16)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuOptions {
    /// CUDA device ordinal.
    pub device: usize,
    /// Storage precision. `Tf32` stores f32 and lets the deterministic
    /// GEMMs take the TF32 kernels where one is measured.
    pub dtype: WeightDtype,
    /// GEMM mode. `None` takes the one `MAMBA_RS_GEMM_MODE` names.
    pub mode: Option<GemmMode>,
    /// Scan of the SSM recurrence over a window. `None` keeps the one the
    /// model's configuration names (a loaded checkpoint's is `Auto`).
    pub scan_mode: Option<ScanMode>,
}

impl GpuOptions {
    /// Run on the device with this ordinal.
    pub fn device(mut self, ordinal: usize) -> Self {
        self.device = ordinal;
        self
    }

    /// Store the weights and activations in this precision.
    pub fn dtype(mut self, dtype: WeightDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Run the GEMMs in this mode whatever `MAMBA_RS_GEMM_MODE` says.
    pub fn mode(mut self, mode: GemmMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Run the scans in this mode whatever the model's configuration says.
    pub fn scan_mode(mut self, mode: ScanMode) -> Self {
        self.scan_mode = Some(mode);
        self
    }

    /// The scan mode a model built with these options runs: this option's
    /// when it is set, else the configuration's own.
    pub(crate) fn scan_mode_or(&self, own: ScanMode) -> ScanMode {
        self.scan_mode.unwrap_or(own)
    }
}
