//! GPU weight and gradient storage for Mamba SSM.
//!
//! Three storage patterns:
//! - **Inference weights** (`GpuMambaWeights`): flat buffer + WeightSlice views.
//!   One cuMemAlloc, one H2D copy, CUDA Graph safe.
//! - **Training weights** (`GpuMambaTrainWeights`): per-tensor GpuBuffer.
//!   Standard PyTorch/standard pattern for optimizer compatibility.
//! - **Gradients** (`GpuMambaGrads`): flat buffer + GradSlice views.
//!   One memset zeros all grads. Industry standard (PyTorch DDP, FSDP2).

use super::buffers::{GpuBuffer, GradSlice, WeightSlice};
use crate::config::MambaConfig;
use crate::weights::MambaWeights;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Inference weights — flat buffer + WeightSlice views (read-only, CUDA Graph safe)
// ---------------------------------------------------------------------------

/// GPU weights for a single Mamba layer (inference — flat buffer views).
pub struct GpuMambaLayerWeights {
    pub norm_weight: WeightSlice,
    pub in_proj_w: WeightSlice,
    pub conv1d_weight: WeightSlice,
    pub conv1d_bias: WeightSlice,
    pub x_proj_w: WeightSlice,
    pub dt_proj_w: WeightSlice,
    pub dt_proj_b: WeightSlice,
    pub a_log: WeightSlice,
    pub d_param: WeightSlice,
    pub out_proj_w: WeightSlice,
}

/// GPU weights for the full Mamba backbone (inference — flat buffer).
pub struct GpuMambaWeights {
    pub flat: GpuBuffer,
    pub input_proj_w: WeightSlice,
    pub input_proj_b: WeightSlice,
    pub layers: Vec<GpuMambaLayerWeights>,
    pub norm_f_weight: WeightSlice,
}

impl GpuMambaWeights {
    /// Upload CPU weights to GPU as a single contiguous allocation.
    pub fn from_cpu(
        stream: &Arc<cudarc::driver::CudaStream>,
        cpu: &MambaWeights,
        cfg: &MambaConfig,
    ) -> Result<Self, String> {
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let dt_rank = cfg.dt_rank();
        let xdbl_dim = cfg.xdbl_dim();

        let per_layer = d_model
            + d_model * 2 * d_inner
            + d_inner * d_conv
            + d_inner
            + d_inner * xdbl_dim
            + dt_rank * d_inner
            + d_inner
            + d_inner * d_state
            + d_inner
            + d_inner * d_model;

        let input_dim = cpu.input_proj_w.len() / d_model;
        let total = input_dim * d_model + d_model + cfg.n_layers * per_layer + d_model;

        let flat = GpuBuffer::zeros(stream, total)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        macro_rules! ws {
            ($data:expr) => {{
                let len = $data.len();
                let slice = WeightSlice::from_offset(base, off, len);
                slice.upload_from_cpu($data)?;
                off += len;
                slice
            }};
        }

        let input_proj_w = ws!(&cpu.input_proj_w);
        let input_proj_b = ws!(&cpu.input_proj_b);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in &cpu.layers {
            layers.push(GpuMambaLayerWeights {
                norm_weight: ws!(&lw.norm_weight),
                in_proj_w: ws!(&lw.in_proj_w),
                conv1d_weight: ws!(&lw.conv1d_weight),
                conv1d_bias: ws!(&lw.conv1d_bias),
                x_proj_w: ws!(&lw.x_proj_w),
                dt_proj_w: ws!(&lw.dt_proj_w),
                dt_proj_b: ws!(&lw.dt_proj_b),
                a_log: ws!(&lw.a_log),
                d_param: ws!(&lw.d_param),
                out_proj_w: ws!(&lw.out_proj_w),
            });
        }

        let norm_f_weight = ws!(&cpu.norm_f_weight);
        debug_assert_eq!(
            off, total,
            "weight layout mismatch: off={off} total={total}"
        );

        Ok(Self {
            flat,
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
        })
    }
}

// ---------------------------------------------------------------------------
// Training weights — per-tensor GpuBuffer (PyTorch/standard standard)
// ---------------------------------------------------------------------------

/// GPU training weights for a single Mamba layer (per-tensor allocation).
pub struct GpuMambaTrainLayerWeights {
    pub norm_weight: GpuBuffer,
    pub in_proj_w: GpuBuffer,
    pub conv1d_weight: GpuBuffer,
    pub conv1d_bias: GpuBuffer,
    pub x_proj_w: GpuBuffer,
    pub dt_proj_w: GpuBuffer,
    pub dt_proj_b: GpuBuffer,
    pub a_log: GpuBuffer,
    pub d_param: GpuBuffer,
    pub out_proj_w: GpuBuffer,
}

/// GPU training weights for the full Mamba backbone (per-tensor allocation).
pub struct GpuMambaTrainWeights {
    pub input_proj_w: GpuBuffer,
    pub input_proj_b: GpuBuffer,
    pub layers: Vec<GpuMambaTrainLayerWeights>,
    pub norm_f_weight: GpuBuffer,
}

impl GpuMambaTrainWeights {
    /// Upload CPU weights to GPU as per-tensor allocations.
    pub fn from_cpu(
        stream: &Arc<cudarc::driver::CudaStream>,
        cpu: &MambaWeights,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cpu.layers.len());
        for lw in &cpu.layers {
            layers.push(GpuMambaTrainLayerWeights {
                norm_weight: GpuBuffer::from_cpu(stream, &lw.norm_weight)?,
                in_proj_w: GpuBuffer::from_cpu(stream, &lw.in_proj_w)?,
                conv1d_weight: GpuBuffer::from_cpu(stream, &lw.conv1d_weight)?,
                conv1d_bias: GpuBuffer::from_cpu(stream, &lw.conv1d_bias)?,
                x_proj_w: GpuBuffer::from_cpu(stream, &lw.x_proj_w)?,
                dt_proj_w: GpuBuffer::from_cpu(stream, &lw.dt_proj_w)?,
                dt_proj_b: GpuBuffer::from_cpu(stream, &lw.dt_proj_b)?,
                a_log: GpuBuffer::from_cpu(stream, &lw.a_log)?,
                d_param: GpuBuffer::from_cpu(stream, &lw.d_param)?,
                out_proj_w: GpuBuffer::from_cpu(stream, &lw.out_proj_w)?,
            });
        }

        Ok(Self {
            input_proj_w: GpuBuffer::from_cpu(stream, &cpu.input_proj_w)?,
            input_proj_b: GpuBuffer::from_cpu(stream, &cpu.input_proj_b)?,
            layers,
            norm_f_weight: GpuBuffer::from_cpu(stream, &cpu.norm_f_weight)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Gradients — flat buffer + GradSlice views (one memset zeros all)
// ---------------------------------------------------------------------------

/// Per-layer gradient views into the flat gradient buffer.
pub struct GpuMambaLayerGrads {
    pub norm_weight: GradSlice,
    pub in_proj_w: GradSlice,
    pub conv1d_weight: GradSlice,
    pub conv1d_bias: GradSlice,
    pub x_proj_w: GradSlice,
    pub dt_proj_w: GradSlice,
    pub dt_proj_b: GradSlice,
    pub a_log: GradSlice,
    pub d_param: GradSlice,
    pub out_proj_w: GradSlice,
}

/// Flat gradient buffer with GradSlice views for all Mamba parameters.
///
/// One `zero()` call clears all gradients. Industry standard layout
/// (PyTorch DDP, FSDP2, standard).
pub struct GpuMambaGrads {
    pub flat: GpuBuffer,
    pub input_proj_w: GradSlice,
    pub input_proj_b: GradSlice,
    pub layers: Vec<GpuMambaLayerGrads>,
    pub norm_f_weight: GradSlice,
}

impl GpuMambaGrads {
    /// Allocate zeroed flat gradient buffer with per-tensor views.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        cfg: &MambaConfig,
        input_dim: usize,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        let per_layer =
            dm + dm * 2 * di + di * dc + di + di * xd + dr * di + di + di * ds + di + di * dm;
        let total = input_dim * dm + dm + cfg.n_layers * per_layer + dm;

        let flat = GpuBuffer::zeros(stream, total)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        macro_rules! gs {
            ($len:expr) => {{
                let len = $len;
                let slice = GradSlice::from_offset(base, off, len);
                off += len;
                slice
            }};
        }

        let input_proj_w = gs!(input_dim * dm);
        let input_proj_b = gs!(dm);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(GpuMambaLayerGrads {
                norm_weight: gs!(dm),
                in_proj_w: gs!(dm * 2 * di),
                conv1d_weight: gs!(di * dc),
                conv1d_bias: gs!(di),
                x_proj_w: gs!(di * xd),
                dt_proj_w: gs!(dr * di),
                dt_proj_b: gs!(di),
                a_log: gs!(di * ds),
                d_param: gs!(di),
                out_proj_w: gs!(di * dm),
            });
        }

        let norm_f_weight = gs!(dm);
        debug_assert_eq!(off, total, "grad layout mismatch: off={off} total={total}");

        Ok(Self {
            flat,
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
        })
    }

    /// Zero all gradients with a single memset (async on stream).
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        self.flat.zero(stream)
    }
}
