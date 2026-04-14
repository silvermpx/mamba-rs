//! GPU weight and gradient storage for Mamba SSM.
//!
//! Three storage patterns:
//! - **Inference weights** (`GpuMambaWeights`): flat buffer + WeightSlice views.
//!   One cuMemAlloc, one H2D copy, CUDA Graph safe.
//! - **Training weights** (`GpuMambaTrainWeights`): per-tensor GpuBuffer.
//!   Standard PyTorch/standard pattern for optimizer compatibility.
//! - **Gradients** (`GpuMambaGrads`): flat buffer + GradSlice views.
//!   One memset zeros all grads. Industry standard (PyTorch DDP, FSDP2).

use super::buffers::{GpuBuffer, GpuByteBuffer, GradSlice, WeightSlice, WeightSliceDyn};
use super::dtype::WeightDtype;
use crate::config::MambaConfig;
use crate::weights::MambaWeights;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Unified weight view trait: abstracts over f32-only (GpuMambaWeights) vs
// mixed-precision (GpuMambaMixedWeights) for inference.
// ---------------------------------------------------------------------------

/// Abstraction over per-layer weights for inference.
/// - Bulk weights return `(ptr, dtype)` — dispatch to sgemm or gemm_ex.
/// - Always-f32 weights return only `ptr`.
pub trait MambaLayerWeightsView {
    // Bulk (dispatches to sgemm if F32, gemm_ex otherwise).
    fn in_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    fn x_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    fn dt_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    fn out_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    // Always-f32
    fn norm_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn dt_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn a_log(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn d_param(&self) -> cudarc::driver::sys::CUdeviceptr;
}

/// Abstraction over backbone-level weights.
pub trait MambaWeightsView {
    type Layer: MambaLayerWeightsView;
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn n_layers(&self) -> usize;
    fn layer(&self, i: usize) -> &Self::Layer;
}

impl MambaLayerWeightsView for GpuMambaLayerWeights {
    fn in_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.in_proj_w.ptr(), WeightDtype::F32)
    }
    fn x_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.x_proj_w.ptr(), WeightDtype::F32)
    }
    fn dt_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.dt_proj_w.ptr(), WeightDtype::F32)
    }
    fn out_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.out_proj_w.ptr(), WeightDtype::F32)
    }
    fn norm_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.norm_weight.ptr()
    }
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_weight.ptr()
    }
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_bias.ptr()
    }
    fn dt_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.dt_proj_b.ptr()
    }
    fn a_log(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.a_log.ptr()
    }
    fn d_param(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.d_param.ptr()
    }
}

impl MambaWeightsView for GpuMambaWeights {
    type Layer = GpuMambaLayerWeights;
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.input_proj_w.ptr(), WeightDtype::F32)
    }
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input_proj_b.ptr()
    }
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.norm_f_weight.ptr()
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, i: usize) -> &Self::Layer {
        &self.layers[i]
    }
}

impl MambaLayerWeightsView for GpuMambaMixedLayerWeights {
    fn in_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.in_proj_w.ptr(), self.in_proj_w.dtype())
    }
    fn x_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.x_proj_w.ptr(), self.x_proj_w.dtype())
    }
    fn dt_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.dt_proj_w.ptr(), self.dt_proj_w.dtype())
    }
    fn out_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.out_proj_w.ptr(), self.out_proj_w.dtype())
    }
    fn norm_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.norm_weight.ptr()
    }
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_weight.ptr()
    }
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_bias.ptr()
    }
    fn dt_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.dt_proj_b.ptr()
    }
    fn a_log(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.a_log.ptr()
    }
    fn d_param(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.d_param.ptr()
    }
}

impl MambaWeightsView for GpuMambaMixedWeights {
    type Layer = GpuMambaMixedLayerWeights;
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.input_proj_w.ptr(), self.input_proj_w.dtype())
    }
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input_proj_b.ptr()
    }
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.norm_f_weight.ptr()
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, i: usize) -> &Self::Layer {
        &self.layers[i]
    }
}

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

        // Use the CPU weights' actual lengths. For HF Mamba (identity_proj),
        // both input_proj_w and input_proj_b are empty; the formula must not
        // assume d_model-length bias when the whole projection is skipped.
        let total = cpu.input_proj_w.len()
            + cpu.input_proj_b.len()
            + cfg.n_layers * per_layer
            + cpu.norm_f_weight.len();

        let flat = GpuBuffer::zeros(stream, total)?;
        // Wait for the async zero-memset to finish before host-sync uploads
        // (cuMemcpyHtoD_v2 on default stream does NOT serialize with custom
        // streams under per-thread default-stream semantics). See
        // `GpuMambaMixedWeights::from_cpu` for details.
        stream
            .synchronize()
            .map_err(|e| format!("sync after f32 weight alloc: {e:?}"))?;
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
// Mixed-precision inference weights — two arenas:
//   - bulk_arena (bf16/f16): linear projection weights (in_proj, x_proj,
//     dt_proj, out_proj, input_proj)
//   - f32_arena: always-f32 tensors (norms, biases, a_log, D, dt_proj_b,
//     conv1d_bias, conv1d_weight, input_proj_b) — critical for numerical
//     stability (exp, softplus, rsqrt, SSM recurrence).
//
// Use this for LLM inference. Training keeps GpuMambaTrainWeights (f32 only).
// ---------------------------------------------------------------------------

pub struct GpuMambaMixedLayerWeights {
    pub norm_weight: WeightSliceDyn,   // f32
    pub in_proj_w: WeightSliceDyn,     // bulk (bf16/f16)
    pub conv1d_weight: WeightSliceDyn, // f32 (small, custom kernel reads directly)
    pub conv1d_bias: WeightSliceDyn,   // f32
    pub x_proj_w: WeightSliceDyn,      // bulk
    pub dt_proj_w: WeightSliceDyn,     // bulk
    pub dt_proj_b: WeightSliceDyn,     // f32
    pub a_log: WeightSliceDyn,         // f32 (used in exp, critical)
    pub d_param: WeightSliceDyn,       // f32 (SSM skip, critical)
    pub out_proj_w: WeightSliceDyn,    // bulk
}

pub struct GpuMambaMixedWeights {
    /// Arena holding bulk weights in bulk_dtype (bf16/f16).
    pub bulk_arena: GpuByteBuffer,
    /// Arena holding always-f32 weights.
    pub f32_arena: GpuByteBuffer,
    /// Dtype of bulk_arena.
    pub bulk_dtype: WeightDtype,
    pub input_proj_w: WeightSliceDyn, // bulk
    pub input_proj_b: WeightSliceDyn, // f32
    pub layers: Vec<GpuMambaMixedLayerWeights>,
    pub norm_f_weight: WeightSliceDyn, // f32
}

impl GpuMambaMixedWeights {
    pub fn from_cpu(
        stream: &Arc<cudarc::driver::CudaStream>,
        cpu: &MambaWeights,
        cfg: &MambaConfig,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let dt_rank = cfg.dt_rank();
        let xdbl_dim = cfg.xdbl_dim();

        // bulk (per layer): in_proj_w + x_proj_w + dt_proj_w + out_proj_w
        let per_layer_bulk =
            d_model * 2 * d_inner + d_inner * xdbl_dim + dt_rank * d_inner + d_inner * d_model;
        // f32 (per layer): norm + conv1d_w + conv1d_b + dt_proj_b + a_log + d_param
        let per_layer_f32 =
            d_model + d_inner * d_conv + d_inner + d_inner + d_inner * d_state + d_inner;

        // Use actual CPU weight lengths — HF Mamba has empty input_proj_w/b
        // (identity_proj), while MambaBackbone::init populates both to d_model.
        let bulk_elems = cpu.input_proj_w.len() + cfg.n_layers * per_layer_bulk;
        let f32_elems =
            cpu.input_proj_b.len() + cfg.n_layers * per_layer_f32 + cpu.norm_f_weight.len();

        let bulk_arena = GpuByteBuffer::zeros(stream, bulk_elems * bulk_dtype.size_bytes())?;
        let f32_arena = GpuByteBuffer::zeros(stream, f32_elems * 4)?;

        // Wait for the async zero-memsets queued by `alloc_zeros` on the custom
        // stream to finish before uploading weight data via `cuMemcpyHtoD_v2`
        // (which runs on the default stream). Without this barrier, under
        // per-thread default-stream semantics (CUDA 12+), the default-stream
        // sync memcpy does NOT serialize with custom-stream async ops; the
        // memset then races with the copy and zeros out just-uploaded data.
        // Observed on mamba-130m-hf bf16 (d_model=768) where layer-0
        // norm_weight was silently overwritten with zeros between upload and
        // first RMSNorm kernel launch, producing zero output and stuck-token
        // decoding.
        stream
            .synchronize()
            .map_err(|e| format!("sync after arena zero-init: {e:?}"))?;

        let bulk_base = bulk_arena.cached_ptr();
        let f32_base = f32_arena.cached_ptr();

        let mut bulk_off = 0usize; // byte offset in bulk_arena
        let mut f32_off = 0usize; // byte offset in f32_arena

        let mut alloc_bulk = |data: &[f32]| -> Result<WeightSliceDyn, String> {
            let len = data.len();
            let slice = WeightSliceDyn::from_byte_offset(bulk_base, bulk_off, len, bulk_dtype);
            slice.upload_from_cpu_f32(data)?;
            bulk_off += len * bulk_dtype.size_bytes();
            Ok(slice)
        };
        let mut alloc_f32 = |data: &[f32]| -> Result<WeightSliceDyn, String> {
            let len = data.len();
            let slice = WeightSliceDyn::from_byte_offset(f32_base, f32_off, len, WeightDtype::F32);
            slice.upload_from_cpu_f32(data)?;
            f32_off += len * 4;
            Ok(slice)
        };

        let input_proj_w = alloc_bulk(&cpu.input_proj_w)?;
        let input_proj_b = alloc_f32(&cpu.input_proj_b)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in &cpu.layers {
            layers.push(GpuMambaMixedLayerWeights {
                norm_weight: alloc_f32(&lw.norm_weight)?,
                in_proj_w: alloc_bulk(&lw.in_proj_w)?,
                conv1d_weight: alloc_f32(&lw.conv1d_weight)?,
                conv1d_bias: alloc_f32(&lw.conv1d_bias)?,
                x_proj_w: alloc_bulk(&lw.x_proj_w)?,
                dt_proj_w: alloc_bulk(&lw.dt_proj_w)?,
                dt_proj_b: alloc_f32(&lw.dt_proj_b)?,
                a_log: alloc_f32(&lw.a_log)?,
                d_param: alloc_f32(&lw.d_param)?,
                out_proj_w: alloc_bulk(&lw.out_proj_w)?,
            });
        }

        let norm_f_weight = alloc_f32(&cpu.norm_f_weight)?;

        debug_assert_eq!(bulk_off, bulk_elems * bulk_dtype.size_bytes());
        debug_assert_eq!(f32_off, f32_elems * 4);

        Ok(Self {
            bulk_arena,
            f32_arena,
            bulk_dtype,
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
