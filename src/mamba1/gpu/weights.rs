//! GPU weight and gradient storage for Mamba SSM.
//!
//! Three storage patterns:
//! - **Inference weights** (`GpuMambaWeights`): flat buffer + WeightSlice views.
//!   One cuMemAlloc, one H2D copy, CUDA Graph safe.
//! - **Training weights** (`GpuMambaTrainWeights`): per-tensor GpuBuffer.
//!   Standard PyTorch/standard pattern for optimizer compatibility.
//! - **Gradients** (`GpuMambaGrads`): flat buffer + GradSlice views.
//!   One memset zeros all grads. Industry standard (PyTorch DDP, FSDP2).

use crate::gpu::block_norm::NormParams;
use crate::gpu::buffers::{GpuBuffer, GpuByteBuffer, GradSlice, WeightSlice, WeightSliceDyn};
use crate::gpu::dtype::WeightDtype;
use crate::mamba1::config::MambaConfig;
use crate::mamba1::weights::MambaWeights;
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
    /// The in_proj and out_proj biases, `None` for a model without them.
    fn in_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr>;
    fn out_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr>;
    fn norm_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    /// The block LayerNorm's bias, or a null pointer for RMSNorm.
    fn norm_bias(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    /// The conv bias, or a null pointer for a model without one: the conv
    /// kernels read a null bias as zero.
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn dt_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn a_log(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn d_param(&self) -> cudarc::driver::sys::CUdeviceptr;
}

/// Abstraction over backbone-level weights.
pub trait MambaWeightsView {
    type Layer: MambaLayerWeightsView;
    /// Whether the model projects its input rows; without the projection
    /// (an official checkpoint's identity input) the rows are the
    /// residual stream itself and `input_proj_w` holds nothing.
    fn has_input_proj(&self) -> bool;
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype);
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr;
    /// The final LayerNorm's bias, or a null pointer for RMSNorm.
    fn norm_f_bias(&self) -> cudarc::driver::sys::CUdeviceptr;
    fn n_layers(&self) -> usize;
    fn layer(&self, i: usize) -> &Self::Layer;
    /// Layer `i`'s gated MLP, `None` for a model without one.
    fn mlp(&self, i: usize) -> Option<MambaMlpPtrs>;
}

/// Device pointers of one layer's gated MLP: the projections in the
/// container's bulk dtype, the norm in f32.
#[derive(Clone, Copy)]
pub struct MambaMlpPtrs {
    pub norm_weight: cudarc::driver::sys::CUdeviceptr,
    /// The norm's bias, a null pointer for RMSNorm.
    pub norm_bias: cudarc::driver::sys::CUdeviceptr,
    pub fc1_w: (cudarc::driver::sys::CUdeviceptr, WeightDtype),
    pub fc2_w: (cudarc::driver::sys::CUdeviceptr, WeightDtype),
    pub hidden: usize,
}

/// The block norm before a layer's mixer.
pub(crate) fn layer_norm_params(layer: &impl MambaLayerWeightsView) -> NormParams {
    NormParams {
        w: layer.norm_weight(),
        b: layer.norm_bias(),
    }
}

/// The final norm of a backbone.
pub(crate) fn final_norm_params(weights: &impl MambaWeightsView) -> NormParams {
    NormParams {
        w: weights.norm_f_weight(),
        b: weights.norm_f_bias(),
    }
}

/// A gated MLP's norm.
pub(crate) fn mlp_norm_params(m: &MambaMlpPtrs) -> NormParams {
    NormParams {
        w: m.norm_weight,
        b: m.norm_bias,
    }
}

/// A tensor's pointer, `None` when the model does not have it.
fn present(
    len: usize,
    ptr: cudarc::driver::sys::CUdeviceptr,
) -> Option<cudarc::driver::sys::CUdeviceptr> {
    (len > 0).then_some(ptr)
}

impl MambaLayerWeightsView for GpuMambaLayerWeights {
    fn in_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.in_proj_w.ptr(), WeightDtype::F32)
    }
    fn in_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.in_proj_b.len(), self.in_proj_b.ptr())
    }
    fn out_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.out_proj_b.len(), self.out_proj_b.ptr())
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
    fn norm_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_bias.len(), self.norm_bias.ptr()).unwrap_or(0)
    }
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_weight.ptr()
    }
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.conv1d_bias.len(), self.conv1d_bias.ptr()).unwrap_or(0)
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
    fn mlp(&self, i: usize) -> Option<MambaMlpPtrs> {
        self.mlp.get(i).map(|m| MambaMlpPtrs {
            norm_weight: m.norm_weight.ptr(),
            norm_bias: present(m.norm_bias.len(), m.norm_bias.ptr()).unwrap_or(0),
            fc1_w: (m.fc1_w.ptr(), WeightDtype::F32),
            fc2_w: (m.fc2_w.ptr(), WeightDtype::F32),
            hidden: m.hidden,
        })
    }
    fn has_input_proj(&self) -> bool {
        !self.input_proj_w.is_empty()
    }
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.input_proj_w.ptr(), WeightDtype::F32)
    }
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input_proj_b.ptr()
    }
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_f_weight.len(), self.norm_f_weight.ptr()).unwrap_or(0)
    }
    fn norm_f_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_f_bias.len(), self.norm_f_bias.ptr()).unwrap_or(0)
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
    fn in_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.in_proj_b.len_elems(), self.in_proj_b.ptr())
    }
    fn out_proj_b(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.out_proj_b.len_elems(), self.out_proj_b.ptr())
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
    fn norm_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_bias.len_elems(), self.norm_bias.ptr()).unwrap_or(0)
    }
    fn conv1d_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.conv1d_weight.ptr()
    }
    fn conv1d_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.conv1d_bias.len_elems(), self.conv1d_bias.ptr()).unwrap_or(0)
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
    fn mlp(&self, i: usize) -> Option<MambaMlpPtrs> {
        self.mlp.get(i).map(|m| MambaMlpPtrs {
            norm_weight: m.norm_weight.ptr(),
            norm_bias: present(m.norm_bias.len_elems(), m.norm_bias.ptr()).unwrap_or(0),
            fc1_w: (m.fc1_w.ptr(), m.fc1_w.dtype()),
            fc2_w: (m.fc2_w.ptr(), m.fc2_w.dtype()),
            hidden: m.hidden,
        })
    }
    fn has_input_proj(&self) -> bool {
        self.input_proj_w.len_elems() > 0
    }
    fn input_proj_w(&self) -> (cudarc::driver::sys::CUdeviceptr, WeightDtype) {
        (self.input_proj_w.ptr(), self.input_proj_w.dtype())
    }
    fn input_proj_b(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input_proj_b.ptr()
    }
    fn norm_f_weight(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_f_weight.len_elems(), self.norm_f_weight.ptr()).unwrap_or(0)
    }
    fn norm_f_bias(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_f_bias.len_elems(), self.norm_f_bias.ptr()).unwrap_or(0)
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
    /// The block LayerNorm's bias, empty for RMSNorm.
    pub norm_bias: WeightSlice,
    pub in_proj_w: WeightSlice,
    /// Empty without `proj_bias`, as is `out_proj_b`.
    pub in_proj_b: WeightSlice,
    pub conv1d_weight: WeightSlice,
    /// Empty without `conv_bias`.
    pub conv1d_bias: WeightSlice,
    pub x_proj_w: WeightSlice,
    pub dt_proj_w: WeightSlice,
    pub dt_proj_b: WeightSlice,
    pub a_log: WeightSlice,
    pub d_param: WeightSlice,
    pub out_proj_w: WeightSlice,
    pub out_proj_b: WeightSlice,
}

/// One layer's gated MLP in the flat inference buffer.
pub struct GpuMambaMlpWeights {
    pub norm_weight: WeightSlice,
    /// The norm's bias, empty for RMSNorm.
    pub norm_bias: WeightSlice,
    pub fc1_w: WeightSlice,
    pub fc2_w: WeightSlice,
    pub hidden: usize,
}

/// GPU weights for the full Mamba backbone (inference — flat buffer).
pub struct GpuMambaWeights {
    pub flat: GpuBuffer,
    pub input_proj_w: WeightSlice,
    pub input_proj_b: WeightSlice,
    pub layers: Vec<GpuMambaLayerWeights>,
    pub norm_f_weight: WeightSlice,
    /// The final LayerNorm's bias, empty for RMSNorm.
    pub norm_f_bias: WeightSlice,
    /// The gated MLP of each layer, after every other weight; empty for a
    /// model without one.
    pub mlp: Vec<GpuMambaMlpWeights>,
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
            + cfg.block_norm.bias_len(d_model)
            + d_model * 2 * d_inner
            + cfg.in_proj_bias_len()
            + d_inner * d_conv
            + cfg.conv_bias_len()
            + d_inner * xdbl_dim
            + dt_rank * d_inner
            + d_inner
            + d_inner * d_state
            + d_inner
            + d_inner * d_model
            + cfg.out_proj_bias_len();

        // Use the CPU weights' actual lengths. For HF Mamba (identity_proj),
        // both input_proj_w and input_proj_b are empty; the formula must not
        // assume d_model-length bias when the whole projection is skipped.
        let total = cpu.input_proj_w.len()
            + cpu.input_proj_b.len()
            + cfg.n_layers * per_layer
            + cpu.norm_f_weight.len()
            + cpu.norm_f_bias.len()
            + cpu
                .mlp
                .iter()
                .map(|m| m.norm_weight.len() + m.norm_bias.len() + m.fc1_w.len() + m.fc2_w.len())
                .sum::<usize>();

        // The uploads run on the stream after the zeroing memset.
        let flat = GpuBuffer::zeros(stream, total)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        macro_rules! ws {
            ($data:expr) => {{
                let len = $data.len();
                let slice = WeightSlice::from_offset(base, off, len);
                slice.upload_from_cpu(stream, $data)?;
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
                norm_bias: ws!(&lw.norm_bias),
                in_proj_w: ws!(&lw.in_proj_w),
                in_proj_b: ws!(&lw.in_proj_b),
                conv1d_weight: ws!(&lw.conv1d_weight),
                conv1d_bias: ws!(&lw.conv1d_bias),
                x_proj_w: ws!(&lw.x_proj_w),
                dt_proj_w: ws!(&lw.dt_proj_w),
                dt_proj_b: ws!(&lw.dt_proj_b),
                a_log: ws!(&lw.a_log),
                d_param: ws!(&lw.d_param),
                out_proj_w: ws!(&lw.out_proj_w),
                out_proj_b: ws!(&lw.out_proj_b),
            });
        }

        let norm_f_weight = ws!(&cpu.norm_f_weight);
        let norm_f_bias = ws!(&cpu.norm_f_bias);
        let mut mlp = Vec::with_capacity(cpu.mlp.len());
        for m in &cpu.mlp {
            mlp.push(GpuMambaMlpWeights {
                norm_weight: ws!(&m.norm_weight),
                norm_bias: ws!(&m.norm_bias),
                fc1_w: ws!(&m.fc1_w),
                fc2_w: ws!(&m.fc2_w),
                hidden: m.hidden(),
            });
        }
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
            norm_f_bias,
            mlp,
        })
    }

    /// A container holding no weights, for an engine whose kernels read the
    /// weights of another container, so none of the model is uploaded.
    pub(crate) fn empty(stream: &Arc<cudarc::driver::CudaStream>) -> Result<Self, String> {
        let flat = GpuBuffer::zeros(stream, 0)?;
        let base = flat.cached_ptr();
        Ok(Self {
            input_proj_w: WeightSlice::from_offset(base, 0, 0),
            input_proj_b: WeightSlice::from_offset(base, 0, 0),
            layers: Vec::new(),
            norm_f_weight: WeightSlice::from_offset(base, 0, 0),
            norm_f_bias: WeightSlice::from_offset(base, 0, 0),
            mlp: Vec::new(),
            flat,
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
    pub norm_bias: WeightSliceDyn,     // f32, empty for RMSNorm
    pub in_proj_w: WeightSliceDyn,     // bulk (bf16/f16)
    pub in_proj_b: WeightSliceDyn,     // f32, empty without proj_bias
    pub conv1d_weight: WeightSliceDyn, // f32 (small, custom kernel reads directly)
    pub conv1d_bias: WeightSliceDyn,   // f32, empty without conv_bias
    pub x_proj_w: WeightSliceDyn,      // bulk
    pub dt_proj_w: WeightSliceDyn,     // bulk
    pub dt_proj_b: WeightSliceDyn,     // f32
    pub a_log: WeightSliceDyn,         // f32 (used in exp, critical)
    pub d_param: WeightSliceDyn,       // f32 (SSM skip, critical)
    pub out_proj_w: WeightSliceDyn,    // bulk
    pub out_proj_b: WeightSliceDyn,    // f32, empty without proj_bias
}

/// One layer's gated MLP in the mixed container: bulk projections, f32 norm.
pub struct GpuMambaMixedMlpWeights {
    pub norm_weight: WeightSliceDyn, // f32
    pub norm_bias: WeightSliceDyn,   // f32, empty for RMSNorm
    pub fc1_w: WeightSliceDyn,       // bulk
    pub fc2_w: WeightSliceDyn,       // bulk
    pub hidden: usize,
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
    pub norm_f_bias: WeightSliceDyn,   // f32, empty for RMSNorm
    /// The gated MLP of each layer, after every other weight in both
    /// arenas; empty for a model without one.
    pub mlp: Vec<GpuMambaMixedMlpWeights>,
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
        // f32 (per layer): norm + its bias + in_proj_b + conv1d_w + conv1d_b
        // + dt_proj_b + a_log + d_param + out_proj_b
        let per_layer_f32 = d_model
            + cfg.block_norm.bias_len(d_model)
            + cfg.in_proj_bias_len()
            + d_inner * d_conv
            + cfg.conv_bias_len()
            + d_inner
            + d_inner * d_state
            + d_inner
            + cfg.out_proj_bias_len();

        // Use actual CPU weight lengths — HF Mamba has empty input_proj_w/b
        // (identity_proj), while MambaBackbone::init populates both to d_model.
        let mlp_bulk: usize = cpu.mlp.iter().map(|m| m.fc1_w.len() + m.fc2_w.len()).sum();
        let mlp_f32: usize = cpu
            .mlp
            .iter()
            .map(|m| m.norm_weight.len() + m.norm_bias.len())
            .sum();
        let bulk_elems = cpu.input_proj_w.len() + cfg.n_layers * per_layer_bulk + mlp_bulk;
        let f32_elems = cpu.input_proj_b.len()
            + cfg.n_layers * per_layer_f32
            + cpu.norm_f_weight.len()
            + cpu.norm_f_bias.len()
            + mlp_f32;

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
            slice.upload_from_cpu_f32(stream, data)?;
            bulk_off += len * bulk_dtype.size_bytes();
            Ok(slice)
        };
        let mut alloc_f32 = |data: &[f32]| -> Result<WeightSliceDyn, String> {
            let len = data.len();
            let slice = WeightSliceDyn::from_byte_offset(f32_base, f32_off, len, WeightDtype::F32);
            slice.upload_from_cpu_f32(stream, data)?;
            f32_off += len * 4;
            Ok(slice)
        };

        let input_proj_w = alloc_bulk(&cpu.input_proj_w)?;
        let input_proj_b = alloc_f32(&cpu.input_proj_b)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in &cpu.layers {
            layers.push(GpuMambaMixedLayerWeights {
                norm_weight: alloc_f32(&lw.norm_weight)?,
                norm_bias: alloc_f32(&lw.norm_bias)?,
                in_proj_w: alloc_bulk(&lw.in_proj_w)?,
                in_proj_b: alloc_f32(&lw.in_proj_b)?,
                conv1d_weight: alloc_f32(&lw.conv1d_weight)?,
                conv1d_bias: alloc_f32(&lw.conv1d_bias)?,
                x_proj_w: alloc_bulk(&lw.x_proj_w)?,
                dt_proj_w: alloc_bulk(&lw.dt_proj_w)?,
                dt_proj_b: alloc_f32(&lw.dt_proj_b)?,
                a_log: alloc_f32(&lw.a_log)?,
                d_param: alloc_f32(&lw.d_param)?,
                out_proj_w: alloc_bulk(&lw.out_proj_w)?,
                out_proj_b: alloc_f32(&lw.out_proj_b)?,
            });
        }

        let norm_f_weight = alloc_f32(&cpu.norm_f_weight)?;
        let norm_f_bias = alloc_f32(&cpu.norm_f_bias)?;
        let mut mlp = Vec::with_capacity(cpu.mlp.len());
        for m in &cpu.mlp {
            mlp.push(GpuMambaMixedMlpWeights {
                norm_weight: alloc_f32(&m.norm_weight)?,
                norm_bias: alloc_f32(&m.norm_bias)?,
                fc1_w: alloc_bulk(&m.fc1_w)?,
                fc2_w: alloc_bulk(&m.fc2_w)?,
                hidden: m.hidden(),
            });
        }

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
            norm_f_bias,
            mlp,
        })
    }
}

// ---------------------------------------------------------------------------
// Training weights — per-tensor GpuBuffer (PyTorch/standard standard)
// ---------------------------------------------------------------------------

/// GPU training weights for a single Mamba layer (per-tensor allocation).
pub struct GpuMambaTrainLayerWeights {
    pub norm_weight: GpuBuffer,
    /// The block LayerNorm's bias, empty for RMSNorm.
    pub norm_bias: GpuBuffer,
    pub in_proj_w: GpuBuffer,
    /// Empty without `proj_bias`, as is `out_proj_b`.
    pub in_proj_b: GpuBuffer,
    pub conv1d_weight: GpuBuffer,
    /// Empty without `conv_bias`.
    pub conv1d_bias: GpuBuffer,
    pub x_proj_w: GpuBuffer,
    pub dt_proj_w: GpuBuffer,
    pub dt_proj_b: GpuBuffer,
    pub a_log: GpuBuffer,
    pub d_param: GpuBuffer,
    pub out_proj_w: GpuBuffer,
    pub out_proj_b: GpuBuffer,
}

impl GpuMambaTrainLayerWeights {
    /// The block norm before the mixer.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.cached_ptr(),
            b: present(self.norm_bias.len(), self.norm_bias.cached_ptr()).unwrap_or(0),
        }
    }

    /// The in_proj bias, `None` for a model without projection biases.
    pub fn in_proj_b_ptr(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.in_proj_b.len(), self.in_proj_b.cached_ptr())
    }
    /// The out_proj bias, `None` for a model without projection biases.
    pub fn out_proj_b_ptr(&self) -> Option<cudarc::driver::sys::CUdeviceptr> {
        present(self.out_proj_b.len(), self.out_proj_b.cached_ptr())
    }
    /// The conv bias, or a null pointer for a model without one: the conv
    /// kernels read a null bias as zero.
    pub fn conv1d_bias_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.conv1d_bias.len(), self.conv1d_bias.cached_ptr()).unwrap_or(0)
    }
}

/// GPU training weights of one layer's gated MLP.
pub struct GpuMambaTrainMlpWeights {
    /// The pre-MLP norm's scale: `[d_model]`.
    pub norm_weight: GpuBuffer,
    /// The pre-MLP LayerNorm's bias `[d_model]`, empty for RMSNorm.
    pub norm_bias: GpuBuffer,
    /// The up projection `[d_model, 2 * hidden]`.
    pub fc1_w: GpuBuffer,
    /// The down projection `[hidden, d_model]`.
    pub fc2_w: GpuBuffer,
}

impl GpuMambaTrainMlpWeights {
    /// The hidden width these weights carry.
    pub fn hidden(&self) -> usize {
        self.fc2_w.len() / self.norm_weight.len().max(1)
    }

    /// The norm before the MLP.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.cached_ptr(),
            b: present(self.norm_bias.len(), self.norm_bias.cached_ptr()).unwrap_or(0),
        }
    }
}

/// GPU training weights for the full Mamba backbone (per-tensor allocation).
pub struct GpuMambaTrainWeights {
    pub input_proj_w: GpuBuffer,
    pub input_proj_b: GpuBuffer,
    pub layers: Vec<GpuMambaTrainLayerWeights>,
    pub norm_f_weight: GpuBuffer,
    /// The final LayerNorm's bias, empty for RMSNorm.
    pub norm_f_bias: GpuBuffer,
    /// The gated MLP of each layer; empty for a model without one.
    pub mlp: Vec<GpuMambaTrainMlpWeights>,
}

impl GpuMambaTrainWeights {
    /// The final norm; none for a backbone built without it.
    pub(crate) fn norm_f(&self) -> NormParams {
        NormParams {
            w: present(self.norm_f_weight.len(), self.norm_f_weight.cached_ptr()).unwrap_or(0),
            b: present(self.norm_f_bias.len(), self.norm_f_bias.cached_ptr()).unwrap_or(0),
        }
    }

    /// Upload CPU weights to GPU as per-tensor allocations.
    pub fn from_cpu(
        stream: &Arc<cudarc::driver::CudaStream>,
        cpu: &MambaWeights,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cpu.layers.len());
        for lw in &cpu.layers {
            layers.push(GpuMambaTrainLayerWeights {
                norm_weight: GpuBuffer::from_cpu(stream, &lw.norm_weight)?,
                norm_bias: GpuBuffer::from_cpu(stream, &lw.norm_bias)?,
                in_proj_w: GpuBuffer::from_cpu(stream, &lw.in_proj_w)?,
                in_proj_b: GpuBuffer::from_cpu(stream, &lw.in_proj_b)?,
                conv1d_weight: GpuBuffer::from_cpu(stream, &lw.conv1d_weight)?,
                conv1d_bias: GpuBuffer::from_cpu(stream, &lw.conv1d_bias)?,
                x_proj_w: GpuBuffer::from_cpu(stream, &lw.x_proj_w)?,
                dt_proj_w: GpuBuffer::from_cpu(stream, &lw.dt_proj_w)?,
                dt_proj_b: GpuBuffer::from_cpu(stream, &lw.dt_proj_b)?,
                a_log: GpuBuffer::from_cpu(stream, &lw.a_log)?,
                d_param: GpuBuffer::from_cpu(stream, &lw.d_param)?,
                out_proj_w: GpuBuffer::from_cpu(stream, &lw.out_proj_w)?,
                out_proj_b: GpuBuffer::from_cpu(stream, &lw.out_proj_b)?,
            });
        }

        let mlp = cpu
            .mlp
            .iter()
            .map(|m| {
                Ok(GpuMambaTrainMlpWeights {
                    norm_weight: GpuBuffer::from_cpu(stream, &m.norm_weight)?,
                    norm_bias: GpuBuffer::from_cpu(stream, &m.norm_bias)?,
                    fc1_w: GpuBuffer::from_cpu(stream, &m.fc1_w)?,
                    fc2_w: GpuBuffer::from_cpu(stream, &m.fc2_w)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            input_proj_w: GpuBuffer::from_cpu(stream, &cpu.input_proj_w)?,
            input_proj_b: GpuBuffer::from_cpu(stream, &cpu.input_proj_b)?,
            layers,
            norm_f_weight: GpuBuffer::from_cpu(stream, &cpu.norm_f_weight)?,
            norm_f_bias: GpuBuffer::from_cpu(stream, &cpu.norm_f_bias)?,
            mlp,
        })
    }
}

// ---------------------------------------------------------------------------
// Gradients — flat buffer + GradSlice views (one memset zeros all)
// ---------------------------------------------------------------------------

/// Per-layer gradient views into the flat gradient buffer.
pub struct GpuMambaLayerGrads {
    pub norm_weight: GradSlice,
    /// Empty for RMSNorm.
    pub norm_bias: GradSlice,
    pub in_proj_w: GradSlice,
    /// Empty without `proj_bias`, as is `out_proj_b`.
    pub in_proj_b: GradSlice,
    pub conv1d_weight: GradSlice,
    /// Empty without `conv_bias`.
    pub conv1d_bias: GradSlice,
    pub x_proj_w: GradSlice,
    pub dt_proj_w: GradSlice,
    pub dt_proj_b: GradSlice,
    pub a_log: GradSlice,
    pub d_param: GradSlice,
    pub out_proj_w: GradSlice,
    pub out_proj_b: GradSlice,
}

impl GpuMambaLayerGrads {
    /// The block LayerNorm's bias gradient, a null pointer for RMSNorm.
    pub fn norm_bias_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_bias.len(), self.norm_bias.ptr()).unwrap_or(0)
    }
    /// The in_proj bias gradient, `None` for a model without the bias.
    pub fn in_proj_b_opt(&self) -> Option<&GradSlice> {
        (!self.in_proj_b.is_empty()).then_some(&self.in_proj_b)
    }
    /// The out_proj bias gradient, `None` for a model without the bias.
    pub fn out_proj_b_opt(&self) -> Option<&GradSlice> {
        (!self.out_proj_b.is_empty()).then_some(&self.out_proj_b)
    }
}

/// Gradient views of one layer's gated MLP.
pub struct GpuMambaMlpGrads {
    pub norm_weight: GradSlice,
    /// Empty for RMSNorm.
    pub norm_bias: GradSlice,
    pub fc1_w: GradSlice,
    pub fc2_w: GradSlice,
}

impl GpuMambaMlpGrads {
    /// The norm's bias gradient, a null pointer for RMSNorm.
    pub fn norm_bias_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_bias.len(), self.norm_bias.ptr()).unwrap_or(0)
    }
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
    /// The gated MLP of each layer, after the layers and before norm_f,
    /// the CPU trainer's tensor order, so the two trainers' grad arenas
    /// and optimizer states are interchangeable; empty for a model
    /// without one.
    pub mlp: Vec<GpuMambaMlpGrads>,
    pub norm_f_weight: GradSlice,
    /// Empty for RMSNorm.
    pub norm_f_bias: GradSlice,
}

impl GpuMambaGrads {
    /// The final LayerNorm's bias gradient, a null pointer for RMSNorm.
    pub fn norm_f_bias_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        present(self.norm_f_bias.len(), self.norm_f_bias.ptr()).unwrap_or(0)
    }

    /// Allocate zeroed flat gradient buffer with per-tensor views.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        cfg: &MambaConfig,
        input_dim: usize,
    ) -> Result<Self, String> {
        Self::new_sized(stream, cfg, input_dim, true)
    }

    /// Like [`Self::new`], but the input-projection slot is sized from
    /// the ACTUAL master shape: the mixed identity convention (empty
    /// input_proj) previously still paid a full `input_dim x d_model`
    /// grad slot — zeroed every step and mirrored into m/v — for a
    /// tensor that does not exist. NOTE: this changes the flat-arena
    /// layout (and therefore the exported optimizer-state blob) for
    /// identity-input_proj models; blobs written by earlier versions
    /// refuse on length at import, loudly.
    pub fn new_sized(
        stream: &Arc<cudarc::driver::CudaStream>,
        cfg: &MambaConfig,
        input_dim: usize,
        has_input_proj: bool,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        let (ipw_len, ipb_len) = if has_input_proj {
            (input_dim * dm, dm)
        } else {
            (0, 0)
        };
        let (ipb, cb, opb) = (
            cfg.in_proj_bias_len(),
            cfg.conv_bias_len(),
            cfg.out_proj_bias_len(),
        );
        let nb = cfg.block_norm.bias_len(dm);
        let per_layer = dm
            + nb
            + dm * 2 * di
            + ipb
            + di * dc
            + cb
            + di * xd
            + dr * di
            + di
            + di * ds
            + di
            + di * dm
            + opb;
        let hidden = cfg.mlp_hidden();
        let per_mlp = if hidden > 0 {
            dm + nb + 3 * dm * hidden
        } else {
            0
        };
        // A segment built without the final norm has no slots for it.
        let (nf, nfb) = cfg.block_norm.final_lens(dm, cfg.final_norm);
        let total = ipw_len + ipb_len + cfg.n_layers * (per_layer + per_mlp) + nf + nfb;

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

        let input_proj_w = gs!(ipw_len);
        let input_proj_b = gs!(ipb_len);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(GpuMambaLayerGrads {
                norm_weight: gs!(dm),
                norm_bias: gs!(nb),
                in_proj_w: gs!(dm * 2 * di),
                in_proj_b: gs!(ipb),
                conv1d_weight: gs!(di * dc),
                conv1d_bias: gs!(cb),
                x_proj_w: gs!(di * xd),
                dt_proj_w: gs!(dr * di),
                dt_proj_b: gs!(di),
                a_log: gs!(di * ds),
                d_param: gs!(di),
                out_proj_w: gs!(di * dm),
                out_proj_b: gs!(opb),
            });
        }

        let mut mlp = Vec::new();
        for _ in (0..cfg.n_layers).filter(|_| hidden > 0) {
            mlp.push(GpuMambaMlpGrads {
                norm_weight: gs!(dm),
                norm_bias: gs!(nb),
                fc1_w: gs!(dm * 2 * hidden),
                fc2_w: gs!(hidden * dm),
            });
        }
        let norm_f_weight = gs!(nf);
        let norm_f_bias = gs!(nfb);
        debug_assert_eq!(off, total, "grad layout mismatch: off={off} total={total}");

        Ok(Self {
            flat,
            input_proj_w,
            input_proj_b,
            layers,
            mlp,
            norm_f_weight,
            norm_f_bias,
        })
    }

    /// Zero all gradients with a single memset (async on stream).
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        self.flat.zero(stream)
    }
}
