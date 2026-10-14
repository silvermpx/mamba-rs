//! GPU weight storage for Mamba-3 SISO.
//!
//! Same dual pattern as Mamba SSM:
//! - **Inference**: flat buffer + WeightSlice views (CUDA Graph safe, read-only)
//! - **Training**: per-tensor GpuBuffer (optimizer compatible)
//! - **Gradients**: flat buffer + GradSlice views (single zero() clears all)

use crate::gpu::block_norm::NormParams;
use crate::gpu::buffers::{GpuBuffer, GpuByteBuffer, GradSlice, WeightSlice, WeightSliceDyn};
use crate::gpu::dtype::WeightDtype;
use crate::mamba3_siso::config::Mamba3Config;
use std::sync::Arc;

type Stream = Arc<cudarc::driver::CudaStream>;

/// A tensor the model may lack, a LayerNorm bias of an RMSNorm model:
/// `None` for an empty host copy.
fn upload_opt(stream: &Stream, src: &[f32]) -> Result<Option<GpuBuffer>, String> {
    if src.is_empty() {
        return Ok(None);
    }
    GpuBuffer::from_cpu(stream, src).map(Some)
}

/// Zeros for a tensor the model may lack: `None` when `len` is 0.
fn zeros_opt(stream: &Stream, len: usize) -> Result<Option<GpuBuffer>, String> {
    if len == 0 {
        return Ok(None);
    }
    GpuBuffer::zeros(stream, len).map(Some)
}

// ═══ Inference weights — flat buffer + WeightSlice ═══

/// GPU inference weights for a single Mamba-3 layer.
pub struct GpuMamba3LayerWeightsInf {
    pub norm_weight: WeightSlice, // [d_model]
    /// The block norm's bias, LayerNorm only.
    pub norm_bias: Option<WeightSlice>, // [d_model]
    pub in_proj_w: WeightSlice,   // [d_model * in_proj_dim]
    pub dt_bias: WeightSlice,     // [nheads]
    pub b_norm_weight: WeightSlice, // [d_state]
    pub c_norm_weight: WeightSlice, // [d_state]
    pub b_bias: WeightSlice,      // [nheads * d_state]
    pub c_bias: WeightSlice,      // [nheads * d_state]
    pub d_param: WeightSlice,     // [nheads]
    pub norm_gate_weight: WeightSlice, // [d_inner]
    pub out_proj_w: WeightSlice,  // [d_inner * d_model]
}

/// GPU inference weights of one layer's gated MLP.
pub struct GpuGatedMlpWeightsInf {
    pub norm_weight: WeightSlice, // [d_model]
    /// The norm's bias, LayerNorm only.
    pub norm_bias: Option<WeightSlice>, // [d_model]
    pub fc1_w: WeightSlice,       // [d_model * 2 * hidden]
    pub fc2_w: WeightSlice,       // [hidden * d_model]
    pub hidden: usize,
}

/// GPU inference weights for the full Mamba-3 backbone (flat buffer).
pub struct GpuMamba3WeightsInf {
    pub input_proj_w: WeightSlice,
    pub input_proj_b: WeightSlice,
    pub layers: Vec<GpuMamba3LayerWeightsInf>,
    pub norm_f_weight: WeightSlice,
    /// The final norm's bias, LayerNorm only.
    pub norm_f_bias: Option<WeightSlice>,
    /// The gated MLP of each layer; empty for a model without one.
    pub mlp: Vec<GpuGatedMlpWeightsInf>,
    pub flat: GpuBuffer,
}

impl GpuMamba3LayerWeightsInf {
    /// The block norm before the mixer.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.ptr(),
            b: self.norm_bias.as_ref().map_or(0, WeightSlice::ptr),
        }
    }
}

impl GpuGatedMlpWeightsInf {
    /// The norm before the MLP.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.ptr(),
            b: self.norm_bias.as_ref().map_or(0, WeightSlice::ptr),
        }
    }
}

impl GpuMamba3WeightsInf {
    /// The final norm.
    pub(crate) fn final_norm(&self) -> NormParams {
        NormParams {
            w: self.norm_f_weight.ptr(),
            b: self.norm_f_bias.as_ref().map_or(0, WeightSlice::ptr),
        }
    }

    /// Upload CPU weights into a single flat GPU buffer with WeightSlice views.
    pub fn from_cpu(
        stream: &Stream,
        cpu: &crate::mamba3_siso::weights::Mamba3Weights,
        _input_dim: usize,
    ) -> Result<Self, String> {
        // Concatenate all weights into a flat Vec
        let mut flat_data = Vec::new();
        flat_data.extend_from_slice(&cpu.input_proj_w);
        flat_data.extend_from_slice(&cpu.input_proj_b);
        for lw in &cpu.layers {
            flat_data.extend_from_slice(&lw.norm_weight);
            flat_data.extend_from_slice(&lw.norm_bias);
            flat_data.extend_from_slice(&lw.in_proj_w);
            flat_data.extend_from_slice(&lw.dt_bias);
            flat_data.extend_from_slice(&lw.b_norm_weight);
            flat_data.extend_from_slice(&lw.c_norm_weight);
            flat_data.extend_from_slice(&lw.b_bias);
            flat_data.extend_from_slice(&lw.c_bias);
            flat_data.extend_from_slice(&lw.d_param);
            flat_data.extend_from_slice(&lw.norm_gate_weight);
            flat_data.extend_from_slice(&lw.out_proj_w);
        }
        flat_data.extend_from_slice(&cpu.norm_f_weight);
        flat_data.extend_from_slice(&cpu.norm_f_bias);
        // The MLP goes after every mixer weight, so the slices above keep
        // their offsets.
        for m in &cpu.mlp {
            flat_data.extend_from_slice(&m.norm_weight);
            flat_data.extend_from_slice(&m.norm_bias);
            flat_data.extend_from_slice(&m.fc1_w);
            flat_data.extend_from_slice(&m.fc2_w);
        }

        let flat = GpuBuffer::from_cpu(stream, &flat_data)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        let mut slice = |len: usize| -> WeightSlice {
            let s = WeightSlice::from_offset(base, off, len);
            off += len;
            s
        };

        let input_proj_w = slice(cpu.input_proj_w.len());
        let input_proj_b = slice(cpu.input_proj_b.len());

        let mut layers = Vec::new();
        for lw in &cpu.layers {
            layers.push(GpuMamba3LayerWeightsInf {
                norm_weight: slice(lw.norm_weight.len()),
                norm_bias: (!lw.norm_bias.is_empty()).then(|| slice(lw.norm_bias.len())),
                in_proj_w: slice(lw.in_proj_w.len()),
                dt_bias: slice(lw.dt_bias.len()),
                b_norm_weight: slice(lw.b_norm_weight.len()),
                c_norm_weight: slice(lw.c_norm_weight.len()),
                b_bias: slice(lw.b_bias.len()),
                c_bias: slice(lw.c_bias.len()),
                d_param: slice(lw.d_param.len()),
                norm_gate_weight: slice(lw.norm_gate_weight.len()),
                out_proj_w: slice(lw.out_proj_w.len()),
            });
        }
        let norm_f_weight = slice(cpu.norm_f_weight.len());
        let norm_f_bias = (!cpu.norm_f_bias.is_empty()).then(|| slice(cpu.norm_f_bias.len()));
        let mlp = cpu
            .mlp
            .iter()
            .map(|m| GpuGatedMlpWeightsInf {
                norm_weight: slice(m.norm_weight.len()),
                norm_bias: (!m.norm_bias.is_empty()).then(|| slice(m.norm_bias.len())),
                fc1_w: slice(m.fc1_w.len()),
                fc2_w: slice(m.fc2_w.len()),
                hidden: m.hidden(),
            })
            .collect();

        Ok(Self {
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
            norm_f_bias,
            mlp,
            flat,
        })
    }

    /// A container holding no weights, for an engine whose kernels read the
    /// weights of another container, so none of the model is uploaded.
    pub(crate) fn empty(stream: &Stream) -> Result<Self, String> {
        let flat = GpuBuffer::zeros(stream, 0)?;
        let base = flat.cached_ptr();
        Ok(Self {
            input_proj_w: WeightSlice::from_offset(base, 0, 0),
            input_proj_b: WeightSlice::from_offset(base, 0, 0),
            layers: Vec::new(),
            norm_f_weight: WeightSlice::from_offset(base, 0, 0),
            norm_f_bias: None,
            mlp: Vec::new(),
            flat,
        })
    }
}

// ═══ Mixed-precision inference weights ═══
//
// For Mamba-3, bulk (bf16/f16) weights are:
//   - in_proj_w (large: d_model × in_proj_dim)
//   - out_proj_w (large: d_inner × d_model)
//   - input_proj_w (if user has non-identity input proj)
//
// All other M3 tensors stay f32 (norms, biases, d_param, dt_bias — all small,
// critical for numerical stability per research).

pub struct GpuMamba3MixedLayerWeights {
    pub norm_weight: WeightSliceDyn, // f32
    /// The block norm's bias, LayerNorm only.
    pub norm_bias: Option<WeightSliceDyn>, // f32
    pub in_proj_w: WeightSliceDyn,   // bulk
    pub dt_bias: WeightSliceDyn,     // f32
    pub b_norm_weight: WeightSliceDyn, // f32
    pub c_norm_weight: WeightSliceDyn, // f32
    pub b_bias: WeightSliceDyn,      // f32
    pub c_bias: WeightSliceDyn,      // f32
    pub d_param: WeightSliceDyn,     // f32
    pub norm_gate_weight: WeightSliceDyn, // f32
    pub out_proj_w: WeightSliceDyn,  // bulk
}

/// One layer's gated MLP in the mixed container: bulk projections, f32 norm.
pub struct GpuMamba3MixedMlpWeights {
    pub norm_weight: WeightSliceDyn, // f32
    /// The norm's bias, LayerNorm only.
    pub norm_bias: Option<WeightSliceDyn>, // f32
    pub fc1_w: WeightSliceDyn,       // bulk
    pub fc2_w: WeightSliceDyn,       // bulk
    pub hidden: usize,
}

pub struct GpuMamba3MixedWeights {
    pub bulk_arena: GpuByteBuffer,
    pub f32_arena: GpuByteBuffer,
    pub bulk_dtype: WeightDtype,
    pub input_proj_w: WeightSliceDyn, // bulk
    pub input_proj_b: WeightSliceDyn, // f32
    pub layers: Vec<GpuMamba3MixedLayerWeights>,
    pub norm_f_weight: WeightSliceDyn, // f32
    /// The final norm's bias, LayerNorm only.
    pub norm_f_bias: Option<WeightSliceDyn>, // f32
    /// The gated MLP of each layer; empty for a model without one.
    pub mlp: Vec<GpuMamba3MixedMlpWeights>,
}

impl GpuMamba3MixedLayerWeights {
    /// The block norm before the mixer.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.ptr(),
            b: self.norm_bias.as_ref().map_or(0, WeightSliceDyn::ptr),
        }
    }
}

impl GpuMamba3MixedMlpWeights {
    /// The norm before the MLP.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.ptr(),
            b: self.norm_bias.as_ref().map_or(0, WeightSliceDyn::ptr),
        }
    }
}

impl GpuMamba3MixedWeights {
    /// The final norm.
    pub(crate) fn final_norm(&self) -> NormParams {
        NormParams {
            w: self.norm_f_weight.ptr(),
            b: self.norm_f_bias.as_ref().map_or(0, WeightSliceDyn::ptr),
        }
    }

    pub fn from_cpu(
        stream: &Stream,
        cpu: &crate::mamba3_siso::weights::Mamba3Weights,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        // Compute arena sizes (in elements)
        let bulk_elems: usize = std::iter::once(cpu.input_proj_w.len())
            .chain(
                cpu.layers
                    .iter()
                    .flat_map(|lw| [lw.in_proj_w.len(), lw.out_proj_w.len()]),
            )
            .chain(cpu.mlp.iter().flat_map(|m| [m.fc1_w.len(), m.fc2_w.len()]))
            .sum();

        let f32_elems: usize = cpu.input_proj_b.len()
            + cpu.norm_f_weight.len()
            + cpu.norm_f_bias.len()
            + cpu
                .layers
                .iter()
                .map(|lw| {
                    lw.norm_weight.len()
                        + lw.norm_bias.len()
                        + lw.dt_bias.len()
                        + lw.b_norm_weight.len()
                        + lw.c_norm_weight.len()
                        + lw.b_bias.len()
                        + lw.c_bias.len()
                        + lw.d_param.len()
                        + lw.norm_gate_weight.len()
                })
                .sum::<usize>()
            + cpu
                .mlp
                .iter()
                .map(|m| m.norm_weight.len() + m.norm_bias.len())
                .sum::<usize>();

        let bulk_arena = GpuByteBuffer::zeros(stream, bulk_elems * bulk_dtype.size_bytes())?;
        let f32_arena = GpuByteBuffer::zeros(stream, f32_elems * 4)?;

        // Wait for the async zero-memsets queued by `alloc_zeros` on the custom
        // stream to finish before uploading weight data via `cuMemcpyHtoD_v2`
        // (which runs on the default stream). Without this barrier, under
        // per-thread default-stream semantics (CUDA 12+), the default-stream
        // sync memcpy does NOT serialize with custom-stream async ops; the
        // memset then races with the copy and zeros out just-uploaded data.
        // See `GpuMambaMixedWeights::from_cpu` for the original bug report.
        stream
            .synchronize()
            .map_err(|e| format!("sync after m3 mixed arena zero-init: {e:?}"))?;

        let bulk_base = bulk_arena.cached_ptr();
        let f32_base = f32_arena.cached_ptr();

        let mut bulk_off = 0usize;
        let mut f32_off = 0usize;

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

        let mut layers = Vec::with_capacity(cpu.layers.len());
        for lw in &cpu.layers {
            layers.push(GpuMamba3MixedLayerWeights {
                norm_weight: alloc_f32(&lw.norm_weight)?,
                norm_bias: if lw.norm_bias.is_empty() {
                    None
                } else {
                    Some(alloc_f32(&lw.norm_bias)?)
                },
                in_proj_w: alloc_bulk(&lw.in_proj_w)?,
                dt_bias: alloc_f32(&lw.dt_bias)?,
                b_norm_weight: alloc_f32(&lw.b_norm_weight)?,
                c_norm_weight: alloc_f32(&lw.c_norm_weight)?,
                b_bias: alloc_f32(&lw.b_bias)?,
                c_bias: alloc_f32(&lw.c_bias)?,
                d_param: alloc_f32(&lw.d_param)?,
                norm_gate_weight: alloc_f32(&lw.norm_gate_weight)?,
                out_proj_w: alloc_bulk(&lw.out_proj_w)?,
            });
        }

        let norm_f_weight = alloc_f32(&cpu.norm_f_weight)?;
        let norm_f_bias = if cpu.norm_f_bias.is_empty() {
            None
        } else {
            Some(alloc_f32(&cpu.norm_f_bias)?)
        };
        // The MLP goes after every mixer weight, so the slices above keep
        // their offsets.
        let mut mlp = Vec::with_capacity(cpu.mlp.len());
        for m in &cpu.mlp {
            mlp.push(GpuMamba3MixedMlpWeights {
                norm_weight: alloc_f32(&m.norm_weight)?,
                norm_bias: if m.norm_bias.is_empty() {
                    None
                } else {
                    Some(alloc_f32(&m.norm_bias)?)
                },
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

// ═══ Training weights — per-tensor GpuBuffer ═══

/// GPU training weights for a single Mamba-3 layer.
pub struct GpuMamba3LayerWeights {
    pub norm_weight: GpuBuffer,
    /// The block norm's bias, LayerNorm only.
    pub norm_bias: Option<GpuBuffer>,
    pub in_proj_w: GpuBuffer,
    pub dt_bias: GpuBuffer,
    pub b_norm_weight: GpuBuffer,
    pub c_norm_weight: GpuBuffer,
    pub b_bias: GpuBuffer,
    pub c_bias: GpuBuffer,
    pub d_param: GpuBuffer,
    pub norm_gate_weight: GpuBuffer,
    pub out_proj_w: GpuBuffer,
}

/// GPU training weights of one layer's gated MLP.
pub struct GpuMamba3MlpWeights {
    /// The pre-MLP norm's scale: `[d_model]`.
    pub norm_weight: GpuBuffer,
    /// The pre-MLP norm's bias, LayerNorm only: `[d_model]`.
    pub norm_bias: Option<GpuBuffer>,
    /// The up projection `[d_model, 2 * hidden]`, value columns then gate.
    pub fc1_w: GpuBuffer,
    /// The down projection `[hidden, d_model]`.
    pub fc2_w: GpuBuffer,
}

impl GpuMamba3MlpWeights {
    fn from_cpu(stream: &Stream, m: &crate::mlp::GatedMlpLayerWeights) -> Result<Self, String> {
        Ok(Self {
            norm_weight: GpuBuffer::from_cpu(stream, &m.norm_weight)?,
            norm_bias: upload_opt(stream, &m.norm_bias)?,
            fc1_w: GpuBuffer::from_cpu(stream, &m.fc1_w)?,
            fc2_w: GpuBuffer::from_cpu(stream, &m.fc2_w)?,
        })
    }

    /// The hidden width.
    pub fn hidden(&self) -> usize {
        self.fc2_w.len() / self.norm_weight.len().max(1)
    }

    /// The norm before the MLP.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.cached_ptr(),
            b: self.norm_bias.as_ref().map_or(0, GpuBuffer::cached_ptr),
        }
    }
}

/// GPU training weights for the full Mamba-3 backbone.
pub struct GpuMamba3Weights {
    pub input_proj_w: GpuBuffer,
    pub input_proj_b: GpuBuffer,
    pub layers: Vec<GpuMamba3LayerWeights>,
    pub norm_f_weight: GpuBuffer,
    /// The final norm's bias, LayerNorm only.
    pub norm_f_bias: Option<GpuBuffer>,
    /// The gated MLP of each layer; empty for a model without one.
    pub mlp: Vec<GpuMamba3MlpWeights>,
}

impl GpuMamba3LayerWeights {
    pub fn from_cpu(
        stream: &Stream,
        lw: &crate::mamba3_siso::weights::Mamba3LayerWeights,
        _cfg: &Mamba3Config,
    ) -> Result<Self, String> {
        Ok(Self {
            norm_weight: GpuBuffer::from_cpu(stream, &lw.norm_weight)?,
            norm_bias: upload_opt(stream, &lw.norm_bias)?,
            in_proj_w: GpuBuffer::from_cpu(stream, &lw.in_proj_w)?,
            dt_bias: GpuBuffer::from_cpu(stream, &lw.dt_bias)?,
            b_norm_weight: GpuBuffer::from_cpu(stream, &lw.b_norm_weight)?,
            c_norm_weight: GpuBuffer::from_cpu(stream, &lw.c_norm_weight)?,
            b_bias: GpuBuffer::from_cpu(stream, &lw.b_bias)?,
            c_bias: GpuBuffer::from_cpu(stream, &lw.c_bias)?,
            d_param: GpuBuffer::from_cpu(stream, &lw.d_param)?,
            norm_gate_weight: GpuBuffer::from_cpu(stream, &lw.norm_gate_weight)?,
            out_proj_w: GpuBuffer::from_cpu(stream, &lw.out_proj_w)?,
        })
    }

    pub fn zeros(stream: &Stream, cfg: &Mamba3Config) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();
        Ok(Self {
            norm_weight: GpuBuffer::zeros(stream, dm)?,
            norm_bias: zeros_opt(stream, cfg.block_norm.bias_len(dm))?,
            in_proj_w: GpuBuffer::zeros(stream, dm * ip)?,
            dt_bias: GpuBuffer::zeros(stream, nh)?,
            b_norm_weight: GpuBuffer::zeros(stream, ds)?,
            c_norm_weight: GpuBuffer::zeros(stream, ds)?,
            b_bias: GpuBuffer::zeros(stream, nh * ds)?,
            c_bias: GpuBuffer::zeros(stream, nh * ds)?,
            d_param: GpuBuffer::zeros(stream, nh)?,
            norm_gate_weight: GpuBuffer::zeros(stream, di)?,
            out_proj_w: GpuBuffer::zeros(stream, di * dm)?,
        })
    }

    /// The block norm before the mixer.
    pub(crate) fn norm(&self) -> NormParams {
        NormParams {
            w: self.norm_weight.cached_ptr(),
            b: self.norm_bias.as_ref().map_or(0, GpuBuffer::cached_ptr),
        }
    }
}

impl GpuMamba3Weights {
    pub fn from_cpu(
        stream: &Stream,
        cpu: &crate::mamba3_siso::weights::Mamba3Weights,
        cfg: &Mamba3Config,
        _input_dim: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            input_proj_w: GpuBuffer::from_cpu(stream, &cpu.input_proj_w)?,
            input_proj_b: GpuBuffer::from_cpu(stream, &cpu.input_proj_b)?,
            layers: cpu
                .layers
                .iter()
                .map(|lw| GpuMamba3LayerWeights::from_cpu(stream, lw, cfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm_f_weight: GpuBuffer::from_cpu(stream, &cpu.norm_f_weight)?,
            norm_f_bias: upload_opt(stream, &cpu.norm_f_bias)?,
            mlp: cpu
                .mlp
                .iter()
                .map(|m| GpuMamba3MlpWeights::from_cpu(stream, m))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub fn zeros(stream: &Stream, cfg: &Mamba3Config, input_dim: usize) -> Result<Self, String> {
        let dm = cfg.d_model;
        Ok(Self {
            input_proj_w: GpuBuffer::zeros(stream, input_dim * dm)?,
            input_proj_b: GpuBuffer::zeros(stream, dm)?,
            layers: (0..cfg.n_layers)
                .map(|_| GpuMamba3LayerWeights::zeros(stream, cfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm_f_weight: GpuBuffer::zeros(stream, dm)?,
            norm_f_bias: zeros_opt(stream, cfg.block_norm.bias_len(dm))?,
            mlp: (0..cfg.n_layers)
                .filter(|_| cfg.mlp_hidden() > 0)
                .map(|_| {
                    let h = cfg.mlp_hidden();
                    Ok(GpuMamba3MlpWeights {
                        norm_weight: GpuBuffer::zeros(stream, dm)?,
                        norm_bias: zeros_opt(stream, cfg.block_norm.bias_len(dm))?,
                        fc1_w: GpuBuffer::zeros(stream, dm * 2 * h)?,
                        fc2_w: GpuBuffer::zeros(stream, h * dm)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        })
    }

    /// The final norm.
    pub(crate) fn final_norm(&self) -> NormParams {
        NormParams {
            w: self.norm_f_weight.cached_ptr(),
            b: self.norm_f_bias.as_ref().map_or(0, GpuBuffer::cached_ptr),
        }
    }
}

// ═══ Gradients — flat buffer + GradSlice ═══

/// GPU gradients for a single Mamba-3 layer.
pub struct GpuMamba3LayerGrads {
    pub norm_weight: GradSlice,
    /// The block norm's bias, LayerNorm only.
    pub norm_bias: Option<GradSlice>,
    pub in_proj_w: GradSlice,
    pub dt_bias: GradSlice,
    pub b_norm_weight: GradSlice,
    pub c_norm_weight: GradSlice,
    pub b_bias: GradSlice,
    pub c_bias: GradSlice,
    pub d_param: GradSlice,
    pub norm_gate_weight: GradSlice,
    pub out_proj_w: GradSlice,
}

/// GPU gradients of one layer's gated MLP.
pub struct GpuMamba3MlpGrads {
    pub norm_weight: GradSlice,
    /// The norm's bias, LayerNorm only.
    pub norm_bias: Option<GradSlice>,
    pub fc1_w: GradSlice,
    pub fc2_w: GradSlice,
}

/// GPU gradients for the full Mamba-3 backbone, laid out in the host
/// trainer's tensor order: the input projection, every layer's mixer, every
/// layer's gated MLP, the final norm. The AdamW moments follow the same
/// layout, so an optimizer state moves between the devices.
pub struct GpuMamba3Grads {
    pub input_proj_w: GradSlice,
    pub input_proj_b: GradSlice,
    pub layers: Vec<GpuMamba3LayerGrads>,
    /// The gated MLP of each layer; empty for a model without one.
    pub mlp: Vec<GpuMamba3MlpGrads>,
    pub norm_f_weight: GradSlice,
    /// The final norm's bias, LayerNorm only.
    pub norm_f_bias: Option<GradSlice>,
    pub flat: GpuBuffer,
}

impl GpuMamba3Grads {
    /// The f32 values one layer's mixer gradients take in the arena.
    pub(crate) fn layer_len(cfg: &Mamba3Config) -> usize {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();
        dm + cfg.block_norm.bias_len(dm)
            + dm * ip
            + nh
            + ds
            + ds
            + nh * ds
            + nh * ds
            + nh
            + di
            + di * dm
    }

    /// Allocate gradient buffer for full Mamba-3 backbone.
    pub fn new(stream: &Stream, cfg: &Mamba3Config, input_dim: usize) -> Result<Self, String> {
        Self::new_sized(stream, cfg, input_dim, true)
    }

    /// Like [`Self::new`], with the input projection's slots only when the
    /// model has the projection: a model that feeds its input straight to
    /// the first layer (an empty `input_proj_w`) has no such gradient,
    /// and the host trainer's state holds none either.
    pub fn new_sized(
        stream: &Stream,
        cfg: &Mamba3Config,
        input_dim: usize,
        has_input_proj: bool,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();
        let nb = cfg.block_norm.bias_len(dm);
        let per_layer = Self::layer_len(cfg);
        let hidden = cfg.mlp_hidden();
        let per_mlp = if hidden > 0 {
            dm + nb + 3 * dm * hidden
        } else {
            0
        };
        let (ipw_len, ipb_len) = if has_input_proj {
            (input_dim * dm, dm)
        } else {
            (0, 0)
        };
        let total = ipw_len + ipb_len + cfg.n_layers * (per_layer + per_mlp) + dm + nb;
        let flat = GpuBuffer::zeros(stream, total)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        let mut slice = |len: usize| -> GradSlice {
            let s = GradSlice::from_offset(base, off, len);
            off += len;
            s
        };

        let input_proj_w = slice(ipw_len);
        let input_proj_b = slice(ipb_len);

        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(GpuMamba3LayerGrads {
                norm_weight: slice(dm),
                norm_bias: (nb > 0).then(|| slice(nb)),
                in_proj_w: slice(dm * ip),
                dt_bias: slice(nh),
                b_norm_weight: slice(ds),
                c_norm_weight: slice(ds),
                b_bias: slice(nh * ds),
                c_bias: slice(nh * ds),
                d_param: slice(nh),
                norm_gate_weight: slice(di),
                out_proj_w: slice(di * dm),
            });
        }
        let mlp = (0..cfg.n_layers)
            .filter(|_| hidden > 0)
            .map(|_| GpuMamba3MlpGrads {
                norm_weight: slice(dm),
                norm_bias: (nb > 0).then(|| slice(nb)),
                fc1_w: slice(dm * 2 * hidden),
                fc2_w: slice(hidden * dm),
            })
            .collect();
        let norm_f_weight = slice(dm);
        let norm_f_bias = (nb > 0).then(|| slice(nb));
        debug_assert_eq!(
            off, total,
            "SISO gradient arena: {off} of {total} values laid out"
        );

        Ok(Self {
            input_proj_w,
            input_proj_b,
            layers,
            mlp,
            norm_f_weight,
            norm_f_bias,
            flat,
        })
    }

    /// Zero all gradients (single memset on flat buffer).
    pub fn zero(&mut self, stream: &Stream) -> Result<(), String> {
        self.flat.zero(stream)
    }
}

// ═══ Weights view — one prefill surface over both containers ═══

/// Raw device pointers for one layer's weights, as the prefill consumes
/// them. Every norm/bias/param tensor is f32 in BOTH containers; only
/// `in_proj_w` / `out_proj_w` follow the container's bulk dtype.
pub struct Mamba3LayerWeightPtrs {
    pub norm_weight: u64,
    /// The block norm's bias, 0 for RMSNorm.
    pub norm_bias: u64,
    pub in_proj_w: u64,
    pub dt_bias: u64,
    pub b_norm_weight: u64,
    pub c_norm_weight: u64,
    pub b_bias: u64,
    pub c_bias: u64,
    pub d_param: u64,
    pub norm_gate_weight: u64,
    pub out_proj_w: u64,
}

/// Device pointers of one layer's gated MLP, in the container's bulk
/// dtype for the projections and f32 for the norm.
pub struct GatedMlpWeightPtrs {
    pub norm_weight: u64,
    /// The norm's bias, 0 for RMSNorm.
    pub norm_bias: u64,
    pub fc1_w: u64,
    pub fc2_w: u64,
    pub hidden: usize,
}

/// The prefill's weight surface, implemented by the f32 inference
/// container and the mixed (typed-bulk) container alike. The prefill
/// branches its GEMM/kernel dtypes on `bulk_dtype()`; the pointer
/// layout is identical either way (same split, same per-layer tensors).
/// House precedent: the M1 prefill's `MambaWeightsView`.
pub trait Mamba3WeightsView {
    /// Dtype of the bulk projection weights (`F32` for the flat f32
    /// container).
    fn bulk_dtype(&self) -> WeightDtype;
    /// `(w_ptr, b_ptr)` of the input projection; validity of the pointers
    /// when the checkpoint has an identity projection is the caller's
    /// `identity_proj` contract, same as today.
    fn input_proj(&self) -> (u64, u64);
    fn n_layers(&self) -> usize;
    fn layer(&self, l: usize) -> Mamba3LayerWeightPtrs;
    /// The final norm's scale and bias, the bias 0 for RMSNorm.
    fn norm_f(&self) -> (u64, u64);
    /// Layer `l`'s gated MLP, `None` for a model without one.
    fn mlp(&self, l: usize) -> Option<GatedMlpWeightPtrs>;
    /// Identity of the underlying device allocations, for the CUDA-graph
    /// replay guards: a captured graph baked every weight pointer in, so
    /// a replay against a different container (or dtype) must refuse
    /// instead of silently running the captured weights.
    fn arena_identity(&self) -> (u64, u64);
}

impl Mamba3WeightsView for GpuMamba3WeightsInf {
    fn bulk_dtype(&self) -> WeightDtype {
        WeightDtype::F32
    }
    fn arena_identity(&self) -> (u64, u64) {
        (self.flat.cached_ptr(), 0)
    }
    fn input_proj(&self) -> (u64, u64) {
        (self.input_proj_w.ptr(), self.input_proj_b.ptr())
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, l: usize) -> Mamba3LayerWeightPtrs {
        let lw = &self.layers[l];
        Mamba3LayerWeightPtrs {
            norm_weight: lw.norm_weight.ptr(),
            norm_bias: lw.norm().b,
            in_proj_w: lw.in_proj_w.ptr(),
            dt_bias: lw.dt_bias.ptr(),
            b_norm_weight: lw.b_norm_weight.ptr(),
            c_norm_weight: lw.c_norm_weight.ptr(),
            b_bias: lw.b_bias.ptr(),
            c_bias: lw.c_bias.ptr(),
            d_param: lw.d_param.ptr(),
            norm_gate_weight: lw.norm_gate_weight.ptr(),
            out_proj_w: lw.out_proj_w.ptr(),
        }
    }
    fn norm_f(&self) -> (u64, u64) {
        let p = self.final_norm();
        (p.w, p.b)
    }
    fn mlp(&self, l: usize) -> Option<GatedMlpWeightPtrs> {
        self.mlp.get(l).map(|m| GatedMlpWeightPtrs {
            norm_weight: m.norm_weight.ptr(),
            norm_bias: m.norm().b,
            fc1_w: m.fc1_w.ptr(),
            fc2_w: m.fc2_w.ptr(),
            hidden: m.hidden,
        })
    }
}

impl Mamba3WeightsView for GpuMamba3MixedWeights {
    fn bulk_dtype(&self) -> WeightDtype {
        self.bulk_dtype
    }
    fn arena_identity(&self) -> (u64, u64) {
        (self.bulk_arena.cached_ptr(), self.f32_arena.cached_ptr())
    }
    fn input_proj(&self) -> (u64, u64) {
        (self.input_proj_w.ptr(), self.input_proj_b.ptr())
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, l: usize) -> Mamba3LayerWeightPtrs {
        let lw = &self.layers[l];
        Mamba3LayerWeightPtrs {
            norm_weight: lw.norm_weight.ptr(),
            norm_bias: lw.norm().b,
            in_proj_w: lw.in_proj_w.ptr(),
            dt_bias: lw.dt_bias.ptr(),
            b_norm_weight: lw.b_norm_weight.ptr(),
            c_norm_weight: lw.c_norm_weight.ptr(),
            b_bias: lw.b_bias.ptr(),
            c_bias: lw.c_bias.ptr(),
            d_param: lw.d_param.ptr(),
            norm_gate_weight: lw.norm_gate_weight.ptr(),
            out_proj_w: lw.out_proj_w.ptr(),
        }
    }
    fn norm_f(&self) -> (u64, u64) {
        let p = self.final_norm();
        (p.w, p.b)
    }
    fn mlp(&self, l: usize) -> Option<GatedMlpWeightPtrs> {
        self.mlp.get(l).map(|m| GatedMlpWeightPtrs {
            norm_weight: m.norm_weight.ptr(),
            norm_bias: m.norm().b,
            fc1_w: m.fc1_w.ptr(),
            fc2_w: m.fc2_w.ptr(),
            hidden: m.hidden,
        })
    }
}
