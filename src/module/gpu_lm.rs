//! Unified GPU-accelerated language model wrapper.
//!
//! Single `GpuMambaLM` struct supports f32 / bf16 / f16 weight storage.
//! Compute is always f32 (CUBLAS_COMPUTE_32F for GEMMs, f32 for custom kernels).
//! Dtype is chosen at construction via `from_hf_with_dtype`.

use std::path::Path;

use crate::mamba_ssm::gpu::blas::{gpu_gemm_ex_forward_raw, gpu_sgemm_forward_raw};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::inference::GpuMambaBackbone;

use crate::hf::embed::embed_lookup;
use crate::hf::load::{HfModel, load_hf};

use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};

/// Internal: embed + optional lm_head storage. F32 uses GpuBuffer (f32 typed);
/// bf16/f16 use GpuByteBuffer (raw bytes, typed via dtype field).
enum EmbedStorage {
    F32 {
        embed: GpuBuffer,
        lm_head: Option<GpuBuffer>,
    },
    Half {
        embed: GpuByteBuffer,
        lm_head: Option<GpuByteBuffer>,
        dtype: WeightDtype,
    },
}

/// Unified GPU Mamba language model.
///
/// Same API regardless of storage dtype:
/// - `from_hf(dir, gpu)` — f32 storage (default, maximum accuracy)
/// - `from_hf_with_dtype(dir, gpu, dtype)` — f32 / bf16 / f16 storage
///
/// ```rust,no_run
/// use mamba_rs::module::gpu_lm::GpuMambaLM;
/// use mamba_rs::module::sample::SampleParams;
/// use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
/// use std::path::Path;
///
/// // f32 (default)
/// let mut lm = GpuMambaLM::from_hf(Path::new("./mamba-130m-hf"), 0).unwrap();
///
/// // bf16 (half VRAM)
/// let mut lm_bf16 = GpuMambaLM::from_hf_with_dtype(
///     Path::new("./mamba-130m-hf"), 0, WeightDtype::Bf16
/// ).unwrap();
///
/// lm.capture_graph().unwrap();
/// let tokens = lm.generate(&[1, 2, 3], &SampleParams::default()).unwrap();
/// ```
pub struct GpuMambaLM {
    backbone: GpuMambaBackbone,
    embed_storage: EmbedStorage,
    /// CPU mirror of embed table for per-token lookup → host scratch buffer.
    embed_cpu: Vec<f32>,
    /// Pre-allocated staging for embed_lookup result (fed into backbone).
    input_cpu: Vec<f32>,
    /// GPU logits output (f32).
    gpu_logits: GpuBuffer,
    /// GPU f32 hidden staging (used only by untied-lm_head half path).
    gpu_hidden: GpuBuffer,
    /// CPU mirror of logits (padded).
    logits_padded_cpu: Vec<f32>,
    /// CPU logits clamped to real vocab_size.
    logits_cpu: Vec<f32>,
    pub vocab_size: usize,
    vocab_size_padded: usize,
    pub d_model: usize,
}

impl GpuMambaLM {
    /// Load HF model with f32 storage.
    pub fn from_hf(dir: &Path, gpu_ordinal: usize) -> Result<Self, String> {
        Self::from_hf_with_dtype(dir, gpu_ordinal, WeightDtype::F32)
    }

    /// Load HF model with explicit storage dtype.
    pub fn from_hf_with_dtype(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let HfModel {
            backbone: cpu_backbone,
            embed,
            lm_head,
            vocab_size,
            vocab_size_padded,
            d_model,
        } = load_hf(dir)?;

        let cfg = *cpu_backbone.config();
        let backbone = GpuMambaBackbone::new_with_dtype(
            gpu_ordinal,
            cpu_backbone.weights(),
            cfg,
            d_model,
            1,
            dtype,
        )?;
        let stream = backbone.stream();

        // Upload embed + optional lm_head in requested dtype.
        let embed_storage = match dtype {
            WeightDtype::F32 => {
                let mut e = GpuBuffer::zeros(stream, vocab_size_padded * d_model)?;
                e.upload(stream, &embed)?;
                let lm = if let Some(ref lm_w) = lm_head {
                    let mut b = GpuBuffer::zeros(stream, lm_w.len())?;
                    b.upload(stream, lm_w)?;
                    Some(b)
                } else {
                    None
                };
                EmbedStorage::F32 {
                    embed: e,
                    lm_head: lm,
                }
            }
            WeightDtype::Bf16 | WeightDtype::F16 => {
                let embed_bytes = embed.len() * dtype.size_bytes();
                let e = GpuByteBuffer::zeros(stream, embed_bytes)?;
                upload_f32_as_dtype(&e, 0, &embed, embed.len(), dtype)?;

                let lm = if let Some(ref lm_w) = lm_head {
                    let lm_bytes = lm_w.len() * dtype.size_bytes();
                    let b = GpuByteBuffer::zeros(stream, lm_bytes)?;
                    upload_f32_as_dtype(&b, 0, lm_w, lm_w.len(), dtype)?;
                    Some(b)
                } else {
                    None
                };
                EmbedStorage::Half {
                    embed: e,
                    lm_head: lm,
                    dtype,
                }
            }
        };

        let gpu_logits = GpuBuffer::zeros(stream, vocab_size_padded)?;
        let gpu_hidden = GpuBuffer::zeros(stream, d_model)?;

        Ok(Self {
            backbone,
            embed_storage,
            embed_cpu: embed,
            input_cpu: vec![0.0; d_model],
            gpu_logits,
            gpu_hidden,
            logits_padded_cpu: vec![0.0; vocab_size_padded],
            logits_cpu: vec![0.0; vocab_size],
            vocab_size,
            vocab_size_padded,
            d_model,
        })
    }

    /// Storage dtype (f32 / bf16 / f16).
    pub fn dtype(&self) -> WeightDtype {
        match &self.embed_storage {
            EmbedStorage::F32 { .. } => WeightDtype::F32,
            EmbedStorage::Half { dtype, .. } => *dtype,
        }
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.backbone.capture_graph()
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &SampleParams,
    ) -> Result<Vec<u32>, String> {
        let mut tokens = Vec::with_capacity(params.max_tokens);
        self.generate_streaming(prompt, params, |tok, _| {
            tokens.push(tok);
        })?;
        Ok(tokens)
    }

    pub fn generate_streaming(
        &mut self,
        prompt: &[u32],
        params: &SampleParams,
        mut cb: impl FnMut(u32, &str),
    ) -> Result<(), String> {
        self.backbone.reset()?;
        let mut rng = Xoshiro256PlusPlus::new(params.seed);

        // Prefill: step-by-step, keep temporal on GPU (no D2H until lm_head).
        for &token_id in prompt {
            let emb = embed_lookup(&self.embed_cpu, token_id, self.d_model, self.vocab_size);
            self.input_cpu.copy_from_slice(emb);
            self.backbone.step_gpu_only(&self.input_cpu)?;
        }
        self.compute_logits()?;

        // Decode loop.
        let mut seen: Vec<u32> = prompt.to_vec();
        for _ in 0..params.max_tokens {
            let next = sample_token(&mut self.logits_cpu, params, &seen, &mut rng);
            if params.eos_token_ids.contains(&next) {
                break;
            }
            seen.push(next);
            cb(next, "");
            let emb = embed_lookup(&self.embed_cpu, next, self.d_model, self.vocab_size);
            self.input_cpu.copy_from_slice(emb);
            self.backbone.step_gpu_only(&self.input_cpu)?;
            self.compute_logits()?;
        }
        Ok(())
    }

    fn compute_logits(&mut self) -> Result<(), String> {
        let ctx = self.backbone.ctx();
        let stream = self.backbone.stream().clone();
        let temporal_ptr = self.backbone.temporal_ptr();

        match &self.embed_storage {
            EmbedStorage::F32 { embed, lm_head } => {
                if let Some(lm) = lm_head {
                    // Untied: logits[1,V] = hidden[1,D] @ lm_head[D,V]
                    // Need hidden as a GpuBuffer for sgemm; download+upload path.
                    self.backbone.download_temporal(&mut self.input_cpu)?;
                    self.gpu_hidden.upload(&stream, &self.input_cpu)?;
                    gpu_sgemm_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        &self.gpu_hidden,
                        lm.cached_ptr(),
                        None,
                        (1, self.d_model, self.vocab_size),
                    )?;
                } else {
                    // Tied: Y[V,1] = embed[V,D] @ temporal[D,1] — temporal as raw W ptr.
                    gpu_sgemm_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        embed,
                        temporal_ptr,
                        None,
                        (self.vocab_size_padded, self.d_model, 1),
                    )?;
                }
            }
            EmbedStorage::Half {
                embed,
                lm_head,
                dtype,
            } => {
                // Downcast temporal f32 → dtype via cast kernel (both GEMM inputs must match).
                let half_bytes = self.d_model * dtype.size_bytes();
                ctx.ensure_half_staging(half_bytes)?;
                let temporal_half_ptr = ctx.half_staging_ptr();
                {
                    use cudarc::driver::PushKernelArg;
                    let n = self.d_model as i32;
                    let kernel = match *dtype {
                        WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
                        WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
                        WeightDtype::F32 => unreachable!(),
                    };
                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(&temporal_half_ptr);
                    builder.arg(&temporal_ptr);
                    builder.arg(&n);
                    use crate::mamba_ssm::gpu::launch::grid_1d;
                    unsafe { builder.launch(grid_1d(self.d_model)) }
                        .map_err(|e| format!("cast temporal: {e:?}"))?;
                }

                if let Some(lm) = lm_head {
                    gpu_gemm_ex_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        temporal_half_ptr,
                        *dtype,
                        lm.cached_ptr(),
                        *dtype,
                        None,
                        (1, self.d_model, self.vocab_size),
                    )?;
                } else {
                    gpu_gemm_ex_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        embed.cached_ptr(),
                        *dtype,
                        temporal_half_ptr,
                        *dtype,
                        None,
                        (self.vocab_size_padded, self.d_model, 1),
                    )?;
                }
            }
        }

        stream
            .synchronize()
            .map_err(|e| format!("logits sync: {e:?}"))?;
        self.gpu_logits
            .download(&stream, &mut self.logits_padded_cpu)?;
        self.logits_cpu
            .copy_from_slice(&self.logits_padded_cpu[..self.vocab_size]);
        Ok(())
    }
}

/// Upload an f32 slice into a byte buffer at given element offset, converting to `dtype`.
fn upload_f32_as_dtype(
    dst: &GpuByteBuffer,
    elem_offset: usize,
    src: &[f32],
    src_elems: usize,
    dtype: WeightDtype,
) -> Result<(), String> {
    assert_eq!(src.len(), src_elems, "src size mismatch");
    let byte_off = elem_offset * dtype.size_bytes();
    let byte_count = src_elems * dtype.size_bytes();
    let dst_ptr = dst.cached_ptr() + byte_off as u64;

    match dtype {
        WeightDtype::F32 => {
            let bytes: &[u8] = bytemuck::cast_slice(src);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod(dst_ptr, bytes)
        }
        WeightDtype::Bf16 => {
            let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod(dst_ptr, bytes)
        }
        WeightDtype::F16 => {
            let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod(dst_ptr, bytes)
        }
    }
}

fn cu_memcpy_htod(dst_ptr: cudarc::driver::sys::CUdeviceptr, bytes: &[u8]) -> Result<(), String> {
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyHtoD_v2(
            dst_ptr,
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len(),
        )
    };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cu_memcpy_htod failed: {result:?}"));
    }
    Ok(())
}
