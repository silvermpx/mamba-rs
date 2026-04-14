//! GPU-accelerated language model wrapper for Mamba text generation.
//!
//! Uses `GpuMambaBackbone` for the SSM step and cuBLAS SGEMM for lm_head.
//! Sampling remains on CPU (negligible cost vs GPU compute).

use std::path::Path;

use crate::mamba_ssm::gpu::blas::{gpu_gemm_ex_forward_raw, gpu_sgemm_forward_raw};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::inference::{GpuMambaBackbone, GpuMambaBackboneMixed};

use crate::hf::embed::embed_lookup;
use crate::hf::load::{HfModel, load_hf};

use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};

pub struct GpuMambaLM {
    backbone: GpuMambaBackbone,
    gpu_embed: GpuBuffer,
    gpu_lm_head: Option<GpuBuffer>,
    gpu_hidden: GpuBuffer,
    gpu_logits: GpuBuffer,
    embed_cpu: Vec<f32>,
    logits_cpu: Vec<f32>,
    logits_padded_cpu: Vec<f32>,
    input_cpu: Vec<f32>,
    pub vocab_size: usize,
    vocab_size_padded: usize,
    pub d_model: usize,
}

impl GpuMambaLM {
    pub fn from_hf(dir: &Path, gpu_ordinal: usize) -> Result<Self, String> {
        let HfModel {
            backbone: cpu_backbone,
            embed,
            lm_head,
            vocab_size,
            vocab_size_padded,
            d_model,
        } = load_hf(dir)?;

        let cfg = *cpu_backbone.config();
        let gpu_bb = GpuMambaBackbone::new(gpu_ordinal, cpu_backbone.weights(), cfg, d_model, 1)?;

        let stream = gpu_bb.stream();

        let mut gpu_embed = GpuBuffer::zeros(stream, vocab_size_padded * d_model)?;
        gpu_embed.upload(stream, &embed)?;

        let gpu_lm_head = if let Some(ref lm_w) = lm_head {
            let mut buf = GpuBuffer::zeros(stream, lm_w.len())?;
            buf.upload(stream, lm_w)?;
            Some(buf)
        } else {
            None
        };

        let gpu_hidden = GpuBuffer::zeros(stream, d_model)?;
        let gpu_logits = GpuBuffer::zeros(stream, vocab_size_padded)?;

        Ok(Self {
            backbone: gpu_bb,
            gpu_embed,
            gpu_lm_head,
            gpu_hidden,
            gpu_logits,
            embed_cpu: embed,
            logits_cpu: vec![0.0; vocab_size],
            logits_padded_cpu: vec![0.0; vocab_size_padded],
            input_cpu: vec![0.0; d_model],
            vocab_size,
            vocab_size_padded,
            d_model,
        })
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.backbone.capture_graph()
    }

    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Result<Vec<u32>, String> {
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

        // Prefill: step-by-step, keep temporal on GPU (no D2H until lm_head)
        for &token_id in prompt {
            let emb = embed_lookup(&self.embed_cpu, token_id, self.d_model, self.vocab_size);
            self.input_cpu.copy_from_slice(emb);
            self.backbone.step_gpu_only(&self.input_cpu)?;
        }
        self.compute_logits_from_gpu()?;

        // Decode loop
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
            self.compute_logits_from_gpu()?;
        }

        Ok(())
    }

    fn compute_logits_from_gpu(&mut self) -> Result<(), String> {
        let ctx = self.backbone.ctx();
        let stream = self.backbone.stream();
        let temporal_ptr = self.backbone.temporal_ptr();

        if let Some(ref lm_head) = self.gpu_lm_head {
            // Untied: need temporal as GpuBuffer x-input.
            // D2H temporal → H2D to gpu_hidden (small: d_model floats).
            self.backbone.download_temporal(&mut self.input_cpu)?;
            self.gpu_hidden.upload(stream, &self.input_cpu)?;
            gpu_sgemm_forward_raw(
                ctx,
                &mut self.gpu_logits,
                &self.gpu_hidden,
                lm_head.cached_ptr(),
                None,
                (1, self.d_model, self.vocab_size),
            )?;
        } else {
            // Tied: Y[V,1] = embed[V,D] @ temporal[D,1] — temporal used as raw W ptr.
            gpu_sgemm_forward_raw(
                ctx,
                &mut self.gpu_logits,
                &self.gpu_embed,
                temporal_ptr,
                None,
                (self.vocab_size_padded, self.d_model, 1),
            )?;
        }

        stream
            .synchronize()
            .map_err(|e| format!("logits sync: {e:?}"))?;

        self.gpu_logits
            .download(stream, &mut self.logits_padded_cpu)?;
        self.logits_cpu
            .copy_from_slice(&self.logits_padded_cpu[..self.vocab_size]);

        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision GPU LM (bf16/fp16 weight storage, f32 compute).
// ---------------------------------------------------------------------------

/// GPU-accelerated Mamba LM with bf16/fp16 weight storage.
///
/// Same API as GpuMambaLM but:
/// - Backbone weights in bf16/fp16 (2x VRAM reduction for bulk linears)
/// - Embedding + lm_head stored as bulk dtype
/// - cuBLAS GemmEx with CUBLAS_COMPUTE_32F — f32 compute regardless of dtype
pub struct GpuMambaLMMixed {
    backbone: GpuMambaBackboneMixed,
    /// Bulk-dtype embedding arena: [vocab_size_padded * d_model] bytes.
    gpu_embed: GpuByteBuffer,
    /// Optional untied lm_head (also bulk dtype).
    gpu_lm_head: Option<GpuByteBuffer>,
    /// f32 staging buffer for hidden state (before lm_head).
    #[allow(dead_code)]
    gpu_hidden: GpuBuffer,
    /// f32 logits output.
    gpu_logits: GpuBuffer,
    embed_cpu: Vec<f32>,
    logits_cpu: Vec<f32>,
    logits_padded_cpu: Vec<f32>,
    input_cpu: Vec<f32>,
    pub vocab_size: usize,
    vocab_size_padded: usize,
    pub d_model: usize,
    dtype: WeightDtype,
}

impl GpuMambaLMMixed {
    pub fn from_hf(dir: &Path, gpu_ordinal: usize, dtype: WeightDtype) -> Result<Self, String> {
        let HfModel {
            backbone: cpu_backbone,
            embed,
            lm_head,
            vocab_size,
            vocab_size_padded,
            d_model,
        } = load_hf(dir)?;

        let cfg = *cpu_backbone.config();
        let gpu_bb = GpuMambaBackboneMixed::new(
            gpu_ordinal,
            cpu_backbone.weights(),
            cfg,
            d_model,
            1,
            dtype,
        )?;

        let stream = gpu_bb.stream();

        // Upload embed as bulk dtype bytes. embed is already [vocab_size_padded * d_model].
        let embed_bytes = embed.len() * dtype.size_bytes();
        let gpu_embed = GpuByteBuffer::zeros(stream, embed_bytes)?;
        upload_f32_as_dtype(&gpu_embed, 0, &embed, embed.len(), dtype)?;

        // Untied lm_head (if present).
        let gpu_lm_head = if let Some(ref lm_w) = lm_head {
            let lm_bytes = lm_w.len() * dtype.size_bytes();
            let buf = GpuByteBuffer::zeros(stream, lm_bytes)?;
            upload_f32_as_dtype(&buf, 0, lm_w, lm_w.len(), dtype)?;
            Some(buf)
        } else {
            None
        };

        let gpu_hidden = GpuBuffer::zeros(stream, d_model)?;
        let gpu_logits = GpuBuffer::zeros(stream, vocab_size_padded)?;

        Ok(Self {
            backbone: gpu_bb,
            gpu_embed,
            gpu_lm_head,
            gpu_hidden,
            gpu_logits,
            embed_cpu: embed,
            logits_cpu: vec![0.0; vocab_size],
            logits_padded_cpu: vec![0.0; vocab_size_padded],
            input_cpu: vec![0.0; d_model],
            vocab_size,
            vocab_size_padded,
            d_model,
            dtype,
        })
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.backbone.capture_graph()
    }

    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Result<Vec<u32>, String> {
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

        for &token_id in prompt {
            let emb = embed_lookup(&self.embed_cpu, token_id, self.d_model, self.vocab_size);
            self.input_cpu.copy_from_slice(emb);
            self.backbone.step_gpu_only(&self.input_cpu)?;
        }
        self.compute_logits_from_gpu()?;

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
            self.compute_logits_from_gpu()?;
        }
        Ok(())
    }

    fn compute_logits_from_gpu(&mut self) -> Result<(), String> {
        let ctx = self.backbone.ctx();
        let stream = self.backbone.stream();
        let temporal_ptr = self.backbone.temporal_ptr();

        // Downcast f32 temporal → bulk dtype via cast kernel into the half staging.
        // Both GEMM inputs must match dtype for cublasGemmEx.
        let half_bytes = self.d_model * self.dtype.size_bytes();
        ctx.ensure_half_staging(half_bytes)?;
        let temporal_half_ptr = ctx.half_staging_ptr();
        {
            use cudarc::driver::PushKernelArg;
            let n = self.d_model as i32;
            let kernel = match self.dtype {
                WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
                WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
                WeightDtype::F32 => unreachable!("F32 path should not reach mixed logits"),
            };
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&temporal_half_ptr);
            builder.arg(&temporal_ptr);
            builder.arg(&n);
            use crate::mamba_ssm::gpu::launch::grid_1d;
            unsafe { builder.launch(grid_1d(self.d_model)) }
                .map_err(|e| format!("cast temporal for logits: {e:?}"))?;
        }

        if let Some(ref lm_head) = self.gpu_lm_head {
            // Untied: Y[1,V] = hidden_half[1,D] @ lm_head[D,V]
            gpu_gemm_ex_forward_raw(
                ctx,
                &mut self.gpu_logits,
                temporal_half_ptr,
                self.dtype,
                lm_head.cached_ptr(),
                self.dtype,
                None,
                (1, self.d_model, self.vocab_size),
            )?;
        } else {
            // Tied: Y[V,1] = embed[V,D] @ temporal_half[D,1]
            gpu_gemm_ex_forward_raw(
                ctx,
                &mut self.gpu_logits,
                self.gpu_embed.cached_ptr(),
                self.dtype,
                temporal_half_ptr,
                self.dtype,
                None,
                (self.vocab_size_padded, self.d_model, 1),
            )?;
        }

        stream
            .synchronize()
            .map_err(|e| format!("logits sync: {e:?}"))?;

        self.gpu_logits
            .download(stream, &mut self.logits_padded_cpu)?;
        self.logits_cpu
            .copy_from_slice(&self.logits_padded_cpu[..self.vocab_size]);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }

    pub fn bulk_dtype(&self) -> WeightDtype {
        self.dtype
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
