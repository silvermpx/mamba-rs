//! GPU-accelerated language model wrapper for Mamba text generation.
//!
//! Uses `GpuMambaBackbone` for the SSM step and cuBLAS SGEMM for lm_head.
//! Sampling remains on CPU (negligible cost vs GPU compute).

use std::path::Path;

use crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::inference::GpuMambaBackbone;

use crate::hf::embed::embed_lookup;
use crate::hf::load::{load_hf, HfModel};

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
        let gpu_bb = GpuMambaBackbone::new(
            gpu_ordinal,
            cpu_backbone.weights(),
            cfg,
            d_model,
            1,
        )?;

        let stream = gpu_bb.stream();

        let mut gpu_embed = GpuBuffer::zeros(&stream, vocab_size_padded * d_model)?;
        gpu_embed.upload(&stream, &embed)?;

        let gpu_lm_head = if let Some(ref lm_w) = lm_head {
            let mut buf = GpuBuffer::zeros(&stream, lm_w.len())?;
            buf.upload(&stream, lm_w)?;
            Some(buf)
        } else {
            None
        };

        let gpu_hidden = GpuBuffer::zeros(&stream, d_model)?;
        let gpu_logits = GpuBuffer::zeros(&stream, vocab_size_padded)?;

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
            self.gpu_hidden.upload(&stream, &self.input_cpu)?;
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
            .download(&stream, &mut self.logits_padded_cpu)?;
        self.logits_cpu
            .copy_from_slice(&self.logits_padded_cpu[..self.vocab_size]);

        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }
}
