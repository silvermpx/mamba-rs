//! Unified GPU-accelerated language model wrapper.
//!
//! Single `GpuMambaLM` struct supports f32 / bf16 / f16 weight storage.
//! Compute is always f32 (CUBLAS_COMPUTE_32F for GEMMs, f32 for custom kernels).
//! Dtype is chosen at construction via `from_hf_with_dtype`.

use std::path::Path;

use crate::mamba_ssm::gpu::blas::{
    TiedLmDims, TypedPtr, gpu_gemm_ex_forward_raw, gpu_gemm_ex_tied_lm_head_raw,
    gpu_sgemm_forward_raw, gpu_sgemm_tied_lm_head_raw,
};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::inference::GpuMambaBackbone;

/// Threshold: if a prompt has more tokens than this, use the parallel prefill
/// path (single kernel launch per layer over all T tokens) instead of the
/// step-by-step loop. Lower values favor parallel; typical LLM prompts ≥ 8
/// already benefit. Conservative default is 4 — parallel scan within a layer
/// amortizes the per-layer kernel-launch overhead over T tokens.
const PREFILL_PARALLEL_THRESHOLD: usize = 4;

use crate::hf::embed::embed_lookup;
use crate::hf::load::{HfModel, load_hf};

use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};
use rayon::prelude::*;

/// Threshold for parallelizing per-slot sampling in `generate_batch`. Below
/// this batch size the rayon job-submit overhead beats the per-slot
/// `sample_token` cost (greedy ≈ 12 µs, top-k+top-p ≈ 30-60 µs).
const SAMPLE_PARALLEL_THRESHOLD: usize = 8;

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
    /// Batch size (number of parallel sequences).
    pub batch: usize,
}

impl GpuMambaLM {
    /// Access the last-computed logits for batch slot `b`.
    /// Length = `vocab_size`. Valid after `generate` / `generate_batch`.
    pub fn last_logits(&self, b: usize) -> &[f32] {
        &self.logits_cpu[b * self.vocab_size..(b + 1) * self.vocab_size]
    }
}

impl GpuMambaLM {
    /// Load HF model with f32 storage, batch=1.
    pub fn from_hf(dir: &Path, gpu_ordinal: usize) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch(dir, gpu_ordinal, WeightDtype::F32, 1)
    }

    /// Load HF model with explicit storage dtype, batch=1.
    pub fn from_hf_with_dtype(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch(dir, gpu_ordinal, dtype, 1)
    }

    /// Load HF model with explicit dtype and batch size.
    ///
    /// `batch > 1` enables parallel generation of multiple independent
    /// sequences sharing the same weights. Each batch slot has its own
    /// recurrent state. Use `generate_batch` to drive them.
    pub fn from_hf_with_dtype_batch(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
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
            batch,
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

        let gpu_logits = GpuBuffer::zeros(stream, batch * vocab_size_padded)?;
        let gpu_hidden = GpuBuffer::zeros(stream, batch * d_model)?;

        Ok(Self {
            backbone,
            embed_storage,
            embed_cpu: embed,
            input_cpu: vec![0.0; batch * d_model],
            gpu_logits,
            gpu_hidden,
            logits_padded_cpu: vec![0.0; batch * vocab_size_padded],
            logits_cpu: vec![0.0; batch * vocab_size],
            vocab_size,
            vocab_size_padded,
            d_model,
            batch,
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

    /// Download the backbone's temporal buffer (last-layer hidden state, last
    /// timestep) as f32 regardless of storage dtype. Intended for debugging /
    /// parity harnesses.
    pub fn debug_download_temporal(&self, out: &mut [f32]) -> Result<(), String> {
        self.backbone.download_temporal(out)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
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
        assert_eq!(
            self.batch, 1,
            "generate_streaming requires batch=1; use generate_batch for batch>1"
        );
        self.backbone.reset()?;
        let mut rng = Xoshiro256PlusPlus::new(params.seed);

        // Prefill: parallel (one kernel per layer over all T) if long enough;
        // otherwise step-by-step (lower overhead for small T).
        if prompt.len() >= PREFILL_PARALLEL_THRESHOLD {
            self.prefill_parallel(prompt)?;
        } else {
            for &token_id in prompt {
                let emb = embed_lookup(&self.embed_cpu, token_id, self.d_model, self.vocab_size);
                self.input_cpu[..self.d_model].copy_from_slice(emb);
                self.backbone
                    .step_gpu_only(&self.input_cpu[..self.d_model])?;
            }
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

    /// Batch generation: generate N sequences in parallel.
    ///
    /// `prompts.len()` must equal `self.batch`. All prompts may have different
    /// lengths; shorter prompts are padded with their last token during prefill
    /// (this affects nothing since their output is discarded until they finish
    /// prefill). Each slot uses per-slot RNG seeded from `params[i].seed` and
    /// its own EOS token list from `params[i].eos_token_ids`.
    ///
    /// Returns one token vector per slot, up to each slot's `max_tokens` or
    /// EOS, whichever is first. Generation stops when ALL slots are finished.
    pub fn generate_batch(
        &mut self,
        prompts: &[&[u32]],
        params: &[SampleParams],
    ) -> Result<Vec<Vec<u32>>, String> {
        assert_eq!(prompts.len(), self.batch, "prompts.len() != batch");
        assert_eq!(params.len(), self.batch, "params.len() != batch");

        self.backbone.reset()?;
        let mut rngs: Vec<Xoshiro256PlusPlus> = params
            .iter()
            .map(|p| Xoshiro256PlusPlus::new(p.seed))
            .collect();

        let b = self.batch;
        let d = self.d_model;
        let vocab_size = self.vocab_size;
        let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let max_tokens = params.iter().map(|p| p.max_tokens).max().unwrap_or(0);

        // Per-slot state for streaming generation loop.
        let mut prompt_pos = vec![0usize; b]; // how many prompt tokens consumed
        let mut finished = vec![false; b];
        let mut outputs: Vec<Vec<u32>> = (0..b).map(|_| Vec::new()).collect();
        // `last_token[slot]` = token to feed next step for this slot.
        let mut last_token = vec![0u32; b];

        // Initial input: first prompt token per slot.
        for i in 0..b {
            if prompts[i].is_empty() {
                finished[i] = true;
                continue;
            }
            last_token[i] = prompts[i][0];
            prompt_pos[i] = 1; // we will feed this token in first step
        }

        let total_steps = max_prompt + max_tokens;

        for _step in 0..total_steps {
            if finished.iter().all(|&f| f) {
                break;
            }

            // Build input batch [b * d_model]: embed lookup per slot.
            for i in 0..b {
                if finished[i] {
                    // Feed zero vector for finished slots (their state update is discarded).
                    for v in &mut self.input_cpu[i * d..(i + 1) * d] {
                        *v = 0.0;
                    }
                } else {
                    let emb = embed_lookup(&self.embed_cpu, last_token[i], d, vocab_size);
                    self.input_cpu[i * d..(i + 1) * d].copy_from_slice(emb);
                }
            }

            // GPU step (all slots in parallel).
            self.backbone.step_gpu_only(&self.input_cpu)?;

            // Compute logits [b * vocab_size_padded] → download to CPU.
            self.compute_logits()?;

            // Per-slot decision. Sampling for decode-phase slots is
            // parallelized across rayon workers when batch is large enough;
            // each slot has disjoint logits (par_chunks_mut), an independent
            // RNG, and read-only access to its own params/outputs. The
            // resulting tokens are applied to per-slot mutable state in a
            // serial pass below (cheap scalar updates, no contention).
            let need_decode_for_slot: Vec<bool> = (0..b)
                .map(|i| !finished[i] && prompt_pos[i] >= prompts[i].len())
                .collect();
            let new_tokens: Vec<Option<u32>> = if b >= SAMPLE_PARALLEL_THRESHOLD {
                self.logits_cpu
                    .par_chunks_mut(vocab_size)
                    .zip(rngs.par_iter_mut())
                    .enumerate()
                    .map(|(i, (slot_logits, rng))| {
                        if need_decode_for_slot[i] {
                            Some(sample_token(slot_logits, &params[i], &outputs[i], rng))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                (0..b)
                    .map(|i| {
                        if need_decode_for_slot[i] {
                            let slot_logits =
                                &mut self.logits_cpu[i * vocab_size..(i + 1) * vocab_size];
                            Some(sample_token(
                                slot_logits,
                                &params[i],
                                &outputs[i],
                                &mut rngs[i],
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            };

            for i in 0..b {
                if finished[i] {
                    continue;
                }
                if prompt_pos[i] < prompts[i].len() {
                    // Still prefilling — feed next prompt token; sampler skipped.
                    last_token[i] = prompts[i][prompt_pos[i]];
                    prompt_pos[i] += 1;
                    continue;
                }
                let next = new_tokens[i].expect("decode-phase slot must have a sampled token");
                if params[i].eos_token_ids.contains(&next)
                    || outputs[i].len() >= params[i].max_tokens
                {
                    finished[i] = true;
                    continue;
                }
                outputs[i].push(next);
                last_token[i] = next;
            }
        }

        Ok(outputs)
    }

    /// Parallel prefill: uploads all T prompt embeddings to GPU at once and
    /// runs one burnin forward per layer (vs T step calls). After this call,
    /// backbone state is at position T and temporal holds the last timestep
    /// hidden state — ready for lm_head + decode.
    fn prefill_parallel(&mut self, prompt: &[u32]) -> Result<(), String> {
        let t = prompt.len();
        let d = self.d_model;
        let b = self.batch;
        let stream = self.backbone.stream().clone();

        // Build flat embed input [B*T*d_model] on CPU (batch=1 for now; batched
        // prefill with different prompt lengths per slot uses step-by-step).
        let mut embed_flat = vec![0.0f32; b * t * d];
        for ti in 0..t {
            let emb = embed_lookup(&self.embed_cpu, prompt[ti], d, self.vocab_size);
            // batch=1 case: sample 0, timestep ti
            embed_flat[ti * d..(ti + 1) * d].copy_from_slice(emb);
        }

        // Upload to GPU.
        let mut ip_out_flat = GpuBuffer::zeros(&stream, b * t * d)?;
        ip_out_flat.upload(&stream, &embed_flat)?;

        // Allocate prefill scratch and dispatch on backbone dtype.
        // Mixed backbone → native bf16/f16 prefill (DtypedBuf scratch).
        // F32 backbone → f32 prefill (GpuBuffer scratch).
        match self.backbone.dtype() {
            WeightDtype::F32 => {
                let mut prefill_scratch = self.backbone.alloc_prefill_scratch(t)?;
                self.backbone
                    .prefill_sequence(&ip_out_flat, &mut prefill_scratch)?;
            }
            WeightDtype::Bf16 | WeightDtype::F16 => {
                let mut prefill_scratch = self.backbone.alloc_prefill_mixed_scratch(t)?;
                self.backbone
                    .prefill_sequence_mixed(&ip_out_flat, &mut prefill_scratch)?;
            }
        }

        Ok(())
    }

    fn compute_logits(&mut self) -> Result<(), String> {
        let ctx = self.backbone.ctx();
        let stream = self.backbone.stream().clone();
        let temporal_ptr = self.backbone.temporal_ptr();
        let b = self.batch;
        let d = self.d_model;

        match &self.embed_storage {
            EmbedStorage::F32 { embed, lm_head } => {
                if let Some(lm) = lm_head {
                    // Untied: logits[B,V] = hidden[B,D] @ lm_head[D,V]
                    // Download batched temporal → upload to gpu_hidden → single batched SGEMM.
                    self.backbone.download_temporal(&mut self.input_cpu)?;
                    self.gpu_hidden.upload(&stream, &self.input_cpu)?;
                    gpu_sgemm_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        &self.gpu_hidden,
                        lm.cached_ptr(),
                        None,
                        (b, d, self.vocab_size),
                    )?;
                } else {
                    // Tied: logits[B,V] = temporal[B,D] @ embed^T[D,V]
                    // Single SGEMM via OP_T on embed (reuses row-major [V,D] buffer).
                    gpu_sgemm_tied_lm_head_raw(
                        ctx,
                        self.gpu_logits.cached_ptr(),
                        temporal_ptr,
                        embed.cached_ptr(),
                        b,
                        d,
                        self.vocab_size_padded,
                    )?;
                }
            }
            EmbedStorage::Half {
                embed,
                lm_head,
                dtype,
            } => {
                // With end-to-end bf16 inference, temporal is already in `dtype`
                // from the Mixed engine — feed directly into lm_head, no staging.
                // Legacy fall-back: if temporal is still f32 (e.g., prefill path
                // for mixed hasn't been fully migrated yet), downcast once.
                let backbone_dtype = self.backbone.temporal_dtype();
                let temporal_half_ptr = if backbone_dtype == *dtype {
                    temporal_ptr
                } else {
                    // Legacy: f32 temporal → cast to half into half_staging.
                    let half_bytes = b * d * dtype.size_bytes();
                    ctx.ensure_half_staging(half_bytes)?;
                    let staging_ptr = ctx.half_staging_ptr();
                    use cudarc::driver::PushKernelArg;
                    let n = (b * d) as i32;
                    let kernel = match *dtype {
                        WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
                        WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
                        WeightDtype::F32 => unreachable!(),
                    };
                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(&staging_ptr);
                    builder.arg(&temporal_ptr);
                    builder.arg(&n);
                    use crate::mamba_ssm::gpu::launch::grid_1d;
                    unsafe { builder.launch(grid_1d(b * d)) }
                        .map_err(|e| format!("cast temporal: {e:?}"))?;
                    staging_ptr
                };

                if let Some(lm) = lm_head {
                    // Untied: Y[B,V] = temporal_half[B,D] @ lm_head[D,V]
                    gpu_gemm_ex_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        TypedPtr {
                            ptr: temporal_half_ptr,
                            dtype: *dtype,
                        },
                        TypedPtr {
                            ptr: lm.cached_ptr(),
                            dtype: *dtype,
                        },
                        None,
                        (b, d, self.vocab_size),
                    )?;
                } else {
                    // Tied: logits[B,V] = temporal_half[B,D] @ embed^T[D,V]
                    gpu_gemm_ex_tied_lm_head_raw(
                        ctx,
                        self.gpu_logits.cached_ptr(),
                        temporal_half_ptr,
                        embed.cached_ptr(),
                        *dtype,
                        TiedLmDims {
                            batch: b,
                            d_model: d,
                            vocab_padded: self.vocab_size_padded,
                        },
                    )?;
                }
            }
        }

        stream
            .synchronize()
            .map_err(|e| format!("logits sync: {e:?}"))?;
        self.gpu_logits
            .download(&stream, &mut self.logits_padded_cpu)?;
        // Both tied and untied paths produce row-major [B, vocab_padded].
        // Slice off padding per slot.
        for bi in 0..self.batch {
            let src = &self.logits_padded_cpu
                [bi * self.vocab_size_padded..bi * self.vocab_size_padded + self.vocab_size];
            let dst = &mut self.logits_cpu[bi * self.vocab_size..(bi + 1) * self.vocab_size];
            dst.copy_from_slice(src);
        }
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
