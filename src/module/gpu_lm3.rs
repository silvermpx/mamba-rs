//! GPU-accelerated language model wrapper for Mamba-3 SISO.
//!
//! Mirrors `GpuMambaLM` (Mamba-1). Same unified API over f32 / bf16 / f16
//! weight storage — dtype chosen at construction via `from_weights_with_dtype`.
//!
//! No real Mamba-3 SISO HF checkpoint is published yet. This wrapper drives
//! a synthetic-weight model + embedding + optional lm_head; useful for
//! benchmarking and integration testing until HF M3 weights land.

use std::sync::Arc;

use crate::mamba_ssm::gpu::blas::{
    TiedLmDims, TypedPtr, gpu_gemm_ex_tied_lm_head_blas, gpu_gemm_typed_raw_no_bias,
    gpu_sgemm_tied_lm_head_blas,
};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::inference::GpuMamba3Backbone;
use crate::mamba3_siso::weights::Mamba3Weights;

use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};
use rayon::prelude::*;

/// Mirror of the M1 LM threshold (`SAMPLE_PARALLEL_THRESHOLD`): below this
/// batch size, rayon job-submit overhead beats the per-slot `sample_token`
/// cost for typical sampling configs. See `module::gpu_lm` for rationale.
const SAMPLE_PARALLEL_THRESHOLD: usize = 8;

/// Internal: embed + optional lm_head storage. F32 uses typed GpuBuffer;
/// bf16/f16 use GpuByteBuffer (raw bytes, typed via `dtype`).
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

/// Unified GPU Mamba-3 language model.
///
/// Wraps `GpuMamba3Backbone` + embedding table + optional lm_head. Mirrors
/// the M1 `GpuMambaLM` API:
/// - `from_weights(cpu_weights, embed, lm_head, cfg, gpu)` — f32 storage.
/// - `from_weights_with_dtype(..., dtype)` — f32 / bf16 / f16 storage.
///
/// Since no HF Mamba-3 SISO checkpoint is public, construction takes CPU
/// weights + embedding + optional lm_head directly (no HF loader).
pub struct GpuMamba3LM {
    backbone: GpuMamba3Backbone,
    embed_storage: EmbedStorage,
    embed_cpu: Vec<f32>,
    input_cpu: Vec<f32>,
    gpu_logits: GpuBuffer,
    gpu_hidden: GpuBuffer,
    logits_padded_cpu: Vec<f32>,
    logits_cpu: Vec<f32>,
    pub vocab_size: usize,
    vocab_size_padded: usize,
    pub d_model: usize,
    pub batch: usize,
}

impl GpuMamba3LM {
    pub fn last_logits(&self, b: usize) -> &[f32] {
        &self.logits_cpu[b * self.vocab_size..(b + 1) * self.vocab_size]
    }
}

/// Arguments bundle for `GpuMamba3LM::build`.
pub struct Mamba3LmBuild<'a> {
    pub cpu_weights: &'a Mamba3Weights,
    pub cfg: Mamba3Config,
    pub embed: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub vocab_size: usize,
    pub gpu_ordinal: usize,
    pub dtype: WeightDtype,
    pub batch: usize,
}

impl GpuMamba3LM {
    /// F32 construction shortcut.
    pub fn from_weights(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
    ) -> Result<Self, String> {
        Self::build(Mamba3LmBuild {
            cpu_weights,
            cfg,
            embed,
            lm_head,
            vocab_size,
            gpu_ordinal,
            dtype: WeightDtype::F32,
            batch: 1,
        })
    }

    /// Batch=1 dtype-aware construction.
    pub fn from_weights_with_dtype(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::build(Mamba3LmBuild {
            cpu_weights,
            cfg,
            embed,
            lm_head,
            vocab_size,
            gpu_ordinal,
            dtype,
            batch: 1,
        })
    }

    /// Dtype + batch-size construction — mirrors
    /// `GpuMambaLM::from_hf_with_dtype_batch` on the M1 side so `generate_batch`
    /// has a flat convenience entry point on M3 too.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_with_dtype_batch(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
    ) -> Result<Self, String> {
        Self::build(Mamba3LmBuild {
            cpu_weights,
            cfg,
            embed,
            lm_head,
            vocab_size,
            gpu_ordinal,
            dtype,
            batch,
        })
    }

    /// Full constructor.
    ///
    /// `embed`: `[vocab_size_padded * d_model]` row-major; `lm_head` (if
    /// untied): `[d_model * vocab_size]`. Tied lm_head → pass `None` and the
    /// embed table is reused as the tied projection matrix.
    pub fn build(args: Mamba3LmBuild<'_>) -> Result<Self, String> {
        let Mamba3LmBuild {
            cpu_weights,
            cfg,
            embed,
            lm_head,
            vocab_size,
            gpu_ordinal,
            dtype,
            batch,
        } = args;
        let d_model = cfg.d_model;
        assert_eq!(
            embed.len() % d_model,
            0,
            "embed size must be multiple of d_model"
        );
        let vocab_size_padded = embed.len() / d_model;
        assert!(
            vocab_size_padded >= vocab_size,
            "vocab_size_padded ({vocab_size_padded}) < vocab_size ({vocab_size})"
        );

        // For identity_proj construction: clear input_proj_w so the engine
        // takes the no-proj fast path (required by Mixed engine).
        let mut weights = cpu_weights.clone();
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();

        let backbone =
            GpuMamba3Backbone::new_with_dtype(gpu_ordinal, &weights, cfg, d_model, batch, dtype)?;
        let stream = backbone.stream();

        // Pad untied lm_head to `vocab_size_padded` rows so the untied GEMM
        // can write into `gpu_logits` with the same row stride as the tied
        // path. Without padding the GEMM emits contiguous [B, vocab_size]
        // while the CPU-side downloader reads with stride vocab_size_padded
        // → every batch slot beyond the first gets wrong logits on any
        // checkpoint whose vocab isn't 64-aligned. Same bug + same fix that
        // landed in M1 (GpuMambaLM) at commit 5dde438.
        let lm_head_padded: Option<Vec<f32>> = lm_head.as_ref().map(|lm| {
            if vocab_size == vocab_size_padded {
                lm.clone()
            } else {
                let mut padded = vec![0.0f32; vocab_size_padded * d_model];
                padded[..vocab_size * d_model].copy_from_slice(lm);
                padded
            }
        });

        let embed_storage = match dtype {
            WeightDtype::F32 => {
                let mut e = GpuBuffer::zeros(stream, embed.len())?;
                e.upload(stream, &embed)?;
                let lm = if let Some(ref lm_w) = lm_head_padded {
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
                upload_f32_as_dtype(&e, &embed, dtype)?;
                let lm = if let Some(ref lm_w) = lm_head_padded {
                    let lm_bytes = lm_w.len() * dtype.size_bytes();
                    let b = GpuByteBuffer::zeros(stream, lm_bytes)?;
                    upload_f32_as_dtype(&b, lm_w, dtype)?;
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

        // Mamba-3 only supports step-by-step prefill for now (chunked SSD
        // kernels aren't templated for bf16 — would require mixed parallel
        // prefill work comparable to the M3 step pipeline itself).
        for &token_id in prompt {
            let emb = embed_lookup(&self.embed_cpu, token_id, self.d_model, self.vocab_size);
            self.input_cpu[..self.d_model].copy_from_slice(emb);
            self.backbone
                .step_gpu_only(&self.input_cpu[..self.d_model])?;
        }
        self.compute_logits()?;

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

        let mut prompt_pos = vec![0usize; b];
        let mut finished = vec![false; b];
        let mut outputs: Vec<Vec<u32>> = (0..b).map(|_| Vec::new()).collect();
        let mut last_token = vec![0u32; b];

        for i in 0..b {
            if prompts[i].is_empty() {
                finished[i] = true;
                continue;
            }
            last_token[i] = prompts[i][0];
            prompt_pos[i] = 1;
        }

        let total_steps = max_prompt + max_tokens;
        for _step in 0..total_steps {
            if finished.iter().all(|&f| f) {
                break;
            }
            for i in 0..b {
                if finished[i] {
                    for v in &mut self.input_cpu[i * d..(i + 1) * d] {
                        *v = 0.0;
                    }
                } else {
                    let emb = embed_lookup(&self.embed_cpu, last_token[i], d, vocab_size);
                    self.input_cpu[i * d..(i + 1) * d].copy_from_slice(emb);
                }
            }
            self.backbone.step_gpu_only(&self.input_cpu)?;
            self.compute_logits()?;

            // Per-slot sampling. Decode-phase slots are computed in parallel
            // when batch >= SAMPLE_PARALLEL_THRESHOLD; mutable state is then
            // applied serially. Each slot has disjoint logits + independent RNG.
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

    fn compute_logits(&mut self) -> Result<(), String> {
        let stream = self.backbone.stream().clone();
        let blas: Arc<cudarc::cublas::CudaBlas> = self.backbone.blas().clone();
        let temporal_ptr = self.backbone.temporal_ptr();
        let b = self.batch;
        let d = self.d_model;

        match &self.embed_storage {
            EmbedStorage::F32 { embed, lm_head } => {
                if let Some(lm) = lm_head {
                    // Untied: logits[B,Vpad] = hidden[B,D] @ lm_head[D,Vpad].
                    // `vocab_size_padded` matches lm_head and gpu_logits row
                    // stride — see padding at lines 194-203 and same-bug fix
                    // in M1 (commit 5dde438).
                    self.backbone.download_temporal(&mut self.input_cpu)?;
                    self.gpu_hidden.upload(&stream, &self.input_cpu)?;
                    gpu_gemm_typed_raw_no_bias(
                        &blas,
                        TypedPtr {
                            ptr: self.gpu_logits.cached_ptr(),
                            dtype: WeightDtype::F32,
                        },
                        TypedPtr {
                            ptr: self.gpu_hidden.cached_ptr(),
                            dtype: WeightDtype::F32,
                        },
                        TypedPtr {
                            ptr: lm.cached_ptr(),
                            dtype: WeightDtype::F32,
                        },
                        (b, d, self.vocab_size_padded),
                    )?;
                } else {
                    gpu_sgemm_tied_lm_head_blas(
                        &blas,
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
                let backbone_dtype = self.backbone.temporal_dtype();
                assert_eq!(
                    backbone_dtype, *dtype,
                    "M3 LM mixed path expects end-to-end matching dtypes (temporal + embed)"
                );

                if let Some(lm) = lm_head {
                    // Untied half path — same padded stride as F32 path above.
                    gpu_gemm_typed_raw_no_bias(
                        &blas,
                        TypedPtr {
                            ptr: self.gpu_logits.cached_ptr(),
                            dtype: WeightDtype::F32,
                        },
                        TypedPtr {
                            ptr: temporal_ptr,
                            dtype: *dtype,
                        },
                        TypedPtr {
                            ptr: lm.cached_ptr(),
                            dtype: *dtype,
                        },
                        (b, d, self.vocab_size_padded),
                    )?;
                } else {
                    gpu_gemm_ex_tied_lm_head_blas(
                        &blas,
                        self.gpu_logits.cached_ptr(),
                        temporal_ptr,
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
            .map_err(|e| format!("M3 logits sync: {e:?}"))?;
        self.gpu_logits
            .download(&stream, &mut self.logits_padded_cpu)?;
        for bi in 0..self.batch {
            let src = &self.logits_padded_cpu
                [bi * self.vocab_size_padded..bi * self.vocab_size_padded + self.vocab_size];
            let dst = &mut self.logits_cpu[bi * self.vocab_size..(bi + 1) * self.vocab_size];
            dst.copy_from_slice(src);
        }
        Ok(())
    }
}

fn embed_lookup(embed: &[f32], token_id: u32, d_model: usize, vocab_size: usize) -> &[f32] {
    let id = (token_id as usize).min(vocab_size.saturating_sub(1));
    &embed[id * d_model..(id + 1) * d_model]
}

fn upload_f32_as_dtype(dst: &GpuByteBuffer, src: &[f32], dtype: WeightDtype) -> Result<(), String> {
    let dst_ptr = dst.cached_ptr();
    match dtype {
        WeightDtype::F32 => {
            let bytes: &[u8] = bytemuck::cast_slice(src);
            cu_memcpy_htod(dst_ptr, bytes)
        }
        WeightDtype::Bf16 => {
            let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            cu_memcpy_htod(dst_ptr, bytes)
        }
        WeightDtype::F16 => {
            let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
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
        return Err(format!("M3 cu_memcpy_htod failed: {result:?}"));
    }
    Ok(())
}
