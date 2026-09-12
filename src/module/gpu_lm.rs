//! Unified GPU-accelerated language model wrapper.
//!
//! Single `GpuMambaLM` struct supports f32 / bf16 / f16 weight storage.
//! Compute is always f32 (CUBLAS_COMPUTE_32F for GEMMs, f32 for custom kernels).
//! Dtype is chosen at construction via `from_hf_with_dtype`.

use std::path::Path;

use crate::mamba_ssm::gpu::blas::{
    TiedLmDims, TypedPtr, gpu_gemm_bi_forward_ptr, gpu_gemm_bi_tied_lm_head_raw,
    gpu_gemm_ex_forward_raw, gpu_gemm_ex_tied_lm_head_raw, presize_tied_lm_head_scratch,
};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::context::GemmMode;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::inference::GpuMambaBackbone;

/// Threshold: if a prompt has more tokens than this, use the parallel prefill
/// path (single kernel launch per layer over all T tokens) instead of the
/// step-by-step loop. Lower values favor parallel; typical LLM prompts ≥ 8
/// already benefit. Conservative default is 4 — parallel scan within a layer
/// amortizes the per-layer kernel-launch overhead over T tokens.
/// Minimum prompt length before `generate_streaming` (batch=1) switches
/// from per-token step-by-step prefill to the parallel prefill kernel.
///
/// HISTORY: was `4`, which meant a 5-token prompt used parallel prefill
/// at batch=1 while the same prompt in `generate_batch` (batch > 1)
/// always uses step-by-step. The two paths are mathematically equivalent
/// but have different numerical precision at bf16 — parallel scan and
/// sequential scan reduce the K-dim in different orders, producing sub-
/// ULP differences that amplify through 24 SSM layers on adversarial
/// prompts (KL ≈ 2.7 on [100..104], originally reported in
/// `bf16_batch_divergence_known`).
///
/// Raised to 64 so typical chat-style prompts (≤ 63 tokens) go through
/// the identical step-by-step path as batched generation, restoring
/// batch-invariant inference at bf16. Longer prompts (documents, RAG
/// contexts) still get the parallel-scan perf win where it matters.
/// For a 5-token prompt the step-by-step path is ~2 ms slower on 130m
/// at bf16 — negligible for any realistic latency budget.
/// Threshold above which a single-batch prefill uses the parallel
/// (T-batched) SSM kernel instead of per-step (T=1) decode. Above this
/// threshold, prefill is faster but uses a different SSM implementation
/// from per-step decode — resulting in a tiny bf16 rounding divergence
/// between `b=1` prefill and `b>1` per-step at the same prompt. Kept
/// high enough that mainstream decode tests stay on the unified T=1
/// path (bit-identical cross-batch); long-context prefills (≥256)
/// trade that for parallel-prefill speed.
const PREFILL_PARALLEL_THRESHOLD: usize = 256;

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
    /// CPU mirror of logits (padded).
    logits_padded_cpu: Vec<f32>,
    /// CPU logits clamped to real vocab_size.
    logits_cpu: Vec<f32>,
    /// Vocabulary size — the number of valid token IDs.
    pub vocab_size: usize,
    vocab_size_padded: usize,
    /// Backbone hidden width (`d_model` in the paper).
    pub d_model: usize,
    /// Batch size (number of parallel sequences).
    pub batch: usize,
}

impl GpuMambaLM {
    /// CUDA context of the underlying backbone.
    ///
    /// Inspect the selected execution policy with
    /// [`crate::mamba_ssm::gpu::context::GpuCtx::gemm_mode`] and
    /// [`crate::mamba_ssm::gpu::context::GpuCtx::bi_gemm_family`]. Storage is
    /// reported separately by [`Self::dtype`]. Graph capture binds this route.
    pub fn ctx(&self) -> &crate::mamba_ssm::gpu::context::GpuCtx {
        self.backbone.ctx()
    }

    /// Access the last-computed logits for batch slot `b`.
    /// Length = `vocab_size`. Valid after `generate` / `generate_batch`.
    pub fn last_logits(&self, b: usize) -> &[f32] {
        &self.logits_cpu[b * self.vocab_size..(b + 1) * self.vocab_size]
    }
}

impl GpuMambaLM {
    /// Load an HF model with f32 storage, a batch of one, and env-selected GEMMs.
    ///
    /// Missing selectors use Deterministic + Inference; invalid or conflicting
    /// selectors are errors. Use [`Self::from_hf_with_mode`] for an explicit
    /// mode and [`Self::ctx`] to inspect the route that graph capture binds.
    /// Errors match [`Self::from_hf_with_dtype_batch`]; `MAMBA_RS_ARCH_RUNG`
    /// remains a separate first-use Inference policy.
    pub fn from_hf(dir: &Path, gpu_ordinal: usize) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch(dir, gpu_ordinal, WeightDtype::F32, 1)
    }

    /// Load an HF model with f32 storage, a batch of one, and an explicit GEMM mode.
    ///
    /// GEMM mode, custom precision/tensor-core controls, and family selectors
    /// in the environment are ignored. `MAMBA_RS_ARCH_RUNG` remains the
    /// separate first-use Inference policy and is not captured by this
    /// constructor. Loading, validation, CUDA setup, upload, and allocation
    /// failures are returned.
    pub fn from_hf_with_mode(
        dir: &Path,
        gpu_ordinal: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch_inner(dir, gpu_ordinal, WeightDtype::F32, 1, Some(mode))
    }

    /// Load an HF model with explicit storage dtype and env-selected GEMMs.
    ///
    /// `dtype` controls storage independently of execution mode. Missing
    /// selectors use Deterministic + Inference; invalid or conflicting values
    /// are errors. See [`Self::from_hf_with_dtype_and_mode`] for explicit mode;
    /// other errors match [`Self::from_hf_with_dtype_batch`].
    pub fn from_hf_with_dtype(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch(dir, gpu_ordinal, dtype, 1)
    }

    /// Load an HF model with explicit storage dtype and GEMM mode, a batch of one.
    ///
    /// Storage precision and GEMM execution are independent. The explicit
    /// lane ignores GEMM mode, custom precision/tensor-core controls, and
    /// family selectors in the environment; `MAMBA_RS_ARCH_RUNG` remains a
    /// separate first-use policy and is not captured by this constructor.
    /// Errors match [`Self::from_hf_with_dtype`].
    pub fn from_hf_with_dtype_and_mode(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch_inner(dir, gpu_ordinal, dtype, 1, Some(mode))
    }

    /// Load HF model with explicit dtype and batch size.
    ///
    /// `batch > 1` enables parallel generation of multiple independent
    /// sequences sharing the same weights. Each batch slot has its own
    /// recurrent state. Storage remains independent of GEMM execution. Missing
    /// selectors use Deterministic + Inference; invalid or conflicting values
    /// are errors. Use [`Self::generate_batch`] to drive the result and
    /// [`Self::ctx`] to inspect its route. `MAMBA_RS_ARCH_RUNG` remains a
    /// separate first-use Inference policy.
    ///
    /// # Errors
    ///
    /// Returns checkpoint, configuration, GEMM-environment, CUDA, upload, or
    /// batch-dependent allocation failures.
    pub fn from_hf_with_dtype_batch(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch_inner(dir, gpu_ordinal, dtype, batch, None)
    }

    /// Load an HF model with explicit storage dtype, batch size, and GEMM mode.
    ///
    /// `dtype` controls storage and `mode` independently controls GEMM
    /// execution. GEMM environment selectors are bypassed except that
    /// `MAMBA_RS_ARCH_RUNG` remains the separate first-use Inference policy and
    /// is not captured by this constructor. Invalid checkpoints, model
    /// configuration, batch-dependent allocation, CUDA setup, or upload
    /// failures are returned. Graph capture remains tied to the complete route.
    pub fn from_hf_with_dtype_batch_and_mode(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::from_hf_with_dtype_batch_inner(dir, gpu_ordinal, dtype, batch, Some(mode))
    }

    fn from_hf_with_dtype_batch_inner(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
        mode: Option<GemmMode>,
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
        let backbone = match mode {
            Some(mode) => GpuMambaBackbone::new_with_dtype_and_mode(
                gpu_ordinal,
                cpu_backbone.weights(),
                cfg,
                d_model,
                batch,
                dtype,
                mode,
            )?,
            None => GpuMambaBackbone::new_with_dtype(
                gpu_ordinal,
                cpu_backbone.weights(),
                cfg,
                d_model,
                batch,
                dtype,
            )?,
        };
        let stream = backbone.stream();

        // Pad the untied lm_head to `vocab_size_padded` columns so the
        // untied GEMM can write into `gpu_logits` with the SAME row stride
        // (`vocab_size_padded`) as the tied path. Without padding, the
        // GEMM writes contiguous rows of `vocab_size` while the CPU reads
        // with stride `vocab_size_padded`, silently producing wrong logits
        // for every batch slot beyond the first when `vocab_size` is not
        // already 64-aligned (e.g. HF mamba-130m's 50280 vs padded 50304).
        let lm_head_padded: Option<Vec<f32>> = lm_head.as_ref().map(|lm| {
            if vocab_size == vocab_size_padded {
                lm.clone()
            } else {
                pad_lm_head_rows(lm, d_model, vocab_size, vocab_size_padded)
            }
        });

        // Upload embed + optional lm_head in requested dtype.
        let embed_storage = match dtype {
            WeightDtype::F32 => {
                let mut e = GpuBuffer::zeros(stream, vocab_size_padded * d_model)?;
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
                upload_f32_as_dtype(stream, &e, 0, &embed, embed.len(), dtype)?;

                let lm = if let Some(ref lm_w) = lm_head_padded {
                    let lm_bytes = lm_w.len() * dtype.size_bytes();
                    let b = GpuByteBuffer::zeros(stream, lm_bytes)?;
                    upload_f32_as_dtype(stream, &b, 0, lm_w, lm_w.len(), dtype)?;
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

        Ok(Self {
            backbone,
            embed_storage,
            embed_cpu: embed,
            input_cpu: vec![0.0; batch * d_model],
            gpu_logits,
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

    /// Capture the backbone's per-step CUDA Graph for accelerated decode.
    /// Run at least one warmup [`Self::generate`] (or any single backbone
    /// step) before capture so cuBLAS settles. The lm_head GEMM runs
    /// eagerly outside the graph; only the backbone step is captured.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if let EmbedStorage::Half {
            lm_head: None,
            dtype,
            ..
        } = &self.embed_storage
        {
            presize_tied_lm_head_scratch(
                self.backbone.ctx(),
                *dtype,
                TiedLmDims {
                    batch: self.batch,
                    d_model: self.d_model,
                    vocab_padded: self.vocab_size_padded,
                },
            )?;
        }
        self.backbone.capture_graph()
    }

    /// Download the backbone's temporal buffer (last-layer hidden state, last
    /// timestep) as f32 regardless of storage dtype. Intended for debugging /
    /// parity harnesses.
    #[doc(hidden)]
    pub fn debug_download_temporal(&self, out: &mut [f32]) -> Result<(), String> {
        self.backbone.download_temporal(out)
    }

    /// Debug-only: reset state, embed `token`, run the backbone but stop after
    /// `layer_limit` layers, and download the post-layer residual to `out`.
    /// Always returns f32 regardless of storage dtype. For step-by-step
    /// f32-vs-bf16 parity bisection.
    #[doc(hidden)]
    pub fn debug_step_one_token(
        &mut self,
        token: u32,
        layer_limit: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.backbone.reset()?;
        let emb =
            crate::hf::embed::embed_lookup(&self.embed_cpu, token, self.d_model, self.vocab_size);
        self.input_cpu[..self.d_model].copy_from_slice(emb);
        self.backbone
            .debug_step_partial(&self.input_cpu[..self.d_model], layer_limit, out)
    }

    /// Reset the backbone's recurrent SSM + conv states to zero. Call
    /// between independent prompts so state from the previous generation
    /// does not leak.
    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }

    /// Greedy / sampled generation. Returns the full sequence of newly
    /// generated tokens (excludes `prompt`). Equivalent to collecting
    /// the callback output of [`Self::generate_streaming`]. Requires
    /// `batch = 1`; for parallel batched decoding use
    /// [`Self::generate_batch`].
    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Result<Vec<u32>, String> {
        let mut tokens = Vec::with_capacity(params.max_tokens);
        self.generate_streaming(prompt, params, |tok, _| {
            tokens.push(tok);
        })?;
        Ok(tokens)
    }

    /// Streaming variant of [`Self::generate`]: invokes `cb(token, "")`
    /// for each newly generated token as it is produced, allowing the
    /// caller to print-as-you-go or stop early. Requires `batch = 1`.
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
            self.input_cpu[..self.d_model].copy_from_slice(emb);
            self.backbone
                .step_gpu_only(&self.input_cpu[..self.d_model])?;
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
                    self.input_cpu[i * d..(i + 1) * d].fill(0.0);
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
                    // Untied: logits[B,Vpad] = hidden[B,D] @ lm_head[D,Vpad].
                    // The hidden state already lives on the GPU (temporal_ptr)
                    // — feed it directly; the old path bounced it through host
                    // memory (D2H + H2D) on every decoded token.
                    gpu_gemm_bi_forward_ptr(
                        ctx,
                        &mut self.gpu_logits,
                        temporal_ptr,
                        lm.cached_ptr(),
                        None,
                        (b, d, self.vocab_size_padded),
                    )?;
                } else {
                    // Tied: logits[B,V] = temporal[B,D] @ embed^T[D,V]
                    // Single SGEMM via OP_T on embed (reuses row-major [V,D] buffer).
                    gpu_gemm_bi_tied_lm_head_raw(
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
                    // Untied: Y[B,Vpad] = temporal_half[B,D] @ lm_head[D,Vpad]
                    // `vocab_size_padded` matches lm_head and gpu_logits row stride.
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
                        (b, d, self.vocab_size_padded),
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
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dst: &GpuByteBuffer,
    elem_offset: usize,
    src: &[f32],
    src_elems: usize,
    dtype: WeightDtype,
) -> Result<(), String> {
    use crate::mamba_ssm::gpu::buffers::cu_memcpy_htod_raw;
    assert_eq!(src.len(), src_elems, "src size mismatch");
    let byte_off = elem_offset * dtype.size_bytes();
    let byte_count = src_elems * dtype.size_bytes();
    let dst_ptr = dst.cached_ptr() + byte_off as u64;

    match dtype {
        WeightDtype::F32 => {
            let bytes: &[u8] = bytemuck::cast_slice(src);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod_raw(stream, dst_ptr, bytes)
        }
        WeightDtype::Bf16 => {
            let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod_raw(stream, dst_ptr, bytes)
        }
        WeightDtype::F16 => {
            let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            assert_eq!(bytes.len(), byte_count);
            cu_memcpy_htod_raw(stream, dst_ptr, bytes)
        }
    }
}

/// Pad an untied lm_head from `[d_model, vocab]` row-major to
/// `[d_model, vocab_padded]` row-major (trailing zeros per ROW).
///
/// The logits GEMM reads the head with stride `vocab_padded`; a flat
/// `copy_from_slice` interleaves source rows into wrong destination
/// offsets and silently produces wrong logits for every vocab that is not
/// already 64-aligned (e.g. 50280 -> 50304 on mamba-130m-hf). The Mamba-1
/// LM fixed this first; the Mamba-3 LM shipped the flat copy until 0.6, and
/// both architectures now share this one implementation.
pub(crate) fn pad_lm_head_rows(
    lm: &[f32],
    d_model: usize,
    vocab_size: usize,
    vocab_size_padded: usize,
) -> Vec<f32> {
    let mut padded = vec![0.0f32; vocab_size_padded * d_model];
    for row in 0..d_model {
        let src = &lm[row * vocab_size..(row + 1) * vocab_size];
        let dst = &mut padded[row * vocab_size_padded..row * vocab_size_padded + vocab_size];
        dst.copy_from_slice(src);
    }
    padded
}

#[cfg(test)]
mod pad_tests {
    use super::pad_lm_head_rows;

    /// Pin: padding is PER-ROW, never a flat copy. With d_model=2,
    /// vocab=3, padded=4 the flat copy would smear row 1 across the row
    /// boundary; the row pad keeps each row's values at its own stride.
    #[test]
    fn pad_lm_head_rows_is_row_strided() {
        let lm = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let got = pad_lm_head_rows(&lm, 2, 3, 4);
        assert_eq!(got, vec![1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 30.0, 0.0]);
        let mut flat = vec![0.0f32; 8];
        flat[..6].copy_from_slice(&lm);
        assert_ne!(got, flat, "flat copy must differ - that was the M3 bug");
    }
}

#[cfg(test)]
mod tied_head_capture_tests {
    use super::*;
    use crate::config::{MambaConfig, ScanMode};
    use crate::mamba_ssm::gpu::context::GemmMode;
    use crate::weights::MambaWeights;

    fn owned_lm(dtype: WeightDtype, tied: bool) -> GpuMambaLM {
        const D_MODEL: usize = 16;
        const VOCAB_PADDED: usize = 512;
        let cfg = MambaConfig {
            d_model: D_MODEL,
            d_state: 4,
            d_conv: 2,
            expand: 1,
            n_layers: 1,
            scan_mode: ScanMode::Sequential,
            rms_norm_eps: 1e-5,
        };
        let mut weights = MambaWeights::init(&cfg, D_MODEL, 0x9031);
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
        let backbone =
            GpuMambaBackbone::new_with_dtype(0, &weights, cfg, D_MODEL, 1, dtype).unwrap();
        let stream = backbone.stream().clone();
        let embed_cpu = vec![0.125; VOCAB_PADDED * D_MODEL];
        let embed_storage = if dtype == WeightDtype::F32 {
            EmbedStorage::F32 {
                embed: GpuBuffer::from_cpu(&stream, &embed_cpu).unwrap(),
                lm_head: (!tied).then(|| GpuBuffer::from_cpu(&stream, &embed_cpu).unwrap()),
            }
        } else {
            let upload = || {
                let buffer =
                    GpuByteBuffer::zeros(&stream, embed_cpu.len() * dtype.size_bytes()).unwrap();
                upload_f32_as_dtype(&stream, &buffer, 0, &embed_cpu, embed_cpu.len(), dtype)
                    .unwrap();
                buffer
            };
            EmbedStorage::Half {
                embed: upload(),
                lm_head: (!tied).then(upload),
                dtype,
            }
        };
        GpuMambaLM {
            backbone,
            embed_storage,
            embed_cpu,
            input_cpu: vec![0.0; D_MODEL],
            gpu_logits: GpuBuffer::zeros(&stream, VOCAB_PADDED).unwrap(),
            logits_padded_cpu: vec![0.0; VOCAB_PADDED],
            logits_cpu: vec![0.0; VOCAB_PADDED],
            vocab_size: VOCAB_PADDED,
            vocab_size_padded: VOCAB_PADDED,
            d_model: D_MODEL,
            batch: 1,
        }
    }

    #[test]
    #[ignore = "needs a CUDA device and NVRTC"]
    fn tied_bf16_m1_capture_reserves_head_scratch_before_freeze() {
        let mut lm = owned_lm(WeightDtype::Bf16, true);
        lm.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();

        // Warm only the backbone. A prior head call would hide a missing
        // reservation in the LM capture wrapper.
        lm.backbone.step_gpu_only(&vec![0.25; lm.d_model]).unwrap();
        lm.capture_graph().unwrap();
        let reserved = lm.ctx().bi_upcast_scratch_ptrs();
        assert_ne!(reserved[0], 0);
        assert_ne!(reserved[1], 0);
        lm.compute_logits().unwrap();
        assert_eq!(lm.ctx().bi_upcast_scratch_ptrs(), reserved);
    }

    #[test]
    #[ignore = "needs a CUDA device and NVRTC"]
    fn m1_tied_and_untied_heads_all_storage_without_vendor_gemm() {
        use crate::mamba_ssm::gpu::blas::vendor_gemm_test::Guard;
        use crate::mamba_ssm::gpu::context::BiGemmFamily;
        let deny = Guard::new(true).unwrap();
        for family in [BiGemmFamily::Inference, BiGemmFamily::Triad] {
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                for tied in [false, true] {
                    eprintln!("M1 head {family:?} {dtype:?} tied={tied}");
                    let mut lm = owned_lm(dtype, tied);
                    lm.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
                    lm.ctx().set_bi_gemm_family(family);
                    lm.ctx().set_bi_tensor_cores(true);
                    let input = vec![0.25; lm.d_model];
                    lm.backbone.step_gpu_only(&input).unwrap();
                    lm.compute_logits().unwrap();
                    let expected: Vec<_> = lm
                        .gpu_logits
                        .to_cpu(lm.backbone.stream())
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    lm.capture_graph().unwrap();
                    lm.backbone.reset().unwrap();
                    lm.backbone.step_gpu_only(&input).unwrap();
                    lm.compute_logits().unwrap();
                    let output = lm.gpu_logits.to_cpu(lm.backbone.stream()).unwrap();
                    assert!(output.iter().all(|x| x.is_finite()));
                    assert_eq!(
                        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected
                    );
                }
            }
        }
        assert_eq!(deny.calls(), 0);
    }
}
