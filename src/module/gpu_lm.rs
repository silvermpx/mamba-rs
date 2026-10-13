//! The language model on the GPU, for every family: the embedding, a GPU
//! backbone, and a head tied to the embedding or held on its own.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use rayon::prelude::*;

use crate::config::ModelFamily;
use crate::gpu::GpuOptions;
use crate::gpu::blas::{
    TiedLmDims, TypedPtr, gpu_gemm_bi_forward_ptr, gpu_gemm_bi_tied_lm_head_raw,
    gpu_gemm_ex_forward_raw, gpu_gemm_ex_tied_lm_head_raw, gpu_gemm_ex_tied_lm_head_widened_raw,
    presize_tied_lm_head_rows,
};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::gpu::context::{GemmMode, GpuCtx};
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::launch_cast;
use crate::hf::embed::embed_lookup;
use crate::hf::load::{HfModel, load_hf};
use crate::hf::load_mamba2::load_hf_mamba2;
use crate::hf::load_mamba3::{load_hf_mamba3, load_hf_mamba3_mimo};
use crate::hf::read_family;
use crate::mamba1::gpu::inference::MambaGpuBackbone;
use crate::mamba2::gpu::Mamba2GpuBackbone;
use crate::mamba3_mimo::gpu::Mamba3MimoGpuBackbone;
use crate::mamba3_siso::gpu::inference::Mamba3GpuBackbone;

use super::gpu_backbone::{AnyGpuBackbone, GpuBackbone};
use super::lm::AnyBackbone;
use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};

/// Prompts of at least this many tokens run through the one-pass prefill,
/// shorter ones token by token. The two routes reduce in different orders,
/// so at bf16 the same prompt can end a few ulps apart. Below this length a
/// single sequence takes the step route batched generation always takes,
/// and generation stays batch-invariant; long prompts trade that for the
/// prefill's speed.
const PREFILL_PARALLEL_THRESHOLD: usize = 256;

/// Below this many sequences the per-slot sampling stays on the calling
/// thread: the rayon hand-off costs more than sampling a slot (about 12 µs
/// greedy, 30 to 60 µs with top-k and top-p).
const SAMPLE_PARALLEL_THRESHOLD: usize = 8;

/// The embedding and the untied head on the device, in the storage
/// precision (f32 bits for f32 and tf32).
struct HeadWeights {
    embed: DtypedBuf,
    lm_head: Option<DtypedBuf>,
    dtype: WeightDtype,
    /// A half embedding widened to f32 once, for a tied head in the
    /// deterministic mode, whose exact reduction reads f32: widening it on
    /// every token moved the whole embedding twice. Built at the first
    /// such logits; the embedding never changes after construction.
    embed_widened: Option<GpuBuffer>,
}

/// A language model on the GPU for any of the four families.
///
/// [`MambaGpuLM::from_hf`] loads a checkpoint directory and reads its family
/// from `config.json`; [`MambaGpuLM::new`] wraps a GPU backbone built in code
/// with its embedding and head. The storage precision, device and GEMM mode
/// are the backbone's.
///
/// ```rust,no_run
/// use mamba_rs::module::sample::SampleParams;
/// use mamba_rs::{GpuOptions, MambaGpuLM, WeightDtype};
/// use std::path::Path;
///
/// let opts = GpuOptions::default().dtype(WeightDtype::Bf16);
/// let mut lm = MambaGpuLM::from_hf(Path::new("./mamba2-130m"), 1, opts).unwrap();
/// lm.capture_graph().unwrap();
/// let tokens = lm.generate(&[1, 2, 3], &SampleParams::default()).unwrap();
/// ```
pub struct MambaGpuLM {
    backbone: AnyGpuBackbone,
    head: HeadWeights,
    /// The backbone's f32 rows cast to the head's half precision, for the
    /// engines that leave f32 rows (Mamba-2, MIMO) under a half head. The
    /// head reads a live allocation of its own, never the context staging.
    head_rows: Option<DtypedBuf>,
    /// The embedding on the host, read for every token fed in.
    embed_cpu: Vec<f32>,
    /// One step's input rows, `[batch, d_model]`.
    input_cpu: Vec<f32>,
    /// A prompt's rows for the one-pass prefill, grown to the longest
    /// prompt seen.
    prompt_cpu: Vec<f32>,
    gpu_logits: GpuBuffer,
    logits_padded_cpu: Vec<f32>,
    logits_cpu: Vec<f32>,
    /// Vocabulary size, the number of valid token ids.
    pub vocab_size: usize,
    vocab_size_padded: usize,
    /// Backbone width (`d_model`).
    pub d_model: usize,
    /// Sequences generated side by side.
    pub batch: usize,
}

impl MambaGpuLM {
    /// Load a checkpoint directory of any family for `batch` sequences, on
    /// the device, in the storage precision and in the GEMM mode `opts`
    /// names. The family comes from the checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// Checkpoint, configuration, `MAMBA_RS_GEMM_MODE`, CUDA, upload and
    /// allocation failures.
    pub fn from_hf(dir: &Path, batch: usize, opts: GpuOptions) -> Result<Self, String> {
        match read_family(dir)? {
            ModelFamily::Mamba1 => {
                let HfModel {
                    backbone,
                    embed,
                    lm_head,
                    vocab_size,
                    d_model,
                    ..
                } = load_hf(dir)?;
                let cfg = *backbone.config();
                let gpu = MambaGpuBackbone::new(backbone.weights(), cfg, d_model, batch, opts)?;
                Self::new(gpu, embed, lm_head, vocab_size)
            }
            ModelFamily::Mamba2 => {
                let m = load_hf_mamba2(dir)?;
                let d_model = m.cfg.d_model;
                let gpu = Mamba2GpuBackbone::new(&m.weights, m.cfg, d_model, batch, opts)?;
                Self::new(gpu, m.embed, m.lm_head, m.vocab_size)
            }
            ModelFamily::Mamba3 => {
                let m = load_hf_mamba3(dir)?;
                let d_model = m.cfg.d_model;
                let gpu = Mamba3GpuBackbone::new(&m.weights, m.cfg, d_model, batch, opts)?;
                Self::new(gpu, m.embed, m.lm_head, m.vocab_size)
            }
            ModelFamily::Mamba3Mimo => {
                let m = load_hf_mamba3_mimo(dir)?;
                let d_model = m.cfg.d_model;
                let gpu = Mamba3MimoGpuBackbone::new(&m.weights, m.cfg, d_model, batch, opts)?;
                Self::new(gpu, m.embed, m.lm_head, m.vocab_size)
            }
        }
    }

    /// Load a language-model checkpoint `MambaLM::save_checkpoint` wrote,
    /// for `batch` sequences, on the device, in the storage precision and
    /// in the GEMM mode `opts` names. The family comes from the file.
    ///
    /// # Errors
    ///
    /// Checkpoint, configuration, CUDA, upload and allocation failures.
    pub fn from_checkpoint(path: &Path, batch: usize, opts: GpuOptions) -> Result<Self, String> {
        let p = super::lm_checkpoint::load(path)?;
        let gpu: AnyGpuBackbone = match &p.backbone {
            AnyBackbone::M1(b) => {
                MambaGpuBackbone::new(b.weights(), *b.config(), b.input_dim(), batch, opts)?.into()
            }
            AnyBackbone::M2(b) => {
                Mamba2GpuBackbone::new(b.weights(), *b.config(), b.input_dim(), batch, opts)?.into()
            }
            AnyBackbone::M3(b) => {
                Mamba3GpuBackbone::new(b.weights(), *b.config(), b.input_dim(), batch, opts)?.into()
            }
            AnyBackbone::M3Mimo(b) => {
                Mamba3MimoGpuBackbone::new(b.weights(), *b.config(), b.input_dim(), batch, opts)?
                    .into()
            }
        };
        Self::new(gpu, p.embed, p.lm_head, p.vocab_size)
    }

    /// Wrap a GPU backbone with its embedding and head.
    ///
    /// `embed` is row-major `[vocab_rows, d_model]`, possibly padded past
    /// `vocab_size`; the rows past it never win sampling. `lm_head` is an
    /// untied head `[d_model, vocab_size]`; `None` ties the head to the
    /// embedding. The backbone reads embeddings directly, so its input
    /// width must be `d_model`. The embedding and head are stored in the
    /// backbone's precision.
    pub fn new(
        backbone: impl Into<AnyGpuBackbone>,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
    ) -> Result<Self, String> {
        let backbone = backbone.into();
        let d_model = backbone.d_model();
        if backbone.input_dim() != d_model {
            return Err(format!(
                "the embedding feeds the backbone directly, so its input width {} must be \
                 d_model {d_model}",
                backbone.input_dim()
            ));
        }
        if d_model == 0 || !embed.len().is_multiple_of(d_model) {
            return Err(format!(
                "embed length {} is not a multiple of d_model {d_model}",
                embed.len()
            ));
        }
        let vocab_size_padded = embed.len() / d_model;
        if vocab_size == 0 || vocab_size > vocab_size_padded {
            return Err(format!(
                "vocab_size {vocab_size} out of range (embed holds {vocab_size_padded} rows)"
            ));
        }
        if let Some(lm) = &lm_head
            && lm.len() != d_model * vocab_size
        {
            return Err(format!(
                "lm_head length {} != d_model*vocab_size {}",
                lm.len(),
                d_model * vocab_size
            ));
        }
        let batch = backbone.batch();
        let stream = backbone.stream().clone();
        let dtype = backbone.dtype();
        let upload = |rows: &[f32]| -> Result<DtypedBuf, String> {
            let buf = DtypedBuf::zeros(&stream, rows.len(), dtype)?;
            buf.upload_f32(&stream, rows)?;
            Ok(buf)
        };
        // The untied head gets the padded width too, so its GEMM writes the
        // logits at the row stride the tied head uses.
        let lm_head = match lm_head {
            Some(lm) if vocab_size != vocab_size_padded => Some(upload(&pad_lm_head_rows(
                &lm,
                d_model,
                vocab_size,
                vocab_size_padded,
            ))?),
            Some(lm) => Some(upload(&lm)?),
            None => None,
        };
        let head = HeadWeights {
            embed: upload(&embed)?,
            lm_head,
            dtype,
            embed_widened: None,
        };
        let head_rows = if matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16)
            && backbone.temporal_dtype() != dtype
        {
            Some(DtypedBuf::zeros(&stream, batch * d_model, dtype)?)
        } else {
            None
        };
        let gpu_logits = GpuBuffer::zeros(&stream, batch * vocab_size_padded)?;
        Ok(Self {
            backbone,
            head,
            head_rows,
            embed_cpu: embed,
            input_cpu: vec![0.0; batch * d_model],
            prompt_cpu: Vec::new(),
            gpu_logits,
            logits_padded_cpu: vec![0.0; batch * vocab_size_padded],
            logits_cpu: vec![0.0; batch * vocab_size],
            vocab_size,
            vocab_size_padded,
            d_model,
            batch,
        })
    }

    /// The family of the backbone.
    pub fn family(&self) -> ModelFamily {
        self.backbone.family()
    }

    /// The GPU backbone.
    pub fn backbone(&self) -> &AnyGpuBackbone {
        &self.backbone
    }

    /// The CUDA context: the GEMM mode and route live here, and a captured
    /// graph binds them.
    pub fn ctx(&self) -> &GpuCtx {
        self.backbone.ctx()
    }

    /// Storage precision of the weights.
    pub fn dtype(&self) -> WeightDtype {
        self.head.dtype
    }

    /// The logits of the last step for sequence `b`, `vocab_size` long.
    /// Valid after a `generate` call.
    pub fn last_logits(&self, b: usize) -> &[f32] {
        &self.logits_cpu[b * self.vocab_size..(b + 1) * self.vocab_size]
    }

    /// Capture the backbone's decode step as one CUDA graph; the head runs
    /// after it, outside the graph. Reserves the head's scratch first,
    /// because nothing can grow once the graph is captured.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        let dtype = self.head.dtype;
        if matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16) {
            let ctx = self.backbone.ctx();
            if self.head.lm_head.is_none() {
                // The deterministic head keeps the embedding widened once
                // and converts only the rows per token.
                presize_tied_lm_head_rows(
                    ctx,
                    dtype,
                    TiedLmDims {
                        batch: self.batch,
                        d_model: self.d_model,
                        vocab_padded: self.vocab_size_padded,
                    },
                )?;
            }
        }
        self.backbone.capture_graph()
    }

    /// Drop the captured graph: steps run eagerly until the next capture.
    pub fn clear_graph(&mut self) -> Result<(), String> {
        self.backbone.clear_graph()
    }

    /// Zero the recurrent state. Each `generate` call starts from it.
    pub fn reset(&mut self) -> Result<(), String> {
        self.backbone.reset()
    }

    /// Greedy or sampled generation for a batch of one: the new tokens,
    /// without the prompt. [`Self::generate_batch`] serves larger batches.
    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Result<Vec<u32>, String> {
        let mut tokens = Vec::with_capacity(params.max_tokens);
        self.generate_streaming(prompt, params, |tok, _| tokens.push(tok))?;
        Ok(tokens)
    }

    /// [`Self::generate`] that hands every new token to `cb` as it is
    /// produced (the text argument is empty).
    pub fn generate_streaming(
        &mut self,
        prompt: &[u32],
        params: &SampleParams,
        mut cb: impl FnMut(u32, &str),
    ) -> Result<(), String> {
        if self.batch != 1 {
            return Err(format!(
                "generate takes a batch of one, this model has {}; use generate_batch",
                self.batch
            ));
        }
        // An empty prompt has no position to continue from, so it generates
        // nothing, the same rule as the batch call and the CPU model.
        if prompt.is_empty() {
            return Ok(());
        }
        self.backbone.reset()?;
        let mut rng = Xoshiro256PlusPlus::new(params.seed);
        if prompt.len() >= PREFILL_PARALLEL_THRESHOLD {
            self.prefill_prompt(prompt)?;
        } else {
            for &token in prompt {
                self.step_token(token)?;
            }
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
            self.step_token(next)?;
            self.compute_logits()?;
        }
        Ok(())
    }

    /// Generate for every sequence of the batch side by side.
    ///
    /// `prompts` and `params` hold one entry per sequence. The prompts are
    /// fed token by token, so they may differ in length; a sequence starts
    /// sampling when its prompt ends, with its own seed and end tokens.
    /// Each sequence stops at an end token or at its `max_tokens`, and the
    /// call returns when all have stopped.
    pub fn generate_batch(
        &mut self,
        prompts: &[&[u32]],
        params: &[SampleParams],
    ) -> Result<Vec<Vec<u32>>, String> {
        let b = self.batch;
        if prompts.len() != b || params.len() != b {
            return Err(format!(
                "generate_batch takes one prompt and one parameter set per sequence: \
                 batch {b}, {} prompts, {} parameter sets",
                prompts.len(),
                params.len()
            ));
        }
        self.backbone.reset()?;
        let mut rngs: Vec<Xoshiro256PlusPlus> = params
            .iter()
            .map(|p| Xoshiro256PlusPlus::new(p.seed))
            .collect();

        let d = self.d_model;
        let vocab_size = self.vocab_size;
        let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let max_tokens = params.iter().map(|p| p.max_tokens).max().unwrap_or(0);

        // How many prompt tokens each sequence has fed, whether it stopped,
        // what it produced, and the token it feeds next.
        let mut prompt_pos = vec![0usize; b];
        let mut finished = vec![false; b];
        let mut outputs: Vec<Vec<u32>> = vec![Vec::new(); b];
        // The tokens each sequence has seen, prompt and output, which its
        // repetition penalty reads, as the single-sequence call reads them.
        let mut seen: Vec<Vec<u32>> = prompts
            .iter()
            .zip(params)
            .map(|(p, sp)| {
                let mut v = Vec::with_capacity(p.len() + sp.max_tokens);
                v.extend_from_slice(p);
                v
            })
            .collect();
        let mut last_token = vec![0u32; b];
        for i in 0..b {
            match prompts[i].first() {
                Some(&first) => {
                    last_token[i] = first;
                    prompt_pos[i] = 1;
                }
                None => finished[i] = true,
            }
        }

        for _ in 0..max_prompt + max_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }
            for i in 0..b {
                let row = &mut self.input_cpu[i * d..(i + 1) * d];
                if finished[i] {
                    // A stopped sequence still steps; its rows are ignored.
                    row.fill(0.0);
                } else {
                    row.copy_from_slice(embed_lookup(
                        &self.embed_cpu,
                        last_token[i],
                        d,
                        vocab_size,
                    ));
                }
            }
            self.backbone.step_gpu_only(&self.input_cpu)?;
            self.compute_logits()?;

            // Sequences past their prompt sample from their own logits with
            // their own generator, side by side when the batch is large.
            let sampling: Vec<bool> = (0..b)
                .map(|i| !finished[i] && prompt_pos[i] >= prompts[i].len())
                .collect();
            let new_tokens: Vec<Option<u32>> = if b >= SAMPLE_PARALLEL_THRESHOLD {
                self.logits_cpu
                    .par_chunks_mut(vocab_size)
                    .zip(rngs.par_iter_mut())
                    .enumerate()
                    .map(|(i, (logits, rng))| {
                        sampling[i].then(|| sample_token(logits, &params[i], &seen[i], rng))
                    })
                    .collect()
            } else {
                self.logits_cpu
                    .chunks_mut(vocab_size)
                    .zip(rngs.iter_mut())
                    .enumerate()
                    .map(|(i, (logits, rng))| {
                        sampling[i].then(|| sample_token(logits, &params[i], &seen[i], rng))
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
                let Some(next) = new_tokens[i] else {
                    return Err(format!("sequence {i} sampled no token"));
                };
                if params[i].eos_token_ids.contains(&next)
                    || outputs[i].len() >= params[i].max_tokens
                {
                    finished[i] = true;
                    continue;
                }
                outputs[i].push(next);
                seen[i].push(next);
                last_token[i] = next;
            }
        }
        Ok(outputs)
    }

    /// Feed one token of the single sequence.
    fn step_token(&mut self, token: u32) -> Result<(), String> {
        let d = self.d_model;
        let emb = embed_lookup(&self.embed_cpu, token, d, self.vocab_size);
        self.input_cpu[..d].copy_from_slice(emb);
        self.backbone.step_gpu_only(&self.input_cpu[..d])
    }

    /// Feed a whole prompt of the single sequence in one pass.
    fn prefill_prompt(&mut self, prompt: &[u32]) -> Result<(), String> {
        let d = self.d_model;
        self.prompt_cpu.clear();
        self.prompt_cpu.reserve(prompt.len() * d);
        for &token in prompt {
            self.prompt_cpu.extend_from_slice(embed_lookup(
                &self.embed_cpu,
                token,
                d,
                self.vocab_size,
            ));
        }
        self.backbone.prefill(&self.prompt_cpu, prompt.len())
    }

    /// The head over the backbone's last rows, then the logits on the host.
    fn compute_logits(&mut self) -> Result<(), String> {
        let ctx = self.backbone.ctx();
        let stream: Arc<CudaStream> = self.backbone.stream().clone();
        let temporal_ptr = self.backbone.temporal_ptr();
        let (b, d, vp) = (self.batch, self.d_model, self.vocab_size_padded);
        if self.head.dtype.is_half()
            && self.head.lm_head.is_none()
            && ctx.gemm_mode() == GemmMode::Deterministic
            && self.head.embed_widened.is_none()
        {
            let widen = if self.head.dtype == WeightDtype::Bf16 {
                &ctx.kernels.cast_bf16_to_f32
            } else {
                &ctx.kernels.cast_f16_to_f32
            };
            let widened = GpuBuffer::zeros(&stream, vp * d)?;
            launch_cast(
                &stream,
                widen,
                (widened.cached_ptr(), 4),
                (self.head.embed.cached_ptr(), self.head.dtype.size_bytes()),
                vp * d,
                format_args!("widen the tied embedding"),
            )?;
            self.head.embed_widened = Some(widened);
        }
        let HeadWeights {
            embed,
            lm_head,
            dtype,
            embed_widened,
        } = &self.head;

        match *dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => match lm_head {
                // logits[B, Vpad] = rows[B, D] @ lm_head[D, Vpad], straight
                // from the rows on the device.
                Some(lm) => gpu_gemm_bi_forward_ptr(
                    ctx,
                    &mut self.gpu_logits,
                    temporal_ptr,
                    lm.cached_ptr(),
                    None,
                    (b, d, vp),
                )?,
                // logits[B, V] = rows[B, D] @ embed^T[D, V], the embedding
                // read transposed in place.
                None => gpu_gemm_bi_tied_lm_head_raw(
                    ctx,
                    self.gpu_logits.cached_ptr(),
                    temporal_ptr,
                    embed.cached_ptr(),
                    b,
                    d,
                    vp,
                )?,
            },
            WeightDtype::Bf16 | WeightDtype::F16 => {
                let dt = *dtype;
                // The Mamba-1 and SISO engines leave half rows; Mamba-2 and
                // MIMO leave f32 rows, cast once into the head's own rows.
                let rows_ptr = match &self.head_rows {
                    None => temporal_ptr,
                    Some(rows) => {
                        let rows_ptr = rows.cached_ptr();
                        let cast = if dt == WeightDtype::Bf16 {
                            &ctx.kernels.cast_f32_to_bf16
                        } else {
                            &ctx.kernels.cast_f32_to_f16
                        };
                        launch_cast(
                            &stream,
                            cast,
                            (rows_ptr, dt.size_bytes()),
                            (temporal_ptr, 4),
                            b * d,
                            format_args!("cast rows for the head"),
                        )?;
                        rows_ptr
                    }
                };
                match lm_head {
                    Some(lm) => gpu_gemm_ex_forward_raw(
                        ctx,
                        &mut self.gpu_logits,
                        TypedPtr {
                            ptr: rows_ptr,
                            dtype: dt,
                        },
                        TypedPtr {
                            ptr: lm.cached_ptr(),
                            dtype: dt,
                        },
                        None,
                        (b, d, vp),
                    )?,
                    None => {
                        let dims = TiedLmDims {
                            batch: b,
                            d_model: d,
                            vocab_padded: vp,
                        };
                        match embed_widened {
                            Some(widened) => gpu_gemm_ex_tied_lm_head_widened_raw(
                                ctx,
                                self.gpu_logits.cached_ptr(),
                                (rows_ptr, embed.cached_ptr()),
                                widened,
                                dt,
                                dims,
                            )?,
                            None => gpu_gemm_ex_tied_lm_head_raw(
                                ctx,
                                self.gpu_logits.cached_ptr(),
                                rows_ptr,
                                embed.cached_ptr(),
                                dt,
                                dims,
                            )?,
                        }
                    }
                }
            }
        }

        stream
            .synchronize()
            .map_err(|e| format!("logits sync: {e:?}"))?;
        self.gpu_logits
            .download(&stream, &mut self.logits_padded_cpu)?;
        // The logits rows are [B, vocab_padded]; drop the padding columns.
        for (dst, src) in self
            .logits_cpu
            .chunks_mut(self.vocab_size)
            .zip(self.logits_padded_cpu.chunks(vp))
        {
            dst.copy_from_slice(&src[..self.vocab_size]);
        }
        Ok(())
    }
}

/// The Mamba-1 language model of 0.7. [`MambaGpuLM`] serves every family
/// and takes its place; this type forwards to one.
pub struct GpuMambaLM {
    lm: MambaGpuLM,
    /// Vocabulary size, the number of valid token ids.
    pub vocab_size: usize,
    /// Backbone width (`d_model`).
    pub d_model: usize,
    /// Sequences generated side by side.
    pub batch: usize,
}

impl GpuMambaLM {
    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf(dir, 1, GpuOptions::default().device(gpu_ordinal))`"
    )]
    pub fn from_hf(dir: &Path, gpu_ordinal: usize) -> Result<Self, String> {
        Self::load(dir, 1, GpuOptions::default().device(gpu_ordinal))
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf` with `GpuOptions::default().device(gpu_ordinal).mode(mode)`"
    )]
    pub fn from_hf_with_mode(
        dir: &Path,
        gpu_ordinal: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::load(dir, 1, GpuOptions::default().device(gpu_ordinal).mode(mode))
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf` with `GpuOptions::default().device(gpu_ordinal).dtype(dtype)`"
    )]
    pub fn from_hf_with_dtype(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::load(
            dir,
            1,
            GpuOptions::default().device(gpu_ordinal).dtype(dtype),
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf` with `GpuOptions::default().device(gpu_ordinal).dtype(dtype).mode(mode)`"
    )]
    pub fn from_hf_with_dtype_and_mode(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        mode: GemmMode,
    ) -> Result<Self, String> {
        let opts = GpuOptions::default()
            .device(gpu_ordinal)
            .dtype(dtype)
            .mode(mode);
        Self::load(dir, 1, opts)
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf(dir, batch, GpuOptions::default().device(gpu_ordinal).dtype(dtype))`"
    )]
    pub fn from_hf_with_dtype_batch(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
    ) -> Result<Self, String> {
        Self::load(
            dir,
            batch,
            GpuOptions::default().device(gpu_ordinal).dtype(dtype),
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaGpuLM::from_hf(dir, batch, GpuOptions::default().device(gpu_ordinal).dtype(dtype).mode(mode))`"
    )]
    pub fn from_hf_with_dtype_batch_and_mode(
        dir: &Path,
        gpu_ordinal: usize,
        dtype: WeightDtype,
        batch: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        let opts = GpuOptions::default()
            .device(gpu_ordinal)
            .dtype(dtype)
            .mode(mode);
        Self::load(dir, batch, opts)
    }

    fn load(dir: &Path, batch: usize, opts: GpuOptions) -> Result<Self, String> {
        let lm = MambaGpuLM::from_hf(dir, batch, opts)?;
        Ok(Self {
            vocab_size: lm.vocab_size,
            d_model: lm.d_model,
            batch: lm.batch,
            lm,
        })
    }

    /// See [`MambaGpuLM::ctx`].
    pub fn ctx(&self) -> &GpuCtx {
        self.lm.ctx()
    }

    /// See [`MambaGpuLM::last_logits`].
    pub fn last_logits(&self, b: usize) -> &[f32] {
        self.lm.last_logits(b)
    }

    /// See [`MambaGpuLM::dtype`].
    pub fn dtype(&self) -> WeightDtype {
        self.lm.dtype()
    }

    /// See [`MambaGpuLM::capture_graph`].
    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.lm.capture_graph()
    }

    /// See [`MambaGpuLM::clear_graph`].
    pub fn clear_graph(&mut self) -> Result<(), String> {
        self.lm.clear_graph()
    }

    /// See [`MambaGpuLM::reset`].
    pub fn reset(&mut self) -> Result<(), String> {
        self.lm.reset()
    }

    /// See [`MambaGpuLM::generate`].
    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Result<Vec<u32>, String> {
        self.lm.generate(prompt, params)
    }

    /// See [`MambaGpuLM::generate_streaming`].
    pub fn generate_streaming(
        &mut self,
        prompt: &[u32],
        params: &SampleParams,
        cb: impl FnMut(u32, &str),
    ) -> Result<(), String> {
        self.lm.generate_streaming(prompt, params, cb)
    }

    /// See [`MambaGpuLM::generate_batch`].
    pub fn generate_batch(
        &mut self,
        prompts: &[&[u32]],
        params: &[SampleParams],
    ) -> Result<Vec<Vec<u32>>, String> {
        self.lm.generate_batch(prompts, params)
    }
}

/// Pad an untied head from `[d_model, vocab]` row-major to
/// `[d_model, vocab_padded]` row-major, with zeros at the end of every row.
///
/// The logits GEMM reads the head at the padded row stride. A flat copy
/// would shift every row after the first into the wrong columns and give
/// wrong logits for any vocabulary that is not already a multiple of the
/// padding (50280 against 50304 on mamba-130m-hf).
pub(crate) fn pad_lm_head_rows(
    lm: &[f32],
    d_model: usize,
    vocab_size: usize,
    vocab_size_padded: usize,
) -> Vec<f32> {
    let mut padded = vec![0.0f32; vocab_size_padded * d_model];
    for (dst, src) in padded
        .chunks_mut(vocab_size_padded)
        .zip(lm.chunks(vocab_size))
    {
        dst[..vocab_size].copy_from_slice(src);
    }
    padded
}

#[cfg(test)]
mod pad_tests {
    use super::pad_lm_head_rows;

    /// Padding is per row, never a flat copy. With d_model=2, vocab=3,
    /// padded=4 a flat copy would smear row 1 across the row boundary; the
    /// row pad keeps each row's values at its own stride.
    #[test]
    fn pad_lm_head_rows_is_row_strided() {
        let lm = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let got = pad_lm_head_rows(&lm, 2, 3, 4);
        assert_eq!(got, vec![1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 30.0, 0.0]);
        let mut flat = vec![0.0f32; 8];
        flat[..6].copy_from_slice(&lm);
        assert_ne!(
            got, flat,
            "a flat copy differs, the bug the Mamba-3 LM once shipped"
        );
    }
}

#[cfg(test)]
mod tied_head_capture_tests {
    use super::*;
    use crate::mamba1::weights::MambaWeights;
    use crate::{config::ScanMode, mamba1::config::MambaConfig};

    fn owned_lm(dtype: WeightDtype, tied: bool) -> MambaGpuLM {
        const D_MODEL: usize = 16;
        const VOCAB: usize = 512;
        let cfg = MambaConfig {
            d_model: D_MODEL,
            d_state: 4,
            d_conv: 2,
            expand: 1,
            n_layers: 1,
            scan_mode: ScanMode::Sequential,
            rms_norm_eps: 1e-5,
            bcdt_rms_eps: None,
        };
        let mut weights = MambaWeights::init(&cfg, D_MODEL, 0x9031);
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
        let backbone = MambaGpuBackbone::new(
            &weights,
            cfg,
            D_MODEL,
            1,
            GpuOptions::default().dtype(dtype),
        )
        .unwrap();
        let embed = vec![0.125; VOCAB * D_MODEL];
        let lm_head = (!tied).then(|| embed.clone());
        MambaGpuLM::new(backbone, embed, lm_head, VOCAB).unwrap()
    }

    /// The batch call penalises repetitions of the prompt as the single
    /// call does: one sequence through `generate_batch` gives the tokens of
    /// `generate`, with a repetition penalty that reaches the prompt.
    #[test]
    #[ignore = "needs a CUDA device and NVRTC"]
    fn batch_generation_sees_the_prompt_like_the_single_call() {
        const D_MODEL: usize = 16;
        const VOCAB: usize = 64;
        let cfg = MambaConfig {
            d_model: D_MODEL,
            d_state: 4,
            d_conv: 2,
            expand: 1,
            n_layers: 1,
            scan_mode: ScanMode::Sequential,
            rms_norm_eps: 1e-5,
            bcdt_rms_eps: None,
        };
        let mut weights = MambaWeights::init(&cfg, D_MODEL, 0x5EED);
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
        let lm = || {
            let backbone =
                MambaGpuBackbone::new(&weights, cfg, D_MODEL, 1, GpuOptions::default()).unwrap();
            let mut x = 0x1234_5678u32;
            let embed: Vec<f32> = (0..VOCAB * D_MODEL)
                .map(|_| {
                    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
                })
                .collect();
            MambaGpuLM::new(backbone, embed, None, VOCAB).unwrap()
        };
        let params = SampleParams {
            repetition_penalty: 1.8,
            max_tokens: 16,
            eos_token_ids: Vec::new(),
            ..SampleParams::default()
        };
        let prompt = [3u32, 9, 9, 17, 3];
        let single = lm().generate(&prompt, &params).unwrap();
        let batch = lm()
            .generate_batch(&[&prompt], std::slice::from_ref(&params))
            .unwrap();
        assert_eq!(batch, vec![single]);
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
        use crate::gpu::blas::vendor_gemm_test::Guard;
        use crate::gpu::context::BiGemmFamily;
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
