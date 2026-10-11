//! The Mamba-3 SISO language model of 0.7. [`MambaGpuLM`] serves every
//! family and takes its place; [`GpuMamba3LM`] forwards to one.

use crate::mamba_ssm::gpu::GpuOptions;
use crate::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::inference::Mamba3GpuBackbone;
use crate::mamba3_siso::weights::Mamba3Weights;

use super::gpu_lm::MambaGpuLM;
use super::sample::SampleParams;

/// The arguments of [`GpuMamba3LM::build`].
pub struct Mamba3LmBuild<'a> {
    /// The backbone weights.
    pub cpu_weights: &'a Mamba3Weights,
    /// The backbone configuration.
    pub cfg: Mamba3Config,
    /// The embedding, row-major `[vocab_rows, d_model]`.
    pub embed: Vec<f32>,
    /// An untied head `[d_model, vocab_size]`, or `None` to tie it to the
    /// embedding.
    pub lm_head: Option<Vec<f32>>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// The CUDA device.
    pub gpu_ordinal: usize,
    /// Storage precision.
    pub dtype: WeightDtype,
    /// Sequences generated side by side.
    pub batch: usize,
}

/// The Mamba-3 SISO language model of 0.7, a front for [`MambaGpuLM`].
pub struct GpuMamba3LM {
    lm: MambaGpuLM,
    /// Vocabulary size, the number of valid token ids.
    pub vocab_size: usize,
    /// Backbone width (`d_model`).
    pub d_model: usize,
    /// Sequences generated side by side.
    pub batch: usize,
}

impl GpuMamba3LM {
    #[deprecated(
        since = "0.8.0",
        note = "build a `Mamba3GpuBackbone` and wrap it with `MambaGpuLM::new`"
    )]
    pub fn from_weights(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
    ) -> Result<Self, String> {
        Self::assemble(
            Mamba3LmBuild {
                cpu_weights,
                cfg,
                embed,
                lm_head,
                vocab_size,
                gpu_ordinal,
                dtype: WeightDtype::F32,
                batch: 1,
            },
            None,
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "build a `Mamba3GpuBackbone` and wrap it with `MambaGpuLM::new`"
    )]
    pub fn from_weights_with_mode(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::assemble(
            Mamba3LmBuild {
                cpu_weights,
                cfg,
                embed,
                lm_head,
                vocab_size,
                gpu_ordinal,
                dtype: WeightDtype::F32,
                batch: 1,
            },
            Some(mode),
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "build a `Mamba3GpuBackbone` and wrap it with `MambaGpuLM::new`"
    )]
    pub fn from_weights_with_dtype(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
        gpu_ordinal: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::assemble(
            Mamba3LmBuild {
                cpu_weights,
                cfg,
                embed,
                lm_head,
                vocab_size,
                gpu_ordinal,
                dtype,
                batch: 1,
            },
            None,
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "build a `Mamba3GpuBackbone` and wrap it with `MambaGpuLM::new`"
    )]
    pub fn build(args: Mamba3LmBuild<'_>) -> Result<Self, String> {
        Self::assemble(args, None)
    }

    #[deprecated(
        since = "0.8.0",
        note = "build a `Mamba3GpuBackbone` and wrap it with `MambaGpuLM::new`"
    )]
    pub fn build_with_mode(args: Mamba3LmBuild<'_>, mode: GemmMode) -> Result<Self, String> {
        Self::assemble(args, Some(mode))
    }

    fn assemble(args: Mamba3LmBuild<'_>, mode: Option<GemmMode>) -> Result<Self, String> {
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
        let mut opts = GpuOptions::default().device(gpu_ordinal).dtype(dtype);
        if let Some(mode) = mode {
            opts = opts.mode(mode);
        }
        let backbone = Mamba3GpuBackbone::new(cpu_weights, cfg, cfg.d_model, batch, opts)?;
        let lm = MambaGpuLM::new(backbone, embed, lm_head, vocab_size)?;
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
