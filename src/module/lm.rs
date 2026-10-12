//! Language model wrapper for Mamba text generation.

use std::path::Path;

use crate::mamba1::cpu::inference::MambaStepScratch;
use crate::mamba1::state::MambaState;
use crate::mamba2::state::Mamba2State;
use crate::mamba3_mimo::state::Mamba3MimoState;
use crate::mamba3_siso::cpu::inference::Mamba3StepScratch;
use crate::mamba3_siso::state::Mamba3State;

use super::backbone::MambaBackbone;
use super::backbone2::{Mamba2Backbone, Mamba2Scratch};
use super::backbone3::Mamba3Backbone;
use super::backbone3_mimo::{Mamba3MimoBackbone, Mamba3MimoScratch};
use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};

use crate::config::ModelFamily;
use crate::hf::embed::{embed_lookup, lm_head_logits, lm_head_logits_untied};
use crate::hf::load::{HfModel, load_hf};
use crate::hf::load_mamba2::load_hf_mamba2;
use crate::hf::load_mamba3::{load_hf_mamba3, load_hf_mamba3_mimo};
use crate::hf::read_family;

// ---------------------------------------------------------------------------
// AnyBackbone / AnyState / AnyScratch — enum dispatch
// ---------------------------------------------------------------------------

pub enum AnyBackbone {
    M1(MambaBackbone),
    M2(Mamba2Backbone),
    M3(Mamba3Backbone),
    M3Mimo(Mamba3MimoBackbone),
}

#[derive(Clone)]
pub enum AnyState {
    M1(MambaState),
    M2(Mamba2State),
    M3(Mamba3State),
    M3Mimo(Mamba3MimoState),
}

impl AnyState {
    pub fn reset(&mut self) {
        match self {
            Self::M1(s) => s.reset(),
            Self::M2(s) => s.reset(),
            Self::M3(s) => s.reset(),
            Self::M3Mimo(s) => s.reset(),
        }
    }
}

pub enum AnyScratch {
    M1(MambaStepScratch),
    M2(Box<Mamba2Scratch>),
    M3(Mamba3StepScratch),
    M3Mimo(Box<Mamba3MimoScratch>),
}

impl AnyBackbone {
    /// The family of the backbone held.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::M1(_) => ModelFamily::Mamba1,
            Self::M2(_) => ModelFamily::Mamba2,
            Self::M3(_) => ModelFamily::Mamba3,
            Self::M3Mimo(_) => ModelFamily::Mamba3Mimo,
        }
    }

    /// Pack the backbone's projections for decode (see
    /// [`MambaBackbone::pack_for_decode`]): the same outputs without a BLAS,
    /// faster on large layers, at the memory of a second copy of them.
    pub fn pack_for_decode(&mut self) {
        match self {
            Self::M1(b) => b.pack_for_decode(),
            Self::M2(b) => b.pack_for_decode(),
            Self::M3(b) => b.pack_for_decode(),
            Self::M3Mimo(b) => b.pack_for_decode(),
        }
    }

    /// One token through the backbone held; see each backbone's
    /// `forward_step`.
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut AnyState,
        scratch: &mut AnyScratch,
    ) {
        match (self, state, scratch) {
            (Self::M1(bb), AnyState::M1(st), AnyScratch::M1(sc)) => {
                bb.forward_step(input, output, st, sc)
            }
            (Self::M2(bb), AnyState::M2(st), AnyScratch::M2(sc)) => {
                bb.forward_step(input, output, st, sc)
            }
            (Self::M3(bb), AnyState::M3(st), AnyScratch::M3(sc)) => {
                bb.forward_step(input, output, st, sc)
            }
            (Self::M3Mimo(bb), AnyState::M3Mimo(st), AnyScratch::M3Mimo(sc)) => {
                bb.forward_step(input, output, st, sc)
            }
            _ => panic!("backbone/state/scratch type mismatch"),
        }
    }

    pub fn d_model(&self) -> usize {
        match self {
            Self::M1(bb) => bb.config().d_model,
            Self::M2(bb) => bb.d_model(),
            Self::M3(bb) => bb.d_model(),
            Self::M3Mimo(bb) => bb.d_model(),
        }
    }

    pub fn alloc_state(&self) -> AnyState {
        match self {
            Self::M1(bb) => AnyState::M1(bb.alloc_state()),
            Self::M2(bb) => AnyState::M2(bb.alloc_state()),
            Self::M3(bb) => AnyState::M3(bb.alloc_state()),
            Self::M3Mimo(bb) => AnyState::M3Mimo(bb.alloc_state()),
        }
    }

    pub fn alloc_scratch(&self) -> AnyScratch {
        match self {
            Self::M1(bb) => AnyScratch::M1(bb.alloc_scratch()),
            Self::M2(bb) => AnyScratch::M2(Box::new(bb.alloc_scratch())),
            Self::M3(bb) => AnyScratch::M3(bb.alloc_scratch()),
            Self::M3Mimo(bb) => AnyScratch::M3Mimo(Box::new(bb.alloc_scratch())),
        }
    }

    /// `seq_len` tokens through the backbone held; see each backbone's
    /// `forward_sequence`.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut AnyState,
        scratch: &mut AnyScratch,
        seq_len: usize,
    ) {
        match (self, state, scratch) {
            (Self::M1(bb), AnyState::M1(st), AnyScratch::M1(sc)) => {
                bb.forward_sequence(inputs, outputs, st, sc, seq_len)
            }
            (Self::M2(bb), AnyState::M2(st), AnyScratch::M2(sc)) => {
                bb.forward_sequence(inputs, outputs, st, sc, seq_len)
            }
            (Self::M3(bb), AnyState::M3(st), AnyScratch::M3(sc)) => {
                bb.forward_sequence(inputs, outputs, st, sc, seq_len)
            }
            (Self::M3Mimo(bb), AnyState::M3Mimo(st), AnyScratch::M3Mimo(sc)) => {
                bb.forward_sequence(inputs, outputs, st, sc, seq_len)
            }
            _ => panic!("backbone/state/scratch type mismatch"),
        }
    }
}

impl From<MambaBackbone> for AnyBackbone {
    fn from(b: MambaBackbone) -> Self {
        Self::M1(b)
    }
}

impl From<Mamba2Backbone> for AnyBackbone {
    fn from(b: Mamba2Backbone) -> Self {
        Self::M2(b)
    }
}

impl From<Mamba3Backbone> for AnyBackbone {
    fn from(b: Mamba3Backbone) -> Self {
        Self::M3(b)
    }
}

impl From<Mamba3MimoBackbone> for AnyBackbone {
    fn from(b: Mamba3MimoBackbone) -> Self {
        Self::M3Mimo(b)
    }
}

// ---------------------------------------------------------------------------
// MambaLM
// ---------------------------------------------------------------------------

pub struct MambaLM {
    pub backbone: AnyBackbone,
    state: AnyState,
    scratch: AnyScratch,
    temporal: Vec<f32>,
    prefill_inputs: Vec<f32>,
    prefill_outputs: Vec<f32>,
    embed: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    logits: Vec<f32>,
    logits_scratch: Vec<f32>,
    pub vocab_size: usize,
    vocab_size_padded: usize,
    pub d_model: usize,
}

impl MambaLM {
    /// Load a checkpoint directory of any family. The family comes from
    /// the checkpoint's `config.json`.
    pub fn from_hf(dir: &Path) -> Result<Self, String> {
        match read_family(dir)? {
            ModelFamily::Mamba1 => {
                let HfModel {
                    backbone,
                    embed,
                    lm_head,
                    vocab_size,
                    ..
                } = load_hf(dir)?;
                Self::new(backbone, embed, lm_head, vocab_size)
            }
            ModelFamily::Mamba2 => {
                let m = load_hf_mamba2(dir)?;
                let d_model = m.cfg.d_model;
                let backbone = Mamba2Backbone::new(m.weights, m.cfg, d_model)?;
                Self::new(backbone, m.embed, None, m.vocab_size)
            }
            ModelFamily::Mamba3 => {
                let m = load_hf_mamba3(dir)?;
                let d_model = m.cfg.d_model;
                let backbone = Mamba3Backbone::new(m.weights, m.cfg, d_model)?;
                Self::new(backbone, m.embed, None, m.vocab_size)
            }
            ModelFamily::Mamba3Mimo => {
                let m = load_hf_mamba3_mimo(dir)?;
                let d_model = m.cfg.d_model;
                let backbone = Mamba3MimoBackbone::new(m.weights, m.cfg, d_model)?;
                Self::new(backbone, m.embed, None, m.vocab_size)
            }
        }
    }

    /// The family of the backbone.
    pub fn family(&self) -> ModelFamily {
        self.backbone.family()
    }

    /// Wrap a CPU backbone of any family with its embedding and head.
    ///
    /// `embed` is row-major `[vocab_rows, d_model]`, possibly padded past
    /// `vocab_size`. `lm_head` is an untied head `[d_model, vocab_size]`
    /// (see [`crate::hf::embed::lm_head_logits_untied`]); `None` ties the
    /// head to the embedding. The weights come from a checkpoint (the
    /// loaders under [`crate::hf`]), a trainer's `snapshot_master` or a
    /// family's serializer, through the backbone's `new`.
    pub fn new(
        backbone: impl Into<AnyBackbone>,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
    ) -> Result<Self, String> {
        Self::from_backbone(backbone.into(), embed, lm_head, vocab_size)
    }

    #[deprecated(since = "0.8.0", note = "use `MambaLM::new`, which takes every family")]
    pub fn from_mamba3(
        backbone: Mamba3Backbone,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
    ) -> Result<Self, String> {
        Self::from_backbone(AnyBackbone::M3(backbone), embed, lm_head, vocab_size)
    }

    fn from_backbone(
        any: AnyBackbone,
        embed: Vec<f32>,
        lm_head: Option<Vec<f32>>,
        vocab_size: usize,
    ) -> Result<Self, String> {
        let d_model = any.d_model();
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
        if let Some(ref lm) = lm_head
            && lm.len() < vocab_size * d_model
        {
            return Err(format!(
                "lm_head length {} < vocab_size*d_model {}",
                lm.len(),
                vocab_size * d_model
            ));
        }

        let state = any.alloc_state();
        let scratch = any.alloc_scratch();
        Ok(Self {
            backbone: any,
            state,
            scratch,
            temporal: vec![0.0f32; d_model],
            prefill_inputs: Vec::new(),
            prefill_outputs: Vec::new(),
            embed,
            lm_head,
            logits: vec![0.0f32; vocab_size],
            logits_scratch: vec![0.0f32; vocab_size_padded],
            vocab_size,
            vocab_size_padded,
            d_model,
        })
    }

    /// The logits of the last step. `MambaLM` generates one sequence, so
    /// `b` must be 0; the argument mirrors `MambaGpuLM::last_logits`.
    /// Valid after a `generate` call.
    pub fn last_logits(&self, b: usize) -> &[f32] {
        assert_eq!(b, 0, "MambaLM is single-batch");
        &self.logits
    }

    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(params.max_tokens);
        self.generate_streaming(prompt, params, |tok, _| {
            tokens.push(tok);
        });
        tokens
    }

    pub fn generate_streaming(
        &mut self,
        prompt: &[u32],
        params: &SampleParams,
        mut cb: impl FnMut(u32, &str),
    ) {
        self.state.reset();
        let mut rng = Xoshiro256PlusPlus::new(params.seed);

        let seq_len = prompt.len();
        let dm = self.d_model;

        // An empty prompt has no position to continue from, so it generates
        // nothing, the same rule as the GPU model.
        if seq_len == 0 {
            return;
        }
        // Batch all prompt embeddings into one contiguous buffer
        self.prefill_inputs.resize(seq_len * dm, 0.0);
        self.prefill_outputs.resize(seq_len * dm, 0.0);
        for (t, &token_id) in prompt.iter().enumerate() {
            let emb = embed_lookup(&self.embed, token_id, dm, self.vocab_size);
            self.prefill_inputs[t * dm..(t + 1) * dm].copy_from_slice(emb);
        }

        // Process entire prompt in one forward_sequence call
        self.backbone.forward_sequence(
            &self.prefill_inputs[..seq_len * dm],
            &mut self.prefill_outputs[..seq_len * dm],
            &mut self.state,
            &mut self.scratch,
            seq_len,
        );

        // Last output → temporal for lm_head
        self.temporal
            .copy_from_slice(&self.prefill_outputs[(seq_len - 1) * dm..seq_len * dm]);
        self.compute_logits();

        let mut seen: Vec<u32> = prompt.to_vec();
        for _ in 0..params.max_tokens {
            let next = sample_token(&mut self.logits, params, &seen, &mut rng);

            if params.eos_token_ids.contains(&next) {
                break;
            }

            seen.push(next);
            cb(next, "");

            let hidden = embed_lookup(&self.embed, next, self.d_model, self.vocab_size);
            self.temporal.copy_from_slice(hidden);
            self.backbone.forward_step(
                hidden,
                &mut self.temporal,
                &mut self.state,
                &mut self.scratch,
            );
            self.compute_logits();
        }
    }

    fn compute_logits(&mut self) {
        if let Some(ref lm_head_w) = self.lm_head {
            lm_head_logits_untied(
                &mut self.logits,
                &self.temporal,
                lm_head_w,
                self.vocab_size,
                self.d_model,
            );
        } else {
            lm_head_logits(
                &mut self.logits,
                &self.temporal,
                &self.embed,
                self.vocab_size,
                self.vocab_size_padded,
                self.d_model,
                &mut self.logits_scratch,
            );
        }
    }

    pub fn save_state(&self) -> AnyState {
        self.state.clone()
    }

    pub fn restore_state(&mut self, s: AnyState) {
        self.state = s;
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }
}
