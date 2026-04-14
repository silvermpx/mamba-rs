//! Language model wrapper for Mamba text generation.

use std::path::Path;

use crate::mamba_ssm::cpu::inference::MambaStepScratch;
use crate::mamba3_siso::cpu::inference::Mamba3StepScratch;
use crate::mamba3_siso::state::Mamba3State;
use crate::state::MambaState;

use super::backbone::MambaBackbone;
use super::backbone3::Mamba3Backbone;
use super::sample::{SampleParams, Xoshiro256PlusPlus, sample_token};

use crate::hf::embed::{embed_lookup, lm_head_logits, lm_head_logits_untied};
use crate::hf::load::{HfModel, load_hf};

// ---------------------------------------------------------------------------
// AnyBackbone / AnyState / AnyScratch — enum dispatch
// ---------------------------------------------------------------------------

pub enum AnyBackbone {
    M1(MambaBackbone),
    M3(Mamba3Backbone),
}

#[derive(Clone)]
pub enum AnyState {
    M1(MambaState),
    M3(Mamba3State),
}

impl AnyState {
    pub fn reset(&mut self) {
        match self {
            Self::M1(s) => s.reset(),
            Self::M3(s) => s.reset(),
        }
    }
}

pub enum AnyScratch {
    M1(MambaStepScratch),
    M3(Mamba3StepScratch),
}

impl AnyBackbone {
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut AnyState,
        scratch: &mut AnyScratch,
    ) {
        match (self, state, scratch) {
            (Self::M1(bb), AnyState::M1(st), AnyScratch::M1(sc)) => {
                bb.forward_step(input, output, st, sc);
            }
            (Self::M3(bb), AnyState::M3(st), AnyScratch::M3(sc)) => {
                bb.forward_step(input, output, sc, &mut st.layers);
            }
            _ => panic!("backbone/state/scratch type mismatch"),
        }
    }

    pub fn d_model(&self) -> usize {
        match self {
            Self::M1(bb) => bb.config().d_model,
            Self::M3(bb) => bb.d_model(),
        }
    }

    pub fn alloc_state(&self) -> AnyState {
        match self {
            Self::M1(bb) => AnyState::M1(bb.alloc_state()),
            Self::M3(bb) => AnyState::M3(bb.alloc_state()),
        }
    }

    pub fn alloc_scratch(&self) -> AnyScratch {
        match self {
            Self::M1(bb) => AnyScratch::M1(bb.alloc_scratch()),
            Self::M3(bb) => AnyScratch::M3(bb.alloc_scratch()),
        }
    }

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
                bb.forward_sequence(inputs, outputs, st, sc, seq_len);
            }
            (Self::M3(bb), AnyState::M3(st), AnyScratch::M3(sc)) => {
                let dm = bb.d_model();
                for t in 0..seq_len {
                    let inp = &inputs[t * dm..(t + 1) * dm];
                    let out = &mut outputs[t * dm..(t + 1) * dm];
                    bb.forward_step(inp, out, sc, &mut st.layers);
                }
            }
            _ => panic!("backbone/state/scratch type mismatch"),
        }
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
    pub fn from_hf(dir: &Path) -> Result<Self, String> {
        let HfModel {
            backbone,
            embed,
            lm_head,
            vocab_size,
            vocab_size_padded,
            d_model,
        } = load_hf(dir)?;

        let any = AnyBackbone::M1(backbone);
        let state = any.alloc_state();
        let scratch = any.alloc_scratch();
        let temporal = vec![0.0f32; d_model];
        let logits = vec![0.0f32; vocab_size];
        let logits_scratch = vec![0.0f32; vocab_size_padded];

        Ok(Self {
            backbone: any,
            state,
            scratch,
            temporal,
            prefill_inputs: Vec::new(),
            prefill_outputs: Vec::new(),
            embed,
            lm_head,
            logits,
            logits_scratch,
            vocab_size,
            vocab_size_padded,
            d_model,
        })
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

        // Empty-prompt path: decode directly from zero state. The prefill path
        // below assumes seq_len >= 1 (reads `prefill_outputs[seq_len - 1]`);
        // for seq_len == 0 we skip it and start decoding with zero temporal.
        if seq_len == 0 {
            self.temporal.fill(0.0);
            self.compute_logits();
        } else {
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
        }

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
