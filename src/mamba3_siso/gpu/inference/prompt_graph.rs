//! A SISO prompt captured as one CUDA graph through the prefill's own
//! window graph ([`Mamba3PrefillGraph`]), over one prompt length. The
//! prompt's buffers hold the graph; the host uploads the rows before a
//! replay, and in the half precisions the prompt's last rows are cast to
//! the decode rows after it, outside the graph.

use super::super::prefill::{Mamba3PrefillGraph, Mamba3PrefillRun};
use super::super::state::GpuMamba3StateBufs;
use super::{M3BackboneEngine, M3BackbonePrompt, M3BackboneScratch, Mamba3GpuBackbone};
use crate::gpu::buffers::{DeviceKeep, GpuBuffer};

/// Run `f` on the prefill run of `engine` over `mamba_input`, carrying the
/// state across the window.
fn with_run<R>(
    engine: &M3BackboneEngine,
    seq_len: usize,
    mamba_input: &GpuBuffer,
    f: impl FnOnce(&Mamba3PrefillRun<'_>) -> R,
) -> R {
    match engine {
        M3BackboneEngine::F32(e) => {
            let dims = e.prefill_dims(seq_len);
            f(&Mamba3PrefillRun {
                ctx: &e.ctx,
                kernels: &e.kernels,
                dims: &dims,
                weights: &e.weights,
                mamba_input,
                identity_proj: e.identity_proj,
                carry_state: true,
            })
        }
        M3BackboneEngine::Mixed(e) => {
            let dims = e.engine.prefill_dims(seq_len);
            f(&Mamba3PrefillRun {
                ctx: &e.engine.ctx,
                kernels: &e.engine.kernels,
                dims: &dims,
                weights: &e.mixed_weights,
                mamba_input,
                identity_proj: e.engine.identity_proj,
                carry_state: true,
            })
        }
    }
}

impl Mamba3GpuBackbone {
    /// The uploaded prompt through its graph, then in the half precisions
    /// the cast of its last rows to the decode rows.
    pub(super) fn replay_prompt(&mut self, prompt: &M3BackbonePrompt) -> Result<(), String> {
        let Some(g) = &prompt.graph else {
            return Err("the prompt holds no graph".into());
        };
        let states = GpuMamba3StateBufs {
            ssm: &mut self.state.ssm_state,
            k: &mut self.state.k_state,
            v: &mut self.state.v_state,
            angle: &mut self.state.angle_state,
        };
        match (&self.engine, &mut self.scratch) {
            (M3BackboneEngine::F32(e), M3BackboneScratch::F32(sc)) => g.replay(
                &e.ctx,
                &e.kernels,
                &e.weights,
                &prompt.input,
                &states,
                &sc.temporal,
            ),
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                g.replay(
                    &e.engine.ctx,
                    &e.engine.kernels,
                    &e.mixed_weights,
                    &prompt.input,
                    &states,
                    &sc.prefill_hidden,
                )?;
                e.hand_out_prefill_hidden(sc)
            }
            _ => Err("M3 prefill: engine/scratch precision arms disagree".to_string()),
        }
    }

    /// Capture the prompt pass of `seq_len` tokens per sequence as one CUDA
    /// graph, which later [`Self::prefill`] calls of that length replay with
    /// the eager bits. The capture leaves the state and the rows where the
    /// caller left them: its warm-up prompt on zero input is undone. A
    /// prompt of another length drops the graph with the buffers it runs
    /// on. Capture the longest prompt before the decode graph: afterwards
    /// the GEMM staging can no longer grow.
    pub fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String> {
        let want = self.batch() * seq_len * self.input_dim();
        if want == 0 {
            return Err("a prompt graph needs at least one row".into());
        }
        let kept = self.prompt.take().filter(|p| p.seq_len == seq_len);
        let mut prompt = match kept {
            Some(prompt) => prompt,
            None => self.alloc_prompt(seq_len)?,
        };
        let result = self.capture_prompt(&mut prompt);
        self.prompt = Some(prompt);
        result
    }

    fn capture_prompt(&mut self, prompt: &mut M3BackbonePrompt) -> Result<(), String> {
        prompt.graph = None;
        let stream = self.stream().clone();
        let f = std::mem::size_of::<f32>();
        let s = &self.state;
        let keep = DeviceKeep::take(
            &stream,
            &[
                (s.ssm_state.cached_ptr(), s.ssm_state.len() * f),
                (s.k_state.cached_ptr(), s.k_state.len() * f),
                (s.v_state.cached_ptr(), s.v_state.len() * f),
                (s.angle_state.cached_ptr(), s.angle_state.len() * f),
                (
                    self.temporal_ptr(),
                    self.batch() * self.d_model() * self.temporal_dtype().size_bytes(),
                ),
            ],
        )?;
        let zeros = vec![0.0f32; prompt.input.len()];
        prompt.input.upload(&stream, &zeros)?;
        self.prefill_sequence(&mut prompt.prefill, &prompt.input, prompt.seq_len, true)?;
        keep.restore(&stream)?;
        let Self {
            engine,
            state,
            scratch,
            ..
        } = self;
        let last_hidden = match scratch {
            M3BackboneScratch::F32(sc) => &mut sc.temporal,
            M3BackboneScratch::Mixed(sc) => &mut sc.prefill_hidden,
        };
        let states = GpuMamba3StateBufs {
            ssm: &mut state.ssm_state,
            k: &mut state.k_state,
            v: &mut state.v_state,
            angle: &mut state.angle_state,
        };
        let M3BackbonePrompt {
            graph,
            seq_len,
            input,
            prefill,
        } = prompt;
        // SAFETY: the graph reads this backbone's own state and rows and
        // the prompt's buffers; the prompt holds the graph and drops it
        // first.
        let captured = with_run(engine, *seq_len, input, |run| unsafe {
            Mamba3PrefillGraph::capture(prefill, run, states, last_hidden)
        })?;
        *graph = Some(captured);
        Ok(())
    }

    /// Whether prompts of the kept length replay a captured graph.
    pub fn has_prefill_graph(&self) -> bool {
        self.prompt.as_ref().is_some_and(|p| p.graph.is_some())
    }

    /// Drop the prompt graph, once a replay in flight finished: prompts run
    /// eagerly until the next capture.
    pub fn clear_prefill_graph(&mut self) {
        if let Some(p) = self.prompt.as_mut() {
            p.graph = None;
        }
    }
}
