//! The Mamba-2 GPU backbone: the engine together with the recurrent state
//! and the buffers it runs on, with the surface every family's GPU backbone
//! has.

use std::sync::Arc;

use crate::gpu::GpuOptions;
use crate::gpu::context::GpuCtx;
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::Mamba2Weights;

use super::inference::{Mamba2GpuEngine, Mamba2GpuPrefill, Mamba2GpuScratch, Mamba2GpuState};

/// A Mamba-2 model on the GPU that owns its state and buffers.
///
/// Decode with [`Self::step`], run a prompt in one pass with
/// [`Self::prefill`], start a new sequence with [`Self::reset`], and replay
/// the decode step as one CUDA graph after [`Self::capture_graph`]. The
/// final-norm rows of the last step or prompt stay on the device
/// ([`Self::temporal_ptr`]) for a head that runs there, and
/// [`Self::download_temporal`] copies them to the host.
pub struct Mamba2GpuBackbone {
    // The engine drops first, and its captured graph with it, before the
    // state and buffers the graph reads.
    engine: Mamba2GpuEngine,
    state: Mamba2GpuState,
    scratch: Mamba2GpuScratch,
    prefill: Option<Mamba2GpuPrefill>,
    dtype: WeightDtype,
}

impl Mamba2GpuBackbone {
    /// Build the GPU backbone of `weights` for `batch` sequences, on the
    /// device, in the storage precision and in the GEMM mode `opts` names
    /// ([`GpuOptions::default`] is device 0, f32 and the mode
    /// `MAMBA_RS_GEMM_MODE` names). Configuration, CUDA, upload or
    /// allocation errors are returned.
    pub fn new(
        weights: &Mamba2Weights,
        cfg: Mamba2Config,
        input_dim: usize,
        batch: usize,
        opts: GpuOptions,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(opts.device)?;
        let engine = match opts.mode {
            Some(mode) => Mamba2GpuEngine::new_with_mode(
                &device, weights, cfg, input_dim, batch, opts.dtype, mode,
            )?,
            None => Mamba2GpuEngine::new(&device, weights, cfg, input_dim, batch, opts.dtype)?,
        };
        let state = engine.alloc_state()?;
        let scratch = engine.alloc_scratch()?;
        Ok(Self {
            engine,
            state,
            scratch,
            prefill: None,
            dtype: opts.dtype,
        })
    }

    /// One token for every sequence: `input` is `[batch, input_dim]`,
    /// `output` (`[batch, d_model]`) receives the final-norm rows.
    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.engine
            .step(input, output, &mut self.state, &mut self.scratch)
    }

    /// [`Self::step`] without the download: the rows stay on the device
    /// for a consumer on the GPU.
    pub fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        self.engine
            .step_gpu_only(input, &mut self.state, &mut self.scratch)
    }

    /// A prompt of `seq_len` tokens per sequence in one pass, from host
    /// rows `[batch, seq_len, input_dim]`, continuing from the current
    /// state. Afterwards the state holds the position after the prompt and
    /// the device rows ([`Self::temporal_ptr`]) the prompt's last position.
    /// The buffers of the last prompt length are kept for the next call.
    pub fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String> {
        if self.prefill.as_ref().map(Mamba2GpuPrefill::seq_len) != Some(seq_len) {
            self.prefill = None;
            self.prefill = Some(self.engine.alloc_prefill(seq_len)?);
        }
        let Some(pf) = self.prefill.as_mut() else {
            return Err("Mamba-2 prefill buffers missing".into());
        };
        self.engine
            .prefill_gpu_only(input, &mut self.state, pf, &mut self.scratch)
    }

    /// Zero the recurrent state, the start of new sequences.
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.ctx)
    }

    /// Capture the decode step as one CUDA graph: an eager step on zero
    /// input, a reset, then the capture. Later steps replay the graph.
    /// Run the longest prompt before capturing: afterwards the GEMM staging
    /// can no longer grow.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        let input = vec![0.0f32; self.batch() * self.input_dim()];
        let mut output = vec![0.0f32; self.batch() * self.d_model()];
        self.step(&input, &mut output)?;
        self.reset()?;
        // SAFETY: the graph reads this backbone's own state and buffers,
        // which stay allocated and in place for as long as the engine (and
        // its graph, dropped first) lives.
        unsafe {
            self.engine
                .capture_graph(&mut self.state, &mut self.scratch)
        }
    }

    /// Whether steps replay a captured graph.
    pub fn has_graph(&self) -> bool {
        self.engine.has_graph()
    }

    /// The final-norm rows of the last step or prompt, f32
    /// `[batch, d_model]`, on the device.
    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.scratch.final_rows().cached_ptr()
    }

    /// The precision of [`Self::temporal_ptr`]'s rows: always f32.
    pub fn temporal_dtype(&self) -> WeightDtype {
        WeightDtype::F32
    }

    /// Copy the final-norm rows of the last step or prompt to the host.
    pub fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
        let stream = &self.engine.ctx.stream;
        stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 temporal sync: {e:?}"))?;
        self.scratch.final_rows().download(stream, output)?;
        stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 temporal sync: {e:?}"))
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba2Config {
        self.engine.config()
    }

    /// Sequences per step.
    pub fn batch(&self) -> usize {
        self.engine.batch()
    }

    /// Width of an input row.
    pub fn input_dim(&self) -> usize {
        self.engine.input_dim()
    }

    /// Width of an output row.
    pub fn d_model(&self) -> usize {
        self.engine.config().d_model
    }

    /// Storage precision of the weights.
    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The CUDA context: the GEMM mode and route live here.
    pub fn ctx(&self) -> &GpuCtx {
        self.engine.ctx()
    }

    /// The stream every launch goes to.
    pub fn stream(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.engine.ctx.stream
    }
}
