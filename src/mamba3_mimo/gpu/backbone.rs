//! The Mamba-3 MIMO GPU backbone: the engine together with the recurrent
//! state and the windows it runs on, with the surface every family's GPU
//! backbone has.

use std::sync::Arc;

use crate::gpu::GpuOptions;
use crate::gpu::buffers::DeviceKeep;
use crate::gpu::context::GpuCtx;
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::Mamba3MimoWeights;

use super::inference::{Mamba3MimoGpuEngine, Mamba3MimoGpuState, Mamba3MimoGpuWindow};

/// A Mamba-3 MIMO model on the GPU that owns its state and buffers.
///
/// Decode with [`Self::step`], run a prompt in one pass with
/// [`Self::prefill`], start a new sequence with [`Self::reset`], and replay
/// the decode step as one CUDA graph after [`Self::capture_graph`]. The
/// final-norm rows of the last step or prompt stay on the device
/// ([`Self::temporal_ptr`]) for a head that runs there, and
/// [`Self::download_temporal`] copies them to the host.
pub struct Mamba3MimoGpuBackbone {
    // The engine drops first, and its captured graph with it, before the
    // state and windows the graph reads.
    engine: Mamba3MimoGpuEngine,
    state: Mamba3MimoGpuState,
    step_window: Mamba3MimoGpuWindow,
    prompt_window: Option<Mamba3MimoGpuWindow>,
    dtype: WeightDtype,
}

impl Mamba3MimoGpuBackbone {
    /// Build the GPU backbone of `weights` for `batch` sequences, on the
    /// device, in the storage precision and in the GEMM mode `opts` names
    /// ([`GpuOptions::default`] is device 0, f32 and the mode
    /// `MAMBA_RS_GEMM_MODE` names). Configuration, CUDA, upload or
    /// allocation errors are returned.
    pub fn new(
        weights: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
        batch: usize,
        opts: GpuOptions,
    ) -> Result<Self, String> {
        let mut cfg = cfg;
        cfg.scan_mode = opts.scan_mode_or(cfg.scan_mode);
        let device = GpuDevice::new(opts.device)?;
        let engine = match opts.mode {
            Some(mode) => Mamba3MimoGpuEngine::new_with_mode(
                &device, weights, cfg, input_dim, batch, opts.dtype, mode,
            )?,
            None => Mamba3MimoGpuEngine::new(&device, weights, cfg, input_dim, batch, opts.dtype)?,
        };
        let state = engine.alloc_state()?;
        let step_window = engine.alloc_window(1)?;
        Ok(Self {
            engine,
            state,
            step_window,
            prompt_window: None,
            dtype: opts.dtype,
        })
    }

    /// One token for every sequence: `input` is `[batch, input_dim]`,
    /// `output` (`[batch, d_model]`) receives the final-norm rows.
    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.engine
            .run(input, output, &mut self.state, &mut self.step_window)
    }

    /// [`Self::step`] without the download: the rows stay on the device
    /// for a consumer on the GPU.
    pub fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        self.engine
            .run_gpu_only(input, &mut self.state, &mut self.step_window)
    }

    /// A prompt of `seq_len` tokens per sequence in one pass, from host
    /// rows `[batch, seq_len, input_dim]`, continuing from the current
    /// state. Afterwards the state holds the position after the prompt and
    /// the device rows ([`Self::temporal_ptr`]) the prompt's last position.
    /// The window of the last prompt length is kept for the next call.
    pub fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String> {
        if self
            .prompt_window
            .as_ref()
            .map(Mamba3MimoGpuWindow::seq_len)
            != Some(seq_len)
        {
            self.prompt_window = None;
            self.prompt_window = Some(self.engine.alloc_window(seq_len)?);
        }
        let Some(win) = self.prompt_window.as_mut() else {
            return Err("MIMO prompt window missing".into());
        };
        self.engine.run_gpu_only(input, &mut self.state, win)?;
        self.engine.gather_last_rows(win, &mut self.step_window)
    }

    /// Zero the recurrent state, the start of new sequences.
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.ctx)
    }

    /// Capture the decode step as one CUDA graph, which later steps replay.
    /// A graph captured before is dropped first. The capture leaves the
    /// recurrent state and the output rows where the caller left them: its
    /// warm-up step on zero input is undone before the graph is recorded.
    /// Run the longest prompt before capturing: afterwards the GEMM staging
    /// can no longer grow.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.engine.clear_graph()?;
        let stream = self.engine.ctx.stream.clone();
        let f = std::mem::size_of::<f32>();
        let rows = self.step_window.final_rows();
        let keep = DeviceKeep::take(
            &stream,
            &[
                (self.state.ssm.cached_ptr(), self.state.ssm.len() * f),
                (self.state.k.cached_ptr(), self.state.k.len() * f),
                (self.state.v.cached_ptr(), self.state.v.len() * f),
                (self.state.angle.cached_ptr(), self.state.angle.len() * f),
                (rows.cached_ptr(), rows.len() * f),
            ],
        )?;
        let input = vec![0.0f32; self.batch() * self.input_dim()];
        let mut output = vec![0.0f32; self.batch() * self.d_model()];
        self.step(&input, &mut output)?;
        keep.restore(&stream)?;
        // SAFETY: the graph reads this backbone's own state and step
        // window, which stay allocated and in place for as long as the
        // engine (and its graph, dropped first) lives.
        unsafe {
            self.engine
                .capture_graph(&mut self.state, &mut self.step_window)
        }
    }

    /// Whether steps replay a captured graph.
    pub fn has_graph(&self) -> bool {
        self.engine.has_graph()
    }

    /// Drop the captured graph: steps run eagerly until the next capture.
    pub fn clear_graph(&mut self) -> Result<(), String> {
        self.engine.clear_graph()
    }

    /// The final-norm rows of the last step or prompt, f32
    /// `[batch, d_model]`, on the device.
    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.step_window.final_rows().cached_ptr()
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
            .map_err(|e| format!("MIMO temporal sync: {e:?}"))?;
        self.step_window.final_rows().download(stream, output)?;
        stream
            .synchronize()
            .map_err(|e| format!("MIMO temporal sync: {e:?}"))
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba3MimoConfig {
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
