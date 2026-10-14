//! The Mamba-3 MIMO GPU backbone: the engine together with the recurrent
//! state and the windows it runs on, with the surface every family's GPU
//! backbone has.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;

use crate::gpu::GpuOptions;
use crate::gpu::buffers::DeviceKeep;
use crate::gpu::context::GpuCtx;
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::train::RowSource;
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::Mamba3MimoWeights;

use super::inference::{Mamba3MimoGpuEngine, Mamba3MimoGpuState, Mamba3MimoGpuWindow};

/// A Mamba-3 MIMO model on the GPU that owns its state and buffers.
///
/// Decode with [`Self::step`], run a prompt in one pass with
/// [`Self::prefill`], start a new sequence with [`Self::reset`], and replay
/// the decode step as one CUDA graph after [`Self::capture_graph`] and the
/// prompt pass of one length after [`Self::capture_prefill_graph`]. The
/// final-norm rows of the last step or prompt stay on the device
/// ([`Self::temporal_ptr`]) for a head that runs there, and
/// [`Self::download_temporal`] copies them to the host.
pub struct Mamba3MimoGpuBackbone {
    // Each window holds its graph, which keeps the kernels' module and the
    // context's resources alive until it is destroyed.
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

    /// The decode step's input rows on the device, `[batch, input_dim]`
    /// f32, in the backbone's own buffer: a layer of the caller's on the
    /// GPU writes them there on the backbone's stream before
    /// [`Self::step_on_device`].
    pub fn step_input_ptr(&self) -> CUdeviceptr {
        self.step_window.input_ptr()
    }

    /// [`Self::step_gpu_only`] on the rows at [`Self::step_input_ptr`]; a
    /// captured decode graph replays with no upload. The bits are the host
    /// step's.
    pub fn step_on_device(&mut self) -> Result<(), String> {
        self.engine
            .run_from(RowSource::Device, &mut self.state, &mut self.step_window)
    }

    /// A prompt of `seq_len` tokens per sequence in one pass, from host
    /// rows `[batch, seq_len, input_dim]`, continuing from the current
    /// state. Afterwards the state holds the position after the prompt and
    /// the device rows ([`Self::temporal_ptr`]) the prompt's last position.
    /// The window of the last prompt length is kept for the next call.
    pub fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String> {
        // An empty prompt leaves the state and the rows as they are.
        if seq_len == 0 && input.is_empty() {
            return Ok(());
        }
        self.keep_prompt(seq_len)?;
        let Some(win) = self.prompt_window.as_mut() else {
            return Err("MIMO prompt window missing".into());
        };
        self.engine.run_gpu_only(input, &mut self.state, win)?;
        self.engine.gather_last_rows(win, &mut self.step_window)
    }

    /// The input rows of a prompt of `seq_len` tokens per sequence on the
    /// device, `[batch, seq_len, input_dim]` f32, in the backbone's own
    /// window of that length (a prompt of another length replaces it): a
    /// layer of the caller's on the GPU writes them there on the backbone's
    /// stream before [`Self::prefill_on_device`].
    pub fn prompt_input_ptr(&mut self, seq_len: usize) -> Result<CUdeviceptr, String> {
        if seq_len == 0 {
            return Err("MIMO prompt input: an empty prompt has no rows".into());
        }
        self.keep_prompt(seq_len)?;
        self.prompt_window
            .as_ref()
            .map(Mamba3MimoGpuWindow::input_ptr)
            .ok_or_else(|| "MIMO prompt window missing".into())
    }

    /// [`Self::prefill`] on the rows at [`Self::prompt_input_ptr`] of the
    /// same `seq_len`; a captured prompt graph replays with no upload. The
    /// bits are the host prompt's.
    pub fn prefill_on_device(&mut self, seq_len: usize) -> Result<(), String> {
        if seq_len == 0 {
            return Ok(());
        }
        let Some(win) = self
            .prompt_window
            .as_mut()
            .filter(|win| win.seq_len() == seq_len)
        else {
            return Err(format!(
                "MIMO prefill on the device: no input rows of {seq_len} tokens, \
                 take prompt_input_ptr first"
            ));
        };
        self.engine
            .run_from(RowSource::Device, &mut self.state, win)?;
        self.engine.gather_last_rows(win, &mut self.step_window)
    }

    /// Every position's rows of the last prompt on the device, `[batch,
    /// seq_len, d_model]` in [`Self::prompt_rows_dtype`]: the final-norm
    /// rows, or the residual stream of a backbone built with `final_norm =
    /// false`, for a layer of the caller's that takes the whole prompt.
    /// None until a prompt length has a window.
    pub fn prompt_rows_ptr(&self) -> Option<CUdeviceptr> {
        self.prompt_window
            .as_ref()
            .map(|win| win.final_rows().cached_ptr())
    }

    /// The precision of [`Self::prompt_rows_ptr`]'s rows: always f32.
    pub fn prompt_rows_dtype(&self) -> WeightDtype {
        WeightDtype::F32
    }

    /// Keep the window of a prompt of `seq_len` tokens per sequence,
    /// replacing one of another length with the graph it holds.
    fn keep_prompt(&mut self, seq_len: usize) -> Result<(), String> {
        if self
            .prompt_window
            .as_ref()
            .map(Mamba3MimoGpuWindow::seq_len)
            != Some(seq_len)
        {
            self.prompt_window = None;
            self.prompt_window = Some(self.engine.alloc_window(seq_len)?);
        }
        Ok(())
    }

    /// Zero the recurrent state, the start of new sequences.
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.ctx)
    }

    /// Capture the prompt pass of `seq_len` tokens per sequence as one CUDA
    /// graph, which later [`Self::prefill`] calls of that length replay with
    /// the eager bits. The capture leaves the state and the rows where the
    /// caller left them: its warm-up prompt on zero input is undone. A
    /// prompt of another length drops the graph with the window it runs
    /// on. Capture the longest prompt before the decode graph: afterwards
    /// the GEMM staging can no longer grow.
    pub fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String> {
        let zeros = vec![0.0f32; self.batch() * seq_len * self.input_dim()];
        self.keep_prompt(seq_len)?;
        let Some(win) = self.prompt_window.as_mut() else {
            return Err("MIMO prompt window missing".into());
        };
        win.clear_graph();
        let stream = self.engine.ctx.stream.clone();
        let keep = keep_state(&stream, &self.state, &self.step_window)?;
        self.engine.run_gpu_only(&zeros, &mut self.state, win)?;
        keep.restore(&stream)?;
        // SAFETY: the graph reads this backbone's own state and the
        // window's buffers; the window holds the graph and drops it first.
        unsafe { self.engine.capture_graph(&mut self.state, win) }
    }

    /// Whether prompts of the kept length replay a captured graph.
    pub fn has_prefill_graph(&self) -> bool {
        self.prompt_window
            .as_ref()
            .is_some_and(Mamba3MimoGpuWindow::has_graph)
    }

    /// Drop the prompt graph: prompts run eagerly until the next capture.
    pub fn clear_prefill_graph(&mut self) {
        if let Some(win) = self.prompt_window.as_mut() {
            win.clear_graph();
        }
    }

    /// Capture the decode step as one CUDA graph, which later steps replay.
    /// A graph captured before is dropped first. The capture leaves the
    /// recurrent state and the output rows where the caller left them: its
    /// warm-up step on zero input is undone before the graph is recorded.
    /// Run and capture the longest prompt before: afterwards the GEMM
    /// staging can no longer grow.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.step_window.clear_graph();
        let stream = self.engine.ctx.stream.clone();
        let keep = keep_state(&stream, &self.state, &self.step_window)?;
        let input = vec![0.0f32; self.batch() * self.input_dim()];
        let mut output = vec![0.0f32; self.batch() * self.d_model()];
        self.step(&input, &mut output)?;
        keep.restore(&stream)?;
        // SAFETY: the graph reads this backbone's own state and the step
        // window's buffers; the window holds the graph and drops it first.
        unsafe {
            self.engine
                .capture_graph(&mut self.state, &mut self.step_window)
        }
    }

    /// Whether steps replay a captured graph.
    pub fn has_graph(&self) -> bool {
        self.step_window.has_graph()
    }

    /// Drop the captured graph: steps run eagerly until the next capture.
    pub fn clear_graph(&mut self) -> Result<(), String> {
        self.step_window.clear_graph();
        Ok(())
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

/// Copy the state and the step window's rows aside, for a warm-up run a
/// capture undoes.
fn keep_state(
    stream: &Arc<cudarc::driver::CudaStream>,
    state: &Mamba3MimoGpuState,
    step_window: &Mamba3MimoGpuWindow,
) -> Result<DeviceKeep, String> {
    let f = std::mem::size_of::<f32>();
    let rows = step_window.final_rows();
    DeviceKeep::take(
        stream,
        &[
            (state.ssm.cached_ptr(), state.ssm.len() * f),
            (state.k.cached_ptr(), state.k.len() * f),
            (state.v.cached_ptr(), state.v.len() * f),
            (state.angle.cached_ptr(), state.angle.len() * f),
            (rows.cached_ptr(), rows.len() * f),
        ],
    )
}
