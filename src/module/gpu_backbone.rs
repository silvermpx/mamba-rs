//! One surface over the GPU backbones of the four families.
//!
//! Every family's GPU backbone has the same methods of its own: decode
//! steps, prompts in one pass, the decode step as a CUDA graph, and the
//! final-norm rows of the last step or prompt on the device for a head that
//! runs there. [`GpuBackbone`] names that surface for code that takes any of
//! them, and [`AnyGpuBackbone`] holds one chosen at run time, for example by
//! the family a checkpoint's config names.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::driver::sys::CUdeviceptr;

use crate::config::ModelFamily;
use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::mamba1::gpu::inference::MambaGpuBackbone;
use crate::mamba2::gpu::Mamba2GpuBackbone;
use crate::mamba3_mimo::gpu::Mamba3MimoGpuBackbone;
use crate::mamba3_siso::gpu::inference::Mamba3GpuBackbone;

/// What the GPU backbone of every family does.
///
/// ```rust,no_run
/// use mamba_rs::{GpuBackbone, GpuOptions, MambaConfig, MambaGpuBackbone, MambaWeights};
///
/// fn last_rows(bb: &mut impl GpuBackbone, prompt: &[f32], seq_len: usize) -> Vec<f32> {
///     bb.reset().unwrap();
///     bb.prefill(prompt, seq_len).unwrap();
///     let mut rows = vec![0.0f32; bb.batch() * bb.d_model()];
///     bb.download_temporal(&mut rows).unwrap();
///     rows
/// }
///
/// let cfg = MambaConfig::default();
/// let weights = MambaWeights::init(&cfg, 64, 7);
/// let mut bb = MambaGpuBackbone::new(&weights, cfg, 64, 1, GpuOptions::default()).unwrap();
/// let rows = last_rows(&mut bb, &vec![0.1f32; 16 * 64], 16);
/// assert_eq!(rows.len(), cfg.d_model);
/// ```
pub trait GpuBackbone {
    /// One token for every sequence: `input` is `[batch, input_dim]`,
    /// `output` (`[batch, d_model]`) receives the final-norm rows.
    fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String>;

    /// [`GpuBackbone::step`] without the download: the rows stay on the
    /// device for a consumer on the GPU.
    fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String>;

    /// The decode step's input rows on the device, `[batch, input_dim]`
    /// f32, in the backbone's own buffer: a caller on the GPU (a layer of
    /// its own below a segment) writes them there on the backbone's stream
    /// before [`GpuBackbone::step_on_device`], and no row crosses the host.
    /// [`GpuBackbone::capture_graph`] overwrites them.
    fn step_input_ptr(&self) -> CUdeviceptr;

    /// [`GpuBackbone::step_gpu_only`] on the rows at
    /// [`GpuBackbone::step_input_ptr`]; a captured decode graph replays with
    /// no upload. The bits are the host step's.
    fn step_on_device(&mut self) -> Result<(), String>;

    /// A prompt of `seq_len` tokens per sequence in one pass, from host rows
    /// `[batch, seq_len, input_dim]`, continuing from the current state.
    /// Afterwards the state holds the position after the prompt and the
    /// device rows the prompt's last position, as after a step. An empty
    /// prompt leaves both as they are.
    fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String>;

    /// The input rows of a prompt of `seq_len` tokens per sequence on the
    /// device, `[batch, seq_len, input_dim]` f32, in the backbone's own
    /// buffers of that length: a caller on the GPU writes them there on the
    /// backbone's stream before [`GpuBackbone::prefill_on_device`]. A prompt
    /// of another length replaces the buffers, and
    /// [`GpuBackbone::capture_prefill_graph`] overwrites the rows. An empty
    /// prompt has none.
    fn prompt_input_ptr(&mut self, seq_len: usize) -> Result<CUdeviceptr, String>;

    /// [`GpuBackbone::prefill`] on the rows at
    /// [`GpuBackbone::prompt_input_ptr`] of the same `seq_len`; a captured
    /// prompt graph replays with no upload. The bits are the host prompt's.
    fn prefill_on_device(&mut self, seq_len: usize) -> Result<(), String>;

    /// Every position's rows of the last prompt on the device, `[batch,
    /// seq_len, d_model]` in [`GpuBackbone::prompt_rows_dtype`]: the
    /// final-norm rows, or the residual stream of a backbone built with
    /// `final_norm = false`, for a layer of the caller's that takes the
    /// whole prompt, such as attention between two segments. The rows at
    /// [`GpuBackbone::temporal_ptr`] are each sequence's last position, in
    /// their own precision. None until a prompt length has buffers.
    fn prompt_rows_ptr(&self) -> Option<CUdeviceptr>;

    /// The precision of [`GpuBackbone::prompt_rows_ptr`]'s rows: f32, but
    /// for the final-norm rows of a bf16 or f16 Mamba-1 backbone, which are
    /// in the storage precision as its [`GpuBackbone::temporal_ptr`] rows.
    fn prompt_rows_dtype(&self) -> WeightDtype;

    /// Zero the recurrent state, the start of new sequences.
    fn reset(&mut self) -> Result<(), String>;

    /// Capture the decode step as one CUDA graph, which later steps replay.
    /// A graph captured before is dropped first, and the state and output
    /// rows stay where they were. Run the longest prompt first: after the
    /// capture the GEMM staging can no longer grow.
    fn capture_graph(&mut self) -> Result<(), String>;

    /// Whether steps replay a captured graph.
    fn has_graph(&self) -> bool;

    /// Drop the captured graph: steps run eagerly until the next capture.
    fn clear_graph(&mut self) -> Result<(), String>;

    /// Capture the prompt pass of `seq_len` tokens per sequence as one CUDA
    /// graph, which later [`GpuBackbone::prefill`] calls of that length
    /// replay with the eager bits; the state and rows stay where they were.
    /// A prompt of another length drops it. Capture the longest prompt
    /// before the decode graph: after a capture the GEMM staging can no
    /// longer grow.
    fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String>;

    /// Whether prompts of the kept length replay a captured graph.
    fn has_prefill_graph(&self) -> bool;

    /// Drop the prompt graph: prompts run eagerly until the next capture.
    fn clear_prefill_graph(&mut self);

    /// The final-norm rows of the last step or prompt, `[batch, d_model]`,
    /// on the device, in [`GpuBackbone::temporal_dtype`].
    fn temporal_ptr(&self) -> CUdeviceptr;

    /// The precision of [`GpuBackbone::temporal_ptr`]'s rows.
    fn temporal_dtype(&self) -> WeightDtype;

    /// Copy the final-norm rows of the last step or prompt to the host as
    /// f32.
    fn download_temporal(&self, output: &mut [f32]) -> Result<(), String>;

    /// Sequences per step.
    fn batch(&self) -> usize;

    /// Width of an input row.
    fn input_dim(&self) -> usize;

    /// Width of an output row.
    fn d_model(&self) -> usize;

    /// Storage precision of the weights.
    fn dtype(&self) -> WeightDtype;

    /// The CUDA context: the GEMM mode and route live here.
    fn ctx(&self) -> &GpuCtx;

    /// The stream every launch goes to.
    fn stream(&self) -> &Arc<CudaStream>;
}

// Each family's backbone has these methods of its own (inherent methods win
// method resolution), so the trait methods hand straight to them.
macro_rules! gpu_backbone_by_inherent_methods {
    ($($ty:ty),+ $(,)?) => {$(
        impl GpuBackbone for $ty {
            fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
                <$ty>::step(self, input, output)
            }
            fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
                <$ty>::step_gpu_only(self, input)
            }
            fn step_input_ptr(&self) -> CUdeviceptr {
                <$ty>::step_input_ptr(self)
            }
            fn step_on_device(&mut self) -> Result<(), String> {
                <$ty>::step_on_device(self)
            }
            fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String> {
                <$ty>::prefill(self, input, seq_len)
            }
            fn prompt_input_ptr(&mut self, seq_len: usize) -> Result<CUdeviceptr, String> {
                <$ty>::prompt_input_ptr(self, seq_len)
            }
            fn prefill_on_device(&mut self, seq_len: usize) -> Result<(), String> {
                <$ty>::prefill_on_device(self, seq_len)
            }
            fn prompt_rows_ptr(&self) -> Option<CUdeviceptr> {
                <$ty>::prompt_rows_ptr(self)
            }
            fn prompt_rows_dtype(&self) -> WeightDtype {
                <$ty>::prompt_rows_dtype(self)
            }
            fn reset(&mut self) -> Result<(), String> {
                <$ty>::reset(self)
            }
            fn capture_graph(&mut self) -> Result<(), String> {
                <$ty>::capture_graph(self)
            }
            fn has_graph(&self) -> bool {
                <$ty>::has_graph(self)
            }
            fn clear_graph(&mut self) -> Result<(), String> {
                <$ty>::clear_graph(self)
            }
            fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String> {
                <$ty>::capture_prefill_graph(self, seq_len)
            }
            fn has_prefill_graph(&self) -> bool {
                <$ty>::has_prefill_graph(self)
            }
            fn clear_prefill_graph(&mut self) {
                <$ty>::clear_prefill_graph(self)
            }
            fn temporal_ptr(&self) -> CUdeviceptr {
                <$ty>::temporal_ptr(self)
            }
            fn temporal_dtype(&self) -> WeightDtype {
                <$ty>::temporal_dtype(self)
            }
            fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
                <$ty>::download_temporal(self, output)
            }
            fn batch(&self) -> usize {
                <$ty>::batch(self)
            }
            fn input_dim(&self) -> usize {
                <$ty>::input_dim(self)
            }
            fn d_model(&self) -> usize {
                <$ty>::d_model(self)
            }
            fn dtype(&self) -> WeightDtype {
                <$ty>::dtype(self)
            }
            fn ctx(&self) -> &GpuCtx {
                <$ty>::ctx(self)
            }
            fn stream(&self) -> &Arc<CudaStream> {
                <$ty>::stream(self)
            }
        }
    )+};
}

gpu_backbone_by_inherent_methods!(
    MambaGpuBackbone,
    Mamba2GpuBackbone,
    Mamba3GpuBackbone,
    Mamba3MimoGpuBackbone,
);

/// The GPU backbone of any family, chosen at run time. Each backbone sits
/// in its own allocation: they differ in size by kilobytes, and the enum
/// moves as one pointer.
pub enum AnyGpuBackbone {
    M1(Box<MambaGpuBackbone>),
    M2(Box<Mamba2GpuBackbone>),
    M3(Box<Mamba3GpuBackbone>),
    M3Mimo(Box<Mamba3MimoGpuBackbone>),
}

impl AnyGpuBackbone {
    /// The family of the backbone held.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::M1(_) => ModelFamily::Mamba1,
            Self::M2(_) => ModelFamily::Mamba2,
            Self::M3(_) => ModelFamily::Mamba3,
            Self::M3Mimo(_) => ModelFamily::Mamba3Mimo,
        }
    }

    /// Whether the backbone held ends in its final norm, as a whole model
    /// does; a segment of a stack does not.
    pub fn final_norm(&self) -> bool {
        match self {
            Self::M1(b) => b.config().final_norm,
            Self::M2(b) => b.config().final_norm,
            Self::M3(b) => b.config().final_norm,
            Self::M3Mimo(b) => b.config().final_norm,
        }
    }

    fn inner(&self) -> &dyn GpuBackbone {
        match self {
            Self::M1(b) => b.as_ref(),
            Self::M2(b) => b.as_ref(),
            Self::M3(b) => b.as_ref(),
            Self::M3Mimo(b) => b.as_ref(),
        }
    }

    fn inner_mut(&mut self) -> &mut dyn GpuBackbone {
        match self {
            Self::M1(b) => b.as_mut(),
            Self::M2(b) => b.as_mut(),
            Self::M3(b) => b.as_mut(),
            Self::M3Mimo(b) => b.as_mut(),
        }
    }
}

impl GpuBackbone for AnyGpuBackbone {
    fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.inner_mut().step(input, output)
    }
    fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        self.inner_mut().step_gpu_only(input)
    }
    fn step_input_ptr(&self) -> CUdeviceptr {
        self.inner().step_input_ptr()
    }
    fn step_on_device(&mut self) -> Result<(), String> {
        self.inner_mut().step_on_device()
    }
    fn prefill(&mut self, input: &[f32], seq_len: usize) -> Result<(), String> {
        self.inner_mut().prefill(input, seq_len)
    }
    fn prompt_input_ptr(&mut self, seq_len: usize) -> Result<CUdeviceptr, String> {
        self.inner_mut().prompt_input_ptr(seq_len)
    }
    fn prefill_on_device(&mut self, seq_len: usize) -> Result<(), String> {
        self.inner_mut().prefill_on_device(seq_len)
    }
    fn prompt_rows_ptr(&self) -> Option<CUdeviceptr> {
        self.inner().prompt_rows_ptr()
    }
    fn prompt_rows_dtype(&self) -> WeightDtype {
        self.inner().prompt_rows_dtype()
    }
    fn reset(&mut self) -> Result<(), String> {
        self.inner_mut().reset()
    }
    fn capture_graph(&mut self) -> Result<(), String> {
        self.inner_mut().capture_graph()
    }
    fn has_graph(&self) -> bool {
        self.inner().has_graph()
    }
    fn clear_graph(&mut self) -> Result<(), String> {
        self.inner_mut().clear_graph()
    }
    fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String> {
        self.inner_mut().capture_prefill_graph(seq_len)
    }
    fn has_prefill_graph(&self) -> bool {
        self.inner().has_prefill_graph()
    }
    fn clear_prefill_graph(&mut self) {
        self.inner_mut().clear_prefill_graph()
    }
    fn temporal_ptr(&self) -> CUdeviceptr {
        self.inner().temporal_ptr()
    }
    fn temporal_dtype(&self) -> WeightDtype {
        self.inner().temporal_dtype()
    }
    fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
        self.inner().download_temporal(output)
    }
    fn batch(&self) -> usize {
        self.inner().batch()
    }
    fn input_dim(&self) -> usize {
        self.inner().input_dim()
    }
    fn d_model(&self) -> usize {
        self.inner().d_model()
    }
    fn dtype(&self) -> WeightDtype {
        self.inner().dtype()
    }
    fn ctx(&self) -> &GpuCtx {
        self.inner().ctx()
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.inner().stream()
    }
}

impl From<MambaGpuBackbone> for AnyGpuBackbone {
    fn from(b: MambaGpuBackbone) -> Self {
        Self::M1(Box::new(b))
    }
}

impl From<Mamba2GpuBackbone> for AnyGpuBackbone {
    fn from(b: Mamba2GpuBackbone) -> Self {
        Self::M2(Box::new(b))
    }
}

impl From<Mamba3GpuBackbone> for AnyGpuBackbone {
    fn from(b: Mamba3GpuBackbone) -> Self {
        Self::M3(Box::new(b))
    }
}

impl From<Mamba3MimoGpuBackbone> for AnyGpuBackbone {
    fn from(b: Mamba3MimoGpuBackbone) -> Self {
        Self::M3Mimo(Box::new(b))
    }
}
