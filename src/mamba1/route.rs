//! Which Mamba-1 work a request takes on the GPU, or why it does not.
//!
//! The module needs no CUDA feature: the census and a CPU caller ask it
//! before building anything, and the GPU constructors ask it first.

use crate::mamba1::config::MambaConfig;
use crate::matrix::{self, Path, Plan, Refusal, Request, Scan};
use crate::precision::GemmMode;

/// The widths a Mamba-1 launch takes as `int`s, over windows of `seq_len`
/// tokens of `batch` sequences.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntWidths {
    pub batch: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub input_dim: usize,
    pub d_inner: usize,
    pub xdbl_dim: usize,
    pub d_conv: usize,
    pub d_state: usize,
}

impl IntWidths {
    pub(crate) fn of(
        cfg: &MambaConfig,
        input_dim: usize,
        (batch, seq_len): (usize, usize),
    ) -> Self {
        Self {
            batch,
            seq_len,
            d_model: cfg.d_model,
            input_dim,
            d_inner: cfg.d_inner(),
            xdbl_dim: cfg.xdbl_dim(),
            d_conv: cfg.d_conv,
            d_state: cfg.d_state,
        }
    }

    /// The values the kernels take as `int`s: the dimensions, the rows of
    /// a window, their widths (the in_proj row of two inner widths, the
    /// t-major row of the parallel scan) and the conv tap rows.
    pub(crate) fn args(&self) -> [(&'static str, usize); 10] {
        [
            ("batch", self.batch),
            ("seq_len", self.seq_len),
            (
                "batch * seq_len rows",
                self.batch.saturating_mul(self.seq_len),
            ),
            ("t-major row", self.seq_len.saturating_add(7)),
            ("d_model", self.d_model),
            ("input width", self.input_dim),
            ("in_proj row", self.d_inner.saturating_mul(2)),
            ("x_proj row", self.xdbl_dim),
            (
                "conv taps of all channels",
                self.d_inner.saturating_mul(self.d_conv),
            ),
            (
                "states of all channels",
                self.d_inner.saturating_mul(self.d_state),
            ),
        ]
    }
}

/// The linear layers of a training step, `(name, input width, output
/// width)`; a model without an input projection takes rows `d_model` wide,
/// which no other layer's operands undercut.
fn linears(cfg: &MambaConfig, input_dim: usize) -> [(&'static str, usize, usize); 7] {
    let (di, h) = (cfg.d_inner(), cfg.mlp_hidden());
    [
        ("input_proj", input_dim, cfg.d_model),
        ("in_proj", cfg.d_model, 2 * di),
        ("x_proj", di, cfg.xdbl_dim()),
        ("dt_proj", cfg.dt_rank(), di),
        ("out_proj", di, cfg.d_model),
        ("mlp fc1", cfg.d_model, 2 * h),
        ("mlp fc2", h, cfg.d_model),
    ]
}

/// Whether and how the GPU runs `req` on a Mamba-1 model of `cfg` fed rows
/// `input_dim` wide. `cfg` must pass [`MambaConfig::validate`].
pub fn admit(cfg: &MambaConfig, input_dim: usize, req: &Request) -> Result<Plan, Refusal> {
    matrix::check_request(req)?;
    matrix::int_args(&IntWidths::of(cfg, input_dim, (req.batch, req.seq_len)).args())?;
    if req.path == Path::Train && req.gemm == GemmMode::Deterministic {
        let rows = req.batch.saturating_mul(req.seq_len);
        matrix::triad_spans(rows, &linears(cfg, input_dim))?;
    }
    let scan = match req.path {
        Path::Decode => Scan::Sequential,
        Path::Prefill | Path::Train => Scan::of(req.scan.use_parallel(req.seq_len, cfg.d_state)),
    };
    Ok(Plan { scan })
}
