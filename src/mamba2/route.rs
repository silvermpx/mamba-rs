//! Which Mamba-2 work a request takes on the GPU, or why it does not.
//!
//! The module needs no CUDA feature: the census and a CPU caller ask it
//! before building anything, and the GPU constructors ask it first. The
//! scan's shared memory never limits a validated shape on a supported part:
//! one channel of the largest state takes about 20 KB of the 99 KB every
//! part since Ampere lets a block opt in to.

use crate::mamba2::config::Mamba2Config;
use crate::matrix::{self, Path, Plan, Refusal, Request, Scan};
use crate::precision::GemmMode;

/// The values the Mamba-2 kernels take as `int`s over windows of `seq_len`
/// tokens of `batch` sequences: the batch, the window, its rows, the widest
/// row, which bounds every column offset and width the kernels take, and
/// the heads of every sequence, one block each on a grid's x axis. Element
/// counts and offsets are 64-bit, so a window is bounded by memory alone
/// past these.
pub(crate) fn int_args(
    cfg: &Mamba2Config,
    input_dim: usize,
    (batch, seq_len): (usize, usize),
) -> [(&'static str, usize); 5] {
    let widest = [
        input_dim,
        cfg.in_proj_out_dim(),
        cfg.d_inner(),
        cfg.d_model,
        2 * cfg.mlp_hidden(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    [
        ("batch", batch),
        ("seq_len", seq_len),
        ("rows", batch.saturating_mul(seq_len)),
        ("widest row", widest),
        // The state passes launch a block per (sequence, head) on x.
        ("batch*heads", batch.saturating_mul(cfg.nheads())),
    ]
}

/// The linear layers of a training step, `(name, input width, output
/// width)`; a model without an input projection takes rows `d_model` wide,
/// which no other layer's operands undercut.
fn linears(cfg: &Mamba2Config, input_dim: usize) -> [(&'static str, usize, usize); 5] {
    let (d, h) = (cfg.d_model, cfg.mlp_hidden());
    [
        ("input_proj", input_dim, d),
        ("in_proj", d, cfg.in_proj_out_dim()),
        ("out_proj", cfg.d_inner(), d),
        ("mlp fc1", d, 2 * h),
        ("mlp fc2", h, d),
    ]
}

/// Whether and how the GPU runs `req` on a Mamba-2 model of `cfg` fed rows
/// `input_dim` wide. `cfg` must pass [`Mamba2Config::validate`]; the
/// request's scan mode stands in for the config's.
pub fn admit(cfg: &Mamba2Config, input_dim: usize, req: &Request) -> Result<Plan, Refusal> {
    matrix::check_request(req)?;
    matrix::int_args(&int_args(cfg, input_dim, (req.batch, req.seq_len)))?;
    if req.path == Path::Train && req.gemm == GemmMode::Deterministic {
        let rows = req.batch.saturating_mul(req.seq_len);
        matrix::triad_spans(rows, &linears(cfg, input_dim))?;
    }
    let cfg = Mamba2Config {
        scan_mode: req.scan,
        ..*cfg
    };
    let scan = match req.path {
        Path::Decode => Scan::Sequential,
        Path::Prefill => Scan::of(cfg.prefill_parallel(req.seq_len)),
        Path::Train => Scan::of(cfg.train_use_parallel_scan()),
    };
    Ok(Plan { scan })
}
