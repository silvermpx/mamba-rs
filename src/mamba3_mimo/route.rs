//! Which Mamba-3 MIMO work a request takes on the GPU, or why it does not.
//!
//! The module needs no CUDA feature: the census and a CPU caller ask it
//! before building anything, and the GPU constructors ask it first. Shared
//! memory never limits a validated shape on a supported part: the
//! recurrence fits one channel of the widest rank and state, and the
//! chunked kernels one token tile, within the 99 KB every part since Ampere
//! lets a block opt in to.

use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::matrix::{self, Path, Plan, Refusal, Request, Scan};

/// The values the MIMO kernels take as `int`s over windows of `seq_len`
/// tokens of `batch` sequences: the batch, the window, its rows, the widest
/// row, which bounds every column offset and width the kernels take, and
/// the heads of every sequence, one block each on a grid's x axis. Element
/// counts and offsets are 64-bit, so a window is bounded by memory alone
/// past these.
pub(crate) fn int_args(
    cfg: &Mamba3MimoConfig,
    input_dim: usize,
    (batch, seq_len): (usize, usize),
) -> [(&'static str, usize); 5] {
    let widest = [
        input_dim,
        cfg.in_proj_out_dim(),
        cfg.bc_width(),
        cfg.mimo_rank * cfg.d_inner(),
        2 * cfg.mlp_hidden(),
        cfg.d_model,
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

/// Whether and how the GPU runs `req` on a MIMO model of `cfg` fed rows
/// `input_dim` wide. `cfg` must pass [`Mamba3MimoConfig::validate`]; the
/// request's scan mode stands in for the config's. A prompt of one token
/// takes the recurrence, which is the decode step.
pub fn admit(cfg: &Mamba3MimoConfig, input_dim: usize, req: &Request) -> Result<Plan, Refusal> {
    matrix::check_request(req)?;
    matrix::int_args(&int_args(cfg, input_dim, (req.batch, req.seq_len)))?;
    let cfg = Mamba3MimoConfig {
        scan_mode: req.scan,
        ..*cfg
    };
    let scan = match req.path {
        Path::Decode => Scan::Sequential,
        Path::Prefill => Scan::of(req.seq_len > 1 && cfg.prefill_parallel(req.seq_len)),
        Path::Train => Scan::of(cfg.train_use_parallel_scan()),
    };
    Ok(Plan { scan })
}
