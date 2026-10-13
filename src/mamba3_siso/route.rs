//! Which Mamba-3 SISO kernels a shape takes on the GPU.
//!
//! [`Mamba3Config::validate`](crate::mamba3_siso::config::Mamba3Config::validate)
//! holds the model envelope: every shape the official implementation
//! accepts. The GPU runs all of them, and this module picks the kernels:
//! - The warp-tier head kernels (step, burn-in, sequential backward,
//!   chunked backward) broadcast across a head with warp shuffles of width
//!   `headdim`, so they serve a head that fits one warp and is a power of
//!   two wide. Every other head takes the block tier: the step and the
//!   sequential scan's training kernels have block-tier twins, and the
//!   chunked backward a tiled one. A head wider than a block takes its
//!   elements in turns.
//! - The chunked backward's dQKV stage takes the tiled kernels wherever
//!   the warp kernel's tiles do not fit its shared-memory budget.
//!
//! The module needs no CUDA feature: a CPU caller can ask which kernels a
//! model takes before building anything.

use crate::mamba3_siso::config::Mamba3Config;
use crate::matrix::{self, Path, Plan, Refusal, Request, Scan};
use crate::precision::GemmMode;

/// The widest head the warp-shuffle kernels serve: one lane per head
/// element in a single warp.
pub const WARP_HEADDIM_MAX: usize = 32;

/// Which head kernels a width takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadTier {
    /// One warp per head, head-shared values broadcast by warp shuffle.
    Warp,
    /// One CTA per head, head-shared values staged in shared memory.
    Block,
}

/// The head tier a width takes on the GPU. Every width the warp kernels
/// serve stays on them, so no existing shape changes kernels.
pub fn head_tier(headdim: usize) -> HeadTier {
    if headdim <= WARP_HEADDIM_MAX && headdim.is_power_of_two() {
        HeadTier::Warp
    } else {
        HeadTier::Block
    }
}

/// The chunk the chunked scan and its backward walk. The GPU state's
/// `CHUNK_SIZE` is this constant, so the bound below and the kernels agree.
pub const CHUNKED_CHUNK_SIZE: usize = 64;

/// The shared memory the chunked backward needs for a head, in bytes: the
/// kernel's tile without its optional pair matrices. Two chunk-by-d_state
/// operand tiles and two chunk-by-headdim tiles, each row padded by one
/// float against bank conflicts, four per-step lanes and two head-state
/// tiles. The backward sizes its launch from this, and the loader opts in
/// to the device's extended budget. 99 KB is the opt-in floor across every
/// supported part since Ampere; some parts offer more, and the floor keeps
/// one portable bound.
pub fn chunked_backward_smem_bytes(headdim: usize, d_state: usize) -> usize {
    let cs = CHUNKED_CHUNK_SIZE;
    (2 * cs * (d_state + 1) + 2 * cs * (headdim + 1) + 4 * cs + 2 * headdim * d_state) * 4
}

/// The chunked backward's shared-memory ceiling.
pub const CHUNKED_BACKWARD_SMEM_MAX: usize = 99 * 1024;

/// The shared memory of the tiled dQKV kernel in bytes, for tiles `tile`
/// columns wide: nine per-step lanes, the three pair triangles of a chunk,
/// four chunk-by-tile operand tiles and two tile-by-tile state tiles, each
/// row padded by one float. It does not depend on headdim or d_state.
pub fn chunked_backward_tiled_smem_bytes(tile: usize) -> usize {
    let cs = CHUNKED_CHUNK_SIZE;
    let row = tile + 1;
    (9 * cs + 3 * (cs * (cs - 1) / 2) + 4 * cs * row + 2 * tile * row) * 4
}

/// Whether the chunked backward's dQKV stage takes the tiled kernels. The
/// warp kernel serves a head of one warp whose tiles fit its budget; every
/// other shape takes the tiled kernels, which fit any shape. Where both
/// serve they give the same bits.
pub fn chunked_backward_tiled(headdim: usize, d_state: usize) -> bool {
    !(headdim <= WARP_HEADDIM_MAX
        && headdim.is_power_of_two()
        && chunked_backward_smem_bytes(headdim, d_state) <= CHUNKED_BACKWARD_SMEM_MAX)
}

/// The most threads a block-tier sequential kernel runs, its launch bound.
/// A wider head runs in slices of this many elements.
pub const BLOCK_TIER_THREADS_MAX: usize = 512;

/// Threads per block of a block-tier sequential kernel: the head rounded up
/// to whole warps, at most [`BLOCK_TIER_THREADS_MAX`].
pub fn block_tier_threads(headdim: usize) -> usize {
    headdim.next_multiple_of(32).min(BLOCK_TIER_THREADS_MAX)
}

/// The shared memory of the block-tier sequential forward in bytes: the
/// step's coefficients and its K, previous K and Q rows.
pub fn sequential_forward_block_smem_bytes(d_state: usize) -> usize {
    (4 + 3 * d_state) * 4
}

/// The shared memory of the block-tier sequential backward in bytes: the
/// step's coefficients and rows, the carried previous-K gradient, and each
/// warp's sums of the 3 * d_state + 3 per-step gradients. The widest block
/// with the largest state needs about 53 KB, inside the opt-in budget.
pub fn sequential_backward_block_smem_bytes(headdim: usize, d_state: usize) -> usize {
    let warps = block_tier_threads(headdim) / 32;
    (4 + 4 * d_state + (3 * d_state + 3) * warps) * 4
}

/// Refuses a head the warp-shuffle kernels cannot run, naming the kernel.
pub fn require_warp(headdim: usize, kernel: &str) -> Result<(), String> {
    if headdim <= WARP_HEADDIM_MAX && headdim.is_power_of_two() {
        Ok(())
    } else {
        Err(format!(
            "{kernel} needs headdim to be a power of two at most {WARP_HEADDIM_MAX} \
             (one warp-shuffle lane per head element); got {headdim}. The CPU runs \
             this shape."
        ))
    }
}

/// The widths a SISO launch over windows of `seq_len` tokens of `batch`
/// sequences takes as `int`s.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntWidths {
    pub batch: usize,
    pub seq_len: usize,
    pub ngroups: usize,
    pub d_model: usize,
    pub input_dim: usize,
    pub in_proj_dim: usize,
    pub nheads: usize,
    pub d_state: usize,
    pub d_inner: usize,
}

impl IntWidths {
    pub(crate) fn of(
        cfg: &Mamba3Config,
        input_dim: usize,
        (batch, seq_len): (usize, usize),
    ) -> Self {
        Self {
            batch,
            seq_len,
            ngroups: cfg.ngroups,
            d_model: cfg.d_model,
            input_dim,
            in_proj_dim: cfg.in_proj_out_dim(),
            nheads: cfg.nheads(),
            d_state: cfg.d_state,
            d_inner: cfg.d_inner(),
        }
    }

    /// The values the window kernels take as `int`s: the batch, the
    /// window, the rows they make, the B and C norm rows of every group,
    /// and the row widths.
    pub(crate) fn args(&self) -> [(&'static str, usize); 9] {
        let rows = self.batch.saturating_mul(self.seq_len);
        [
            ("batch", self.batch),
            ("seq_len", self.seq_len),
            ("batch*seq_len", rows),
            ("batch*seq_len*ngroups", rows.saturating_mul(self.ngroups)),
            ("in_proj_dim", self.in_proj_dim),
            ("d_model", self.d_model),
            ("input width", self.input_dim),
            ("nheads*d_state", self.nheads.saturating_mul(self.d_state)),
            (
                "nheads*headdim*d_state",
                self.d_inner.saturating_mul(self.d_state),
            ),
        ]
    }

    /// The values the decode step takes as `int`s: the sequences, and the
    /// B and C norm rows of every group.
    pub(crate) fn step_args(&self) -> [(&'static str, usize); 2] {
        [
            ("batch", self.batch),
            ("batch*ngroups", self.batch.saturating_mul(self.ngroups)),
        ]
    }
}

/// The linear layers of a training step, `(name, input width, output
/// width)`; a model without an input projection takes rows `d_model` wide,
/// which no other layer's operands undercut.
fn linears(cfg: &Mamba3Config, input_dim: usize) -> [(&'static str, usize, usize); 5] {
    let (d, h) = (cfg.d_model, cfg.mlp_hidden());
    [
        ("input_proj", input_dim, d),
        ("in_proj", d, cfg.in_proj_out_dim()),
        ("out_proj", cfg.d_inner(), d),
        ("mlp fc1", d, 2 * h),
        ("mlp fc2", h, d),
    ]
}

/// Whether and how the GPU runs `req` on a SISO model of `cfg` fed rows
/// `input_dim` wide. `cfg` must pass [`Mamba3Config::validate`]; the
/// request's scan mode stands in for the config's.
pub fn admit(cfg: &Mamba3Config, input_dim: usize, req: &Request) -> Result<Plan, Refusal> {
    matrix::check_request(req)?;
    let widths = IntWidths::of(cfg, input_dim, (req.batch, req.seq_len));
    let cfg = Mamba3Config {
        scan_mode: req.scan,
        ..*cfg
    };
    let scan = match req.path {
        Path::Decode => {
            matrix::int_args(&widths.step_args())?;
            return Ok(Plan {
                scan: Scan::Sequential,
            });
        }
        Path::Prefill => Scan::of(cfg.prefill_use_parallel_scan()),
        Path::Train => Scan::of(cfg.train_use_parallel_scan()),
    };
    matrix::int_args(&widths.args())?;
    if req.path == Path::Train && req.gemm == GemmMode::Deterministic {
        let rows = req.batch.saturating_mul(req.seq_len);
        matrix::triad_spans(rows, &linears(&cfg, input_dim))?;
    }
    Ok(Plan { scan })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The official checkpoint shape runs on the block tier, its chunked
    /// training through the tiled backward.
    #[test]
    fn official_shape_runs_on_the_block_tier() {
        let cfg = Mamba3Config {
            d_model: 768,
            headdim: 64,
            d_state: 128,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        assert_eq!(head_tier(cfg.headdim), HeadTier::Block);
        assert!(chunked_backward_tiled(cfg.headdim, cfg.d_state));
    }

    /// Every width the warp kernels serve stays on them, and every other
    /// width takes the block tier, however wide.
    #[test]
    fn head_tier_keeps_every_warp_width_on_the_warp_tier() {
        for hd in (1..=1024usize).chain([1056, 2048, 4096]) {
            let want = if hd <= 32 && hd.is_power_of_two() {
                HeadTier::Warp
            } else {
                HeadTier::Block
            };
            assert_eq!(head_tier(hd), want, "headdim {hd}");
        }
    }

    /// The block-tier sequential kernels take a head of up to 512 in one
    /// slice and a wider one in slices of 512, and their backward fits the
    /// opt-in budget at the largest state whatever the head.
    #[test]
    fn block_tier_slices_stay_inside_the_launch_bound() {
        assert_eq!(block_tier_threads(48), 64);
        assert_eq!(block_tier_threads(512), 512);
        assert_eq!(block_tier_threads(513), 512);
        assert_eq!(block_tier_threads(4096), 512);
        for hd in [100usize, 512, 640, 4096] {
            let smem = sequential_backward_block_smem_bytes(hd, 256);
            assert!(smem <= 99 * 1024, "headdim {hd}: {smem} B");
        }
    }

    /// A shape whose warp-kernel tiles exceed the opt-in budget runs, and its
    /// chunked training takes the tiled backward.
    #[test]
    fn chunked_training_takes_the_tiled_backward_past_the_warp_budget() {
        let cfg = Mamba3Config {
            d_state: 256,
            headdim: 32,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        assert!(chunked_backward_tiled(cfg.headdim, cfg.d_state));
    }

    /// The warp kernel's bound is its tile with its padded rows. Headdim 16
    /// keeps the warp kernel up to d_state 142 and takes the tiled one at
    /// 143, whose padded tile is past 99 KB.
    #[test]
    fn chunked_backward_bound_counts_the_padded_rows() {
        assert_eq!(chunked_backward_smem_bytes(16, 142), 101_120);
        assert_eq!(chunked_backward_smem_bytes(16, 143), 101_760);
        assert!(!chunked_backward_tiled(16, 142));
        assert!(chunked_backward_tiled(16, 143));
        assert!(chunked_backward_tiled(48, 16));
        assert!(chunked_backward_tiled(64, 16));
        assert!(!chunked_backward_tiled(32, 16));
    }

    /// Every head width the warp kernels serve passes, and nothing else.
    #[test]
    fn require_warp_is_the_old_headdim_predicate() {
        for hd in 0..=1024usize {
            let old = hd <= 32 && hd.is_power_of_two();
            assert_eq!(require_warp(hd, "k").is_ok(), old, "headdim {hd}");
        }
    }
}
