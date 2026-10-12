//! Which Mamba-3 SISO shapes the GPU kernels can run.
//!
//! [`Mamba3Config::validate`] holds the model envelope: every shape the
//! official implementation accepts, all of which the CPU runs. The GPU
//! kernels have narrower limits of their own:
//! - The warp-tier head kernels (step, burn-in, sequential backward,
//!   chunked backward) broadcast across a head with warp shuffles of width
//!   `headdim`, so a head must fit one warp and be a power of two wide.
//!   Wider heads take the block tier: the step and the sequential scan's
//!   training kernels have block-tier twins, and the chunked backward a
//!   tiled one.
//! - The chunked forward's fallback tiles run `2 * headdim` threads.
//!
//! Every GPU entry point checks its shape here first, so a shape the
//! kernels cannot run is refused with the reason instead of computing
//! wrong values.
//!
//! The module needs no CUDA feature: a CPU caller can ask whether a model
//! will run on the GPU before building anything.

use super::config::Mamba3Config;

/// The widest head the warp-shuffle kernels serve: one lane per head
/// element in a single warp.
pub const WARP_HEADDIM_MAX: usize = 32;

/// The widest head the GPU serves: the chunked forward's fallback tiles
/// launch two heads of `headdim` threads in one 1024-thread block.
pub const GPU_HEADDIM_MAX: usize = 512;

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
pub fn head_tier(headdim: usize) -> Result<HeadTier, String> {
    if headdim <= WARP_HEADDIM_MAX && headdim.is_power_of_two() {
        Ok(HeadTier::Warp)
    } else if (1..=GPU_HEADDIM_MAX).contains(&headdim) {
        Ok(HeadTier::Block)
    } else {
        Err(format!(
            "headdim {headdim} is wider than the {GPU_HEADDIM_MAX} the GPU kernels serve; \
             the CPU runs this shape"
        ))
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

/// Threads per block of a block-tier head kernel: the head rounded up to
/// whole warps.
pub fn block_tier_threads(headdim: usize) -> usize {
    headdim.next_multiple_of(32)
}

/// The shared memory of the block-tier sequential forward in bytes: the
/// step's coefficients and its K, previous K and Q rows.
pub fn sequential_forward_block_smem_bytes(d_state: usize) -> usize {
    (4 + 3 * d_state) * 4
}

/// The shared memory of the block-tier sequential backward in bytes: the
/// step's coefficients and rows, the carried previous-K gradient, and each
/// warp's sums of the 3 * d_state + 3 per-step gradients. The widest GPU
/// head with the largest state needs about 53 KB, inside the opt-in budget.
pub fn sequential_backward_block_smem_bytes(headdim: usize, d_state: usize) -> usize {
    let warps = headdim.div_ceil(32);
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

/// Checks that the GPU kernels can run a model whose heads are `headdim`
/// wide, for inference and for training alike.
pub fn check(headdim: usize) -> Result<(), String> {
    head_tier(headdim).map(|_| ())
}

/// [`check`] for a whole config.
pub fn check_config(cfg: &Mamba3Config) -> Result<(), String> {
    check(cfg.headdim)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The official checkpoint shape runs on the block tier, for inference
    /// and for training on either scan: the chunked one through the tiled
    /// backward, the sequential one on the block-tier kernels.
    #[test]
    fn official_shape_runs_on_the_block_tier() {
        let cfg = Mamba3Config {
            d_model: 768,
            headdim: 64,
            d_state: 128,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        assert_eq!(head_tier(cfg.headdim), Ok(HeadTier::Block));
        check_config(&cfg).unwrap();
        assert!(chunked_backward_tiled(cfg.headdim, cfg.d_state));
    }

    /// Every width the warp kernels serve stays on them; other widths take
    /// the block tier up to the GPU ceiling, and wider heads are refused.
    #[test]
    fn head_tier_keeps_every_warp_width_on_the_warp_tier() {
        for hd in 1..=1024usize {
            let tier = head_tier(hd);
            if hd <= 32 && hd.is_power_of_two() {
                assert_eq!(tier, Ok(HeadTier::Warp), "headdim {hd}");
            } else if hd <= GPU_HEADDIM_MAX {
                assert_eq!(tier, Ok(HeadTier::Block), "headdim {hd}");
            } else {
                assert!(tier.is_err(), "headdim {hd}");
            }
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
        check_config(&cfg).unwrap();
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

    /// The gated output norm serves every width: past 1024 channels and on
    /// heads that are no power of two.
    #[test]
    fn gated_norm_serves_every_width() {
        for (d_model, headdim) in [(1024usize, 32usize), (768, 64), (48, 48), (300, 100)] {
            let cfg = Mamba3Config {
                d_model,
                expand: 2,
                headdim,
                is_outproj_norm: true,
                ..Mamba3Config::default()
            };
            cfg.validate().unwrap();
            check_config(&cfg).unwrap();
        }
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
