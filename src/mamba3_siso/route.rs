//! Which Mamba-3 SISO shapes the GPU kernels can run.
//!
//! [`Mamba3Config::validate`] holds the model envelope: every shape the
//! official implementation accepts, all of which the CPU runs. The GPU
//! kernels have narrower limits of their own:
//! - The warp-tier head kernels (step, burn-in, sequential backward,
//!   chunked backward) broadcast across a head with warp shuffles of width
//!   `headdim`, so a head must fit one warp and be a power of two wide.
//!   Wider heads take the block tier, which so far serves inference only.
//! - The chunked forward's fallback tiles run `2 * headdim` threads.
//! - The chunked backward stages its tiles in shared memory.
//! - The fused output norm runs one thread per channel and reduces each
//!   head with a tree that assumes a power-of-two head.
//!
//! Every GPU entry point checks its shape here first, so a shape the
//! kernels cannot run is refused with the reason instead of computing
//! wrong values.
//!
//! The module needs no CUDA feature: a CPU caller can ask whether a model
//! will run on the GPU before building anything.

use super::config::Mamba3Config;

/// What the GPU runs the model for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePurpose {
    /// Decode steps and prompt prefill.
    Inference,
    /// The training forward and backward.
    Training,
}

/// The shape facts the GPU kernels depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteShape {
    pub headdim: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub is_outproj_norm: bool,
    /// The training forward and backward run the chunked scan. Ignored for
    /// inference.
    pub chunked_training: bool,
}

impl RouteShape {
    pub fn of(cfg: &Mamba3Config) -> Self {
        Self {
            headdim: cfg.headdim,
            d_state: cfg.d_state,
            d_inner: cfg.d_inner(),
            is_outproj_norm: cfg.is_outproj_norm,
            chunked_training: cfg.train_use_parallel_scan(),
        }
    }
}

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

/// The widest channel count the fused gated output norm serves: it runs
/// one thread per channel in a single block.
pub const FUSED_OUT_NORM_D_INNER_MAX: usize = 1024;

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

/// Checks that the GPU kernels can run `shape` for `purpose`.
pub fn check(shape: &RouteShape, purpose: RoutePurpose) -> Result<(), String> {
    let tier = head_tier(shape.headdim)?;
    if purpose == RoutePurpose::Training {
        require_warp(
            shape.headdim,
            "Mamba-3 GPU training (its burn-in and backward kernels have no block tier yet)",
        )?;
    }
    if shape.is_outproj_norm && shape.d_inner > FUSED_OUT_NORM_D_INNER_MAX {
        return Err(format!(
            "is_outproj_norm on the GPU needs d_inner ({}) <= {FUSED_OUT_NORM_D_INNER_MAX}: \
             the fused gated norm runs one thread per channel in a single block",
            shape.d_inner
        ));
    }
    if shape.is_outproj_norm && tier == HeadTier::Block && !shape.headdim.is_power_of_two() {
        return Err(format!(
            "is_outproj_norm on the GPU needs a power-of-two headdim, got {}: the fused \
             gated norm's per-head reduction tree assumes one",
            shape.headdim
        ));
    }
    if purpose == RoutePurpose::Training && shape.chunked_training {
        let smem = chunked_backward_smem_bytes(shape.headdim, shape.d_state);
        if smem > CHUNKED_BACKWARD_SMEM_MAX {
            return Err(format!(
                "headdim {} with d_state {} needs {} KB of shared memory in the chunked \
                 backward, over the 99 KB opt-in budget; train it with the sequential \
                 scan or wait for the state-tiled backward",
                shape.headdim,
                shape.d_state,
                smem / 1024
            ));
        }
    }
    Ok(())
}

/// [`check`] for a whole config.
pub fn check_config(cfg: &Mamba3Config, purpose: RoutePurpose) -> Result<(), String> {
    check(&RouteShape::of(cfg), purpose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScanMode;

    /// The official checkpoint shape runs inference on the block tier,
    /// and its GPU training is refused until the block tier has training
    /// kernels, with the reason.
    #[test]
    fn official_shape_runs_inference_on_the_block_tier() {
        let cfg = Mamba3Config {
            d_model: 768,
            headdim: 64,
            d_state: 128,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        assert_eq!(head_tier(cfg.headdim), Ok(HeadTier::Block));
        check_config(&cfg, RoutePurpose::Inference).unwrap();
        let err = check_config(&cfg, RoutePurpose::Training).unwrap_err();
        assert!(err.contains("headdim"), "{err}");
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

    /// Shapes whose chunked-backward tiles exceed the opt-in budget are
    /// refused for chunked training, naming the resource, and accepted for
    /// inference and for sequential training.
    #[test]
    fn chunked_backward_smem_budget_rejected_loudly() {
        let cfg = Mamba3Config {
            d_state: 256,
            headdim: 32,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        let err = check_config(&cfg, RoutePurpose::Training).unwrap_err();
        assert!(err.contains("shared memory"), "{err}");
        check_config(&cfg, RoutePurpose::Inference).unwrap();
        let sequential = Mamba3Config {
            scan_mode: ScanMode::Sequential,
            ..cfg
        };
        check_config(&sequential, RoutePurpose::Training).unwrap();
    }

    /// The bound is the kernel's tile with its padded rows. Headdim 16
    /// trains chunked up to d_state 142 and is refused at 143, whose padded
    /// tile is past 99 KB; the unpadded bound admitted it and the backward
    /// refused it at the first step.
    #[test]
    fn chunked_backward_bound_counts_the_padded_rows() {
        let at = |d_state| Mamba3Config {
            d_state,
            headdim: 16,
            ..Mamba3Config::default()
        };
        assert_eq!(chunked_backward_smem_bytes(16, 142), 101_120);
        assert_eq!(chunked_backward_smem_bytes(16, 143), 101_760);
        check_config(&at(142), RoutePurpose::Training).unwrap();
        let err = check_config(&at(143), RoutePurpose::Training).unwrap_err();
        assert!(err.contains("shared memory"), "{err}");
    }

    /// The fused gated norm cannot serve more than 1024 channels; without
    /// the norm, wide models run.
    #[test]
    fn gated_norm_d_inner_cap_rejected_loudly() {
        let big_gated = Mamba3Config {
            d_model: 1024,
            expand: 2,
            headdim: 32,
            is_outproj_norm: true,
            ..Mamba3Config::default()
        };
        big_gated.validate().unwrap();
        let err = check_config(&big_gated, RoutePurpose::Inference).unwrap_err();
        assert!(err.contains("is_outproj_norm"), "{err}");
        let big_silu = Mamba3Config {
            is_outproj_norm: false,
            ..big_gated
        };
        check_config(&big_silu, RoutePurpose::Inference).unwrap();
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
