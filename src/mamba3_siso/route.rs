//! Which Mamba-3 SISO shapes the GPU kernels can run.
//!
//! [`Mamba3Config::validate`] holds the model envelope: every shape the
//! official implementation accepts, all of which the CPU runs. The GPU
//! kernels have narrower limits of their own. The step, burn-in,
//! sequential-backward and chunked-backward kernels broadcast across a
//! head with warp shuffles of width `headdim`, so a head must fit one warp
//! and its width must be a power of two. The chunked backward stages its
//! tiles in shared memory, and the fused output norm runs one thread per
//! channel. Every GPU entry point checks its shape here first, so a shape
//! the kernels cannot run is refused with the reason instead of computing
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

/// The shared memory the chunked backward needs for a head, in bytes.
///
/// The kernel stages two chunk-by-d_state operand tiles (chunk size 64)
/// plus smaller pieces in dynamic shared memory, and the loader opts in to
/// the device's extended budget. 99 KB is the opt-in floor across every
/// supported part since Ampere; some parts offer more, and the floor keeps
/// one portable bound.
pub fn chunked_backward_smem_bytes(headdim: usize, d_state: usize) -> usize {
    let cs = 64usize;
    (2 * cs * d_state + 2 * cs * headdim + 4 * cs + 2 * headdim * d_state) * 4
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
    require_warp(
        shape.headdim,
        "the Mamba-3 GPU step, burn-in and backward kernels",
    )?;
    if shape.is_outproj_norm && shape.d_inner > FUSED_OUT_NORM_D_INNER_MAX {
        return Err(format!(
            "is_outproj_norm on the GPU needs d_inner ({}) <= {FUSED_OUT_NORM_D_INNER_MAX}: \
             the fused gated norm runs one thread per channel in a single block",
            shape.d_inner
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

    /// The official checkpoint shape is a valid model the GPU kernels do
    /// not run yet, and the refusal says why.
    #[test]
    fn official_shape_is_a_model_the_warp_kernels_refuse() {
        let cfg = Mamba3Config {
            d_model: 768,
            headdim: 64,
            d_state: 128,
            ..Mamba3Config::default()
        };
        cfg.validate().unwrap();
        let err = check_config(&cfg, RoutePurpose::Inference).unwrap_err();
        assert!(err.contains("headdim"), "{err}");
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
