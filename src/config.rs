//! Configuration shared by the model families: the family itself, the
//! kind of the block norms and the scan mode of the GPU recurrence.

// Mamba-1's configuration lived here up to 0.7. Code written then still
// finds it here for one release, its home is `mamba1::config`.
#[doc(hidden)]
pub use crate::mamba1::config::MambaConfig;

/// The model families this crate runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Mamba (Gu and Dao, 2023), the selective scan.
    Mamba1,
    /// Mamba-2 (Dao and Gu, 2024), the state space duality.
    Mamba2,
    /// Mamba-3 SISO (Lahoti et al., 2026), one input and one output channel
    /// per head.
    Mamba3,
    /// Mamba-3 MIMO, a rank of input and output channels per head.
    Mamba3Mimo,
}

/// The kind of the block norms: the norm before each mixer, the one before
/// an MLP and the final norm. mamba_ssm picks it with `rms_norm`; every
/// official checkpoint is RMSNorm. The norms inside the mixers stay RMS
/// either way, as in the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlockNorm {
    /// RMSNorm, `x / sqrt(mean(x^2) + eps) * weight` (`rms_norm = true`).
    #[default]
    Rms,
    /// LayerNorm with the biased variance, `(x - mean) / sqrt(var + eps) *
    /// weight + bias` (`rms_norm = false`, `nn.LayerNorm`).
    Layer,
}

impl BlockNorm {
    /// The kind mamba_ssm's `rms_norm` flag names.
    pub const fn from_rms_norm(rms_norm: bool) -> Self {
        if rms_norm { Self::Rms } else { Self::Layer }
    }

    /// The value of mamba_ssm's `rms_norm` flag for this kind.
    pub const fn rms_norm(self) -> bool {
        matches!(self, Self::Rms)
    }

    /// The length of a norm's bias over `width` features: `width` for
    /// LayerNorm, 0 for RMSNorm.
    pub const fn bias_len(self, width: usize) -> usize {
        match self {
            Self::Rms => 0,
            Self::Layer => width,
        }
    }

    /// The statistics a norm keeps per row for its backward: the RMS for
    /// RMSNorm; the standard deviation and the mean, in that order, for
    /// LayerNorm.
    pub const fn stats_per_row(self) -> usize {
        match self {
            Self::Rms => 1,
            Self::Layer => 2,
        }
    }

    /// The lengths of the final norm's scale and bias over `width`
    /// features, none for a backbone built without the final norm (a
    /// segment of a stack, `final_norm = false`).
    pub const fn final_lens(self, width: usize, final_norm: bool) -> (usize, usize) {
        if final_norm {
            (width, self.bias_len(width))
        } else {
            (0, 0)
        }
    }
}

/// Scan mode for GPU SSM recurrence.
///
/// Controls whether the SSM recurrence uses sequential O(T) scan
/// or parallel prefix scan O(T) work / O(log T) depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScanMode {
    /// Sequential O(T) scan — zero overhead, optimal for short sequences.
    Sequential,
    /// Parallel prefix scan — warp shuffle + shared memory, optimal for long T.
    Parallel,
    /// Auto-select: Sequential for T <= 64, Parallel for T > 64
    /// (PARALLEL_SCAN_THRESHOLD, measured on RTX 6000 Ada).
    #[default]
    Auto,
}

impl ScanMode {
    /// Auto threshold: sequential at or below, parallel above. Single source
    /// of truth for the GPU dispatch (re-exported as
    /// `mamba1::gpu::forward::PARALLEL_SCAN_THRESHOLD`).
    pub const PARALLEL_SCAN_THRESHOLD: usize = 64;

    /// Resolve the effective scan choice for a sequence length and state dim.
    ///
    /// An explicit mode is honoured at every size: the sequential kernels
    /// are sized to the state at compile time. `Auto` above 64 states takes
    /// the parallel scan at any length, the faster path there, and the
    /// route existing `Auto` runs took.
    pub fn use_parallel(self, seq_len: usize, d_state: usize) -> bool {
        match self {
            Self::Sequential => false,
            Self::Parallel => true,
            Self::Auto => d_state > 64 || seq_len > Self::PARALLEL_SCAN_THRESHOLD,
        }
    }

    /// Parse `auto`, `sequential` or `parallel`; ASCII whitespace around
    /// the name is ignored.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim_ascii() {
            "auto" => Ok(Self::Auto),
            "sequential" => Ok(Self::Sequential),
            "parallel" => Ok(Self::Parallel),
            _ => Err(format!(
                "{value:?} is not a scan mode (use auto, sequential, or parallel)"
            )),
        }
    }

    /// Resolve `Auto` for a shape. Delegates to [`Self::use_parallel`] so the
    /// two entry points can never disagree.
    pub fn resolve(self, seq_len: usize, d_state: usize) -> Self {
        if self.use_parallel(seq_len, d_state) {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }
}

/// The bounds every family's `validate` keeps a config inside. A config
/// comes from files (safetensors metadata, an HF `config.json`) as readily
/// as from code, so every size it implies must be exact: the widths stay
/// under these bounds, and the layers' parameters fit one allocation,
/// which keeps every width and tensor size a family derives from its
/// config inside a `usize`.
pub(crate) mod envelope {
    /// The widest `d_model` and `d_intermediate` a config names.
    pub const MAX_WIDTH: usize = 1 << 20;
    /// The deepest stack a config names.
    pub const MAX_LAYERS: usize = 4096;

    /// Refuses `n_layers` layers of `per_layer` parameters (`None` when the
    /// count passed u128) whose f32 bytes exceed the largest allocation,
    /// `isize::MAX` bytes.
    pub fn check_param_bytes(per_layer: Option<u128>, n_layers: usize) -> Result<(), String> {
        per_layer
            .and_then(|p| p.checked_mul(n_layers as u128))
            .and_then(|p| p.checked_mul(4))
            .filter(|&bytes| bytes <= isize::MAX as u128)
            .map(|_| ())
            .ok_or_else(|| {
                "the layers' parameters exceed the largest allocation (isize::MAX bytes of f32)"
                    .into()
            })
    }

    /// The sum of the products of `terms`, exact in u128; `None` past it.
    pub fn sum_of_products(terms: &[&[u128]]) -> Option<u128> {
        terms.iter().try_fold(0u128, |sum, factors| {
            let product = factors.iter().try_fold(1u128, |p, &f| p.checked_mul(f))?;
            sum.checked_add(product)
        })
    }
}

/// The sweep every family's config tests run: each field at sizes around
/// the bounds and past them, alone and in pairs, through a check that
/// validates the config and, when it passes, derives every width.
#[cfg(test)]
pub(crate) mod envelope_sweep {
    pub const SIZES: [usize; 20] = [
        0,
        1,
        2,
        3,
        7,
        64,
        256,
        257,
        1024,
        1025,
        1 << 20,
        (1 << 20) + 1,
        1 << 31,
        1 << 32,
        1 << 40,
        1 << 61,
        1 << 62,
        1 << 63,
        usize::MAX / 2,
        usize::MAX,
    ];

    pub const FLOATS: [f32; 10] = [
        0.0,
        -0.0,
        1e-30,
        1e-4,
        0.5,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    /// Runs `check` on `base` with every field of `fields` at every size,
    /// alone and in every pair.
    pub fn sweep<C: Copy>(base: C, fields: &[fn(&mut C, usize)], check: impl Fn(C)) {
        for (i, set_a) in fields.iter().enumerate() {
            for a in SIZES {
                let mut one = base;
                set_a(&mut one, a);
                check(one);
                for set_b in &fields[i + 1..] {
                    for b in SIZES {
                        let mut two = one;
                        set_b(&mut two, b);
                        check(two);
                    }
                }
            }
        }
    }
}
