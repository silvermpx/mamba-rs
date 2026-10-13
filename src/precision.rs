//! The precision a model is stored and multiplied in, and the GEMM policy,
//! as plain values every build names: the GPU engines (feature `cuda`) map
//! them to their cuBLAS settings, and the matrix census plans with them
//! without a device.

/// Weight storage precision for the GPU engines (f32 / tf32 / bf16 / f16).
///
/// Every kernel accumulates in f32. Only bulk linear weights (in_proj,
/// out_proj, x_proj, dt_proj, embed, lm_head) and activations between GEMMs
/// use reduced precision; norms, biases, a_log and D stay f32. `Tf32` is f32
/// storage whose deterministic GEMM products run on the tensor cores in
/// TF32 where a measured kernel exists.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, Hash)]
pub enum WeightDtype {
    /// IEEE 754 single precision. Default. Largest memory footprint and
    /// the safe choice for math sensitive to precision (full pre-training,
    /// long-horizon RL). Every product is exact f32: one fused
    /// multiply-add per step, in one fixed order.
    #[default]
    F32,
    /// f32 storage with deterministic TF32 products. In the deterministic
    /// GEMM mode each operand is rounded to TF32 once, the tensor cores
    /// multiply, and the sums accumulate in f32 in a fixed order; the same
    /// bits come back run to run, eager or graph. A model context (the
    /// backbones and the language model) runs TF32 for every f32 product of
    /// its layers; a trainer only where a measured TF32 kernel exists for
    /// the shape and the board, the exact f32 kernel elsewhere, as for the
    /// language model's tied head. It is not the vendor's Fast TF32. Under
    /// [`GemmMode::CublasFast`]
    /// the vendor's TF32 already applies to every f32 product; under
    /// `CublasPedantic` the products are pedantic f32. Memory, buffers and
    /// checkpoints are those of `F32`.
    Tf32,
    /// IEEE 754 half precision (`half::f16`). Tightest dynamic range —
    /// requires the dynamic loss scaler in training to avoid gradient
    /// underflow. ~2× memory savings and ~1.3× tok/s speedup at
    /// inference vs f32 on Ada / Hopper.
    F16,
    /// Brain float 16 (`half::bf16`). Same exponent range as f32 with
    /// reduced mantissa — no loss scaler needed in training. The
    /// recommended default for mixed-precision: ~2× memory, ~1.3×
    /// tok/s, no overflow regime to manage.
    Bf16,
}

impl WeightDtype {
    /// The dtype of the bytes in memory: `Tf32` is stored as `F32`.
    pub fn storage(self) -> Self {
        match self {
            Self::Tf32 => Self::F32,
            other => other,
        }
    }

    /// Byte size of one element.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::Tf32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// Whether the bytes in memory are f32 (`F32` and `Tf32`).
    pub fn is_f32(self) -> bool {
        matches!(self, Self::F32 | Self::Tf32)
    }

    pub fn is_half(self) -> bool {
        !self.is_f32()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Tf32 => "tf32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// Selects the GEMM implementation and cuBLAS numeric policy for a GPU context.
///
/// [`GemmMode::Deterministic`] is the default. It selects the context's own
/// fixed-reduction-order GEMM kernels. The two cuBLAS modes select vendor
/// GEMMs with either ordinary or pedantic f32 compute. The storage precision
/// ([`WeightDtype`]) is the other
/// setting and is independent of the mode.
///
/// A GPU context takes it at construction (`GpuCtx::new_with_mode`, feature
/// `cuda`) or changes to it later (`GpuCtx::set_gemm_mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GemmMode {
    /// Use the context's own deterministic GEMM kernels: the same bits run
    /// to run, eager or graph. `WeightDtype::F32` multiplies exactly;
    /// `WeightDtype::Tf32` takes the deterministic TF32 kernels where one is
    /// measured.
    #[default]
    Deterministic,
    /// Use cuBLAS with ordinary f32 GemmEx compute and TF32 handle math.
    ///
    /// This permits vendor TF32 and other cuBLAS optimizations for every f32
    /// product, whichever of `F32` and `Tf32` the weights were stored as.
    CublasFast,
    /// Use cuBLAS with pedantic f32 GemmEx compute and pedantic handle math.
    CublasPedantic,
}

impl GemmMode {
    /// Parse a value accepted by `MAMBA_RS_GEMM_MODE`.
    ///
    /// ASCII whitespace around the value is ignored. The remaining text must
    /// be exactly `deterministic`, `cublas-fast`, or `cublas-pedantic`; names
    /// are lowercase and case-sensitive.
    pub fn parse_env_value(value: &str) -> Result<Self, String> {
        match value.trim_ascii() {
            "deterministic" => Ok(Self::Deterministic),
            "cublas-fast" => Ok(Self::CublasFast),
            "cublas-pedantic" => Ok(Self::CublasPedantic),
            _ => Err(format!(
                "MAMBA_RS_GEMM_MODE={value:?} is not a recognized GEMM mode \
                 (use deterministic, cublas-fast, or cublas-pedantic)"
            )),
        }
    }

    /// Return the canonical environment spelling for this mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::CublasFast => "cublas-fast",
            Self::CublasPedantic => "cublas-pedantic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(WeightDtype::F32.size_bytes(), 4);
        assert_eq!(WeightDtype::Tf32.size_bytes(), 4);
        assert_eq!(WeightDtype::F16.size_bytes(), 2);
        assert_eq!(WeightDtype::Bf16.size_bytes(), 2);
    }

    #[test]
    fn is_half() {
        assert!(!WeightDtype::F32.is_half());
        assert!(!WeightDtype::Tf32.is_half());
        assert!(WeightDtype::F16.is_half());
        assert!(WeightDtype::Bf16.is_half());
    }

    #[test]
    fn tf32_is_stored_as_f32() {
        assert_eq!(WeightDtype::Tf32.storage(), WeightDtype::F32);
        assert_eq!(WeightDtype::Bf16.storage(), WeightDtype::Bf16);
        assert_eq!(WeightDtype::Tf32.as_str(), "tf32");
    }

    #[test]
    fn gemm_mode_names_round_trip() {
        for mode in [
            GemmMode::Deterministic,
            GemmMode::CublasFast,
            GemmMode::CublasPedantic,
        ] {
            assert_eq!(GemmMode::parse_env_value(mode.as_str()), Ok(mode));
        }
        assert!(GemmMode::parse_env_value("fast").is_err());
    }
}
