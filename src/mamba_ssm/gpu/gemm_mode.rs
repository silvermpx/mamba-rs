use cudarc::cublas::sys::{cublasComputeType_t, cublasMath_t};

/// Selects the GEMM implementation and cuBLAS numeric policy for a GPU context.
///
/// [`GemmMode::Deterministic`] is the default. It selects the context's own
/// fixed-reduction-order GEMM kernels. The two cuBLAS modes select vendor
/// GEMMs with either ordinary or pedantic f32 compute. The storage precision
/// ([`WeightDtype`](crate::mamba_ssm::gpu::dtype::WeightDtype)) is the other
/// setting and is independent of the mode.
///
/// Use [`crate::mamba_ssm::gpu::context::GpuCtx::new_with_mode`] when the mode
/// is known at construction, or
/// [`crate::mamba_ssm::gpu::context::GpuCtx::set_gemm_mode`] to change an
/// existing usable context.
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

    pub(crate) const fn batch_invariant(self) -> bool {
        matches!(self, Self::Deterministic)
    }

    pub(crate) const fn fast_gemm(self) -> bool {
        matches!(self, Self::CublasFast)
    }

    pub(crate) const fn tf32(self) -> bool {
        matches!(self, Self::CublasFast)
    }

    pub(crate) const fn cublas_math(self) -> cublasMath_t {
        match self {
            Self::Deterministic | Self::CublasPedantic => cublasMath_t::CUBLAS_PEDANTIC_MATH,
            Self::CublasFast => cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        }
    }

    pub(crate) fn vendor_compute(self) -> Result<cublasComputeType_t, String> {
        match self {
            Self::Deterministic => {
                Err("deterministic GEMM mode cannot cross a cuBLAS GemmEx dispatch boundary".into())
            }
            Self::CublasFast => Ok(cublasComputeType_t::CUBLAS_COMPUTE_32F),
            Self::CublasPedantic => Ok(cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC),
        }
    }
}

pub(crate) trait MathModeBackend {
    type Mode: Copy + std::fmt::Debug + Eq;

    fn query(&mut self) -> Result<Self::Mode, String>;
    fn update(&mut self, mode: Self::Mode) -> Result<(), String>;
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MathTransitionError {
    Recoverable(String),
    Unusable(String),
}

pub(crate) fn change_math_mode<B: MathModeBackend>(
    backend: &mut B,
    target: B::Mode,
) -> Result<(), MathTransitionError> {
    let previous = backend.query().map_err(|error| {
        MathTransitionError::Recoverable(format!(
            "query current cuBLAS math mode before GEMM mode change: {error}"
        ))
    })?;
    if previous == target {
        return Ok(());
    }

    let update_error = match backend.update(target) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    if matches!(backend.query(), Ok(actual) if actual == previous) {
        return Err(MathTransitionError::Recoverable(update_error));
    }

    let restore_error = backend.update(previous).err();
    match backend.query() {
        Ok(actual) if actual == previous => Err(MathTransitionError::Recoverable(update_error)),
        Ok(actual) => Err(MathTransitionError::Unusable(format!(
            "cuBLAS math update failed ({update_error}); rollback could not be verified: \
             expected {previous:?}, found {actual:?}{}",
            restore_error
                .as_deref()
                .map(|error| format!("; restore call failed ({error})"))
                .unwrap_or_default()
        ))),
        Err(verify_error) => Err(MathTransitionError::Unusable(format!(
            "cuBLAS math update failed ({update_error}); rollback could not be verified \
             because the final query failed ({verify_error}){}",
            restore_error
                .as_deref()
                .map(|error| format!("; restore call failed ({error})"))
                .unwrap_or_default()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeMath {
        Pedantic,
        Fast,
    }

    enum Operation {
        Query(Result<FakeMath, &'static str>),
        Update(FakeMath, Result<(), &'static str>),
    }

    struct ScriptedBackend {
        operations: VecDeque<Operation>,
    }

    impl ScriptedBackend {
        fn new(operations: impl IntoIterator<Item = Operation>) -> Self {
            Self {
                operations: operations.into_iter().collect(),
            }
        }

        fn finish(self) {
            assert!(self.operations.is_empty(), "unused backend operations");
        }
    }

    impl MathModeBackend for ScriptedBackend {
        type Mode = FakeMath;

        fn query(&mut self) -> Result<Self::Mode, String> {
            match self.operations.pop_front().expect("scripted query") {
                Operation::Query(result) => result.map_err(str::to_owned),
                Operation::Update(_, _) => panic!("expected scripted query"),
            }
        }

        fn update(&mut self, mode: Self::Mode) -> Result<(), String> {
            match self.operations.pop_front().expect("scripted update") {
                Operation::Update(expected, result) => {
                    assert_eq!(mode, expected);
                    result.map_err(str::to_owned)
                }
                Operation::Query(_) => panic!("expected scripted update"),
            }
        }
    }

    #[test]
    fn math_transition_publishes_only_after_successful_update() {
        let mut backend = ScriptedBackend::new([
            Operation::Query(Ok(FakeMath::Pedantic)),
            Operation::Update(FakeMath::Fast, Ok(())),
        ]);
        assert_eq!(change_math_mode(&mut backend, FakeMath::Fast), Ok(()));
        backend.finish();
    }

    #[test]
    fn math_transition_stops_on_initial_query_failure() {
        let mut backend = ScriptedBackend::new([Operation::Query(Err("query failed"))]);
        assert!(matches!(
            change_math_mode(&mut backend, FakeMath::Fast),
            Err(MathTransitionError::Recoverable(error)) if error.contains("query failed")
        ));
        backend.finish();
    }

    #[test]
    fn failed_math_update_with_unchanged_handle_is_recoverable() {
        let mut backend = ScriptedBackend::new([
            Operation::Query(Ok(FakeMath::Pedantic)),
            Operation::Update(FakeMath::Fast, Err("update failed")),
            Operation::Query(Ok(FakeMath::Pedantic)),
        ]);
        assert_eq!(
            change_math_mode(&mut backend, FakeMath::Fast),
            Err(MathTransitionError::Recoverable("update failed".into()))
        );
        backend.finish();
    }

    #[test]
    fn failed_math_update_restores_a_changed_handle() {
        let mut backend = ScriptedBackend::new([
            Operation::Query(Ok(FakeMath::Pedantic)),
            Operation::Update(FakeMath::Fast, Err("update failed")),
            Operation::Query(Ok(FakeMath::Fast)),
            Operation::Update(FakeMath::Pedantic, Ok(())),
            Operation::Query(Ok(FakeMath::Pedantic)),
        ]);
        assert_eq!(
            change_math_mode(&mut backend, FakeMath::Fast),
            Err(MathTransitionError::Recoverable("update failed".into()))
        );
        backend.finish();
    }

    #[test]
    fn unverified_math_rollback_marks_the_transition_unusable() {
        let mut backend = ScriptedBackend::new([
            Operation::Query(Ok(FakeMath::Pedantic)),
            Operation::Update(FakeMath::Fast, Err("update failed")),
            Operation::Query(Ok(FakeMath::Fast)),
            Operation::Update(FakeMath::Pedantic, Err("restore failed")),
            Operation::Query(Ok(FakeMath::Fast)),
        ]);
        assert!(matches!(
            change_math_mode(&mut backend, FakeMath::Fast),
            Err(MathTransitionError::Unusable(error))
                if error.contains("update failed") && error.contains("restore failed")
        ));
        backend.finish();
    }

    #[test]
    fn vendor_compute_mapping_is_literal_for_all_modes() {
        assert!(GemmMode::Deterministic.vendor_compute().is_err());
        assert_eq!(
            GemmMode::CublasFast.vendor_compute().unwrap(),
            cublasComputeType_t::CUBLAS_COMPUTE_32F
        );
        assert_eq!(
            GemmMode::CublasPedantic.vendor_compute().unwrap(),
            cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
        );
    }
}
