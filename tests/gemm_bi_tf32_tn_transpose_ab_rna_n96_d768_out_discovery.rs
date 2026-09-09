//! Test-only Ada TF32 TN d768-out joint A/B pre-RNA discovery.
//! Production kernels and dispatch remain untouched.

#[path = "support/triad_tf32_tn_transpose_ab_rna_n96_source.rs"]
mod candidate_source;
#[path = "support/triad_tn_transpose_n96_source.rs"]
mod raw_retained_source;
#[path = "support/triad_tf32_tn_transpose_rna_n96_source.rs"]
#[allow(dead_code)]
mod retained_source;

const FIXED_N96_SOURCE: &str = include_str!("../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

const FAST_THRESHOLD: f64 = 0.99;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BracketOrder {
    Abba,
    Baab,
}

impl BracketOrder {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    const fn name(self) -> &'static str {
        match self {
            Self::Abba => "ABBA",
            Self::Baab => "BAAB",
        }
    }

    const fn candidate_slots(self) -> [bool; 4] {
        match self {
            Self::Abba => [false, true, true, false],
            Self::Baab => [true, false, false, true],
        }
    }
}

fn candidate_over_comparator_ratio(
    order: BracketOrder,
    observations: [f64; 4],
) -> Result<f64, String> {
    if observations
        .iter()
        .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err(format!("invalid bracket observations: {observations:?}"));
    }
    let candidate_slots = order.candidate_slots();
    let mut candidate = 0.0;
    let mut comparator = 0.0;
    for (index, sample) in observations.into_iter().enumerate() {
        if candidate_slots[index] {
            candidate += sample;
        } else {
            comparator += sample;
        }
    }
    Ok(candidate / comparator)
}

fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .flatten()
            .all(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < threshold)
}

#[test]
fn rna_oracle_rounds_finite_ties_away_from_zero() {
    for (input, expected) in [
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x3f80_0fff, 0x3f80_0000),
        (0x3f80_1000, 0x3f80_2000),
        (0x3f80_1001, 0x3f80_2000),
        (0xbf80_1000, 0xbf80_2000),
        (0x3f80_1fff, 0x3f80_2000),
        (0x3f80_2000, 0x3f80_2000),
        (0x7f7f_ffff, 0x7f80_0000),
    ] {
        assert_eq!(candidate_source::tf32_rna_bits(input), expected);
    }
}

#[test]
fn scratch_oracle_is_an_independent_rna_transpose_with_zero_padding() {
    let input = [
        0x3f80_1000,
        0xbf80_0fff,
        0x4000_1001,
        0xc000_2000,
        0x0000_0000,
        0x8000_0000,
    ];
    let expected = [
        0x3f80_2000,
        0xc000_2000,
        0,
        0,
        0xbf80_0000,
        0x0000_0000,
        0,
        0,
        0x4000_2000,
        0x8000_0000,
        0,
        0,
    ];
    assert!(candidate_source::validate_rna_transposed_words(&input, 2, 3, 4, &expected).is_ok());
    let mut wrong = expected;
    wrong[4] ^= 0x2000;
    assert!(candidate_source::validate_rna_transposed_words(&input, 2, 3, 4, &wrong).is_err());

    let b = [0x0000_0000, 0x8000_0000, 0x3f80_1000, 0xbf80_0fff];
    let b_expected = [0x0000_0000, 0x8000_0000, 0x3f80_2000, 0xbf80_0000];
    assert!(candidate_source::validate_rna_words(&b, &b_expected).is_ok());
    let mut wrong_b = b_expected;
    wrong_b[2] ^= 0x2000;
    assert!(candidate_source::validate_rna_words(&b, &wrong_b).is_err());
}

#[test]
fn adapter_moves_b_rna_into_the_existing_a_rna_preprocess_and_is_reversible() {
    let raw_retained = raw_retained_source::candidate_source(FIXED_N96_SOURCE).unwrap();
    let retained = retained_source::compose_candidate_source(&raw_retained).unwrap();
    let candidate = candidate_source::compose_candidate_source(&retained).unwrap();
    assert_eq!(
        candidate
            .matches(candidate_source::CANDIDATE_GEMM_SYMBOL)
            .count(),
        1
    );
    assert_eq!(
        candidate
            .matches(candidate_source::CANDIDATE_PREPROCESS_SYMBOL)
            .count(),
        1
    );
    assert!(!candidate.contains(candidate_source::RETAINED_GEMM_SYMBOL));
    assert!(!candidate.contains(candidate_source::RETAINED_PREPROCESS_SYMBOL));
    assert_eq!(
        candidate.matches("fragments.a[m_atom][0] = raw0;").count(),
        1
    );
    assert_eq!(
        candidate.matches("fragments.a[m_atom][1] = raw1;").count(),
        1
    );
    assert_eq!(
        candidate.matches("fragments.a[m_atom][2] = raw2;").count(),
        1
    );
    assert_eq!(
        candidate.matches("fragments.a[m_atom][3] = raw3;").count(),
        1
    );
    assert_eq!(
        candidate
            .matches("tf32n96_round(__float_as_uint(b_step")
            .count(),
        0
    );
    assert!(!candidate.contains("unsigned b[3][2];"));
    assert!(candidate.contains("void tf32n96_mma_pre_rounded_b("));
    assert_eq!(candidate.matches("unsigned b_fragment[2] = {").count(), 3);
    assert!(!candidate.contains("acc[m_atom][n_atom], fragments.a[m_atom], b_fragment"));
    for n_atom in 0..3 {
        for word in 0..2 {
            assert_eq!(
                candidate
                    .matches(&format!(
                        "__float_as_uint(b_step[offsets.b[{n_atom}][{word}]])"
                    ))
                    .count(),
                1
            );
        }
    }
    assert!(candidate.contains("tf32n96_round(tile[(int)threadIdx.x][(int)threadIdx.y + offset])"));
    assert!(candidate.contains("output_b[linear] = tf32n96_round(input_b[linear]);"));
    assert_eq!(
        candidate_source::restore_retained_source(&candidate).unwrap(),
        retained
    );
}

#[test]
fn strict_gate_requires_four_finite_positive_p50_p95_pairs_below_point99() {
    assert!(all_strata_below(&[[0.98, 0.989]; 4], FAST_THRESHOLD));
    assert!(!all_strata_below(&[[0.98, 0.99]; 4], FAST_THRESHOLD));
    assert!(!all_strata_below(&[[0.98, 0.989]; 3], FAST_THRESHOLD));
    assert!(!all_strata_below(
        &[
            [0.98, 0.989],
            [0.98, f64::NAN],
            [0.98, 0.989],
            [0.98, 0.989]
        ],
        FAST_THRESHOLD,
    ));
}

#[test]
fn bracket_ratio_assigns_candidate_slots_without_external_helper_api() {
    assert_eq!(
        candidate_over_comparator_ratio(BracketOrder::Abba, [10.0, 4.0, 6.0, 10.0]).unwrap(),
        0.5
    );
    assert_eq!(
        candidate_over_comparator_ratio(BracketOrder::Baab, [4.0, 10.0, 10.0, 6.0]).unwrap(),
        0.5
    );
    assert!(candidate_over_comparator_ratio(BracketOrder::Abba, [1.0, 0.0, 1.0, 1.0]).is_err());
}
#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;

#[cfg(feature = "cuda")]
mod cuda_suite {
    use super::*;
    use cudarc::{
        cublas::{result as blas_result, sys as blas},
        driver::{
            CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig,
            PushKernelArg, sys,
        },
    };
    use mamba_rs::mamba_ssm::gpu::{
        buffers::GpuBuffer,
        context::{BiGemmFamily, F32TriadPolicy, GpuCtx},
        device::GpuDevice,
        dtype::WeightDtype,
        graph_capture::capture_into_graph,
        kernels::cuda_include_paths,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        ffi::{CStr, c_void},
        sync::Arc,
    };

    const ENV: &str = "MAMBA_TRIAD_TF32_TN_TRANSPOSE_AB_RNA_N96_D768_OUT_DISCOVERY";
    const SHARED: usize = 86_016;
    const OPS: usize = 1;
    const WARMUPS: usize = 8;
    const GUARD: usize = 64;
    const ALIGNMENT: u64 = 256;
    const GUARD_BITS: u32 = 0x7fc0_b196;
    const POISON_BITS: u32 = 0x7fc0_c196;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Shape {
        m: usize,
        k: usize,
        n: usize,
    }

    impl Shape {
        fn grid(self) -> Result<(u32, u32, u32), String> {
            let blocks = self
                .k
                .div_ceil(128)
                .checked_mul(self.n.div_ceil(96))
                .ok_or("N96 grid overflows usize")?;
            Ok((
                u32::try_from(blocks).map_err(|_| "N96 grid exceeds u32")?,
                1,
                1,
            ))
        }

        fn config(self) -> Result<LaunchConfig, String> {
            Ok(LaunchConfig {
                grid_dim: self.grid()?,
                block_dim: (256, 1, 1),
                shared_mem_bytes: SHARED as u32,
            })
        }

        fn transpose_stride(self) -> Result<usize, String> {
            raw_retained_source::padded_stride(self.m)
        }

        fn transpose_config(self) -> Result<LaunchConfig, String> {
            let stride = self.transpose_stride()?;
            Ok(LaunchConfig {
                grid_dim: (
                    u32::try_from(self.k.max(1))
                        .map_err(|_| "transpose columns exceed u32")?
                        .div_ceil(32),
                    u32::try_from(stride.max(1))
                        .map_err(|_| "transpose stride exceeds u32")?
                        .div_ceil(32),
                    1,
                ),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            })
        }
    }

    #[derive(Clone, Copy)]
    struct Case {
        label: &'static str,
        shape: Shape,
    }

    const D768_OUT: Case = Case {
        label: "d768_out_proj",
        shape: Shape {
            m: 2_048,
            k: 1_536,
            n: 768,
        },
    };
    const FULL_TILE: Case = Case {
        label: "full_tile_tn_32x128x96",
        shape: Shape {
            m: 32,
            k: 128,
            n: 96,
        },
    };
    const TAIL: Case = Case {
        label: "tail_tn_129x65x100",
        shape: Shape {
            m: 129,
            k: 65,
            n: 100,
        },
    };
    const K0: Case = Case {
        label: "reduction0_tn_0x65x100",
        shape: Shape {
            m: 0,
            k: 65,
            n: 100,
        },
    };

    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Params {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }
    unsafe impl DeviceRepr for Params {}

    impl Params {
        fn new(shape: Shape) -> Result<Self, String> {
            Ok(Self {
                alpha: 1.0,
                beta: 1.0,
                m: i32::try_from(shape.k).map_err(|_| "physical M exceeds i32")?,
                k: i32::try_from(shape.m).map_err(|_| "physical K exceeds i32")?,
                n: i32::try_from(shape.n).map_err(|_| "N exceeds i32")?,
                lda: i32::try_from(shape.transpose_stride()?).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(shape.n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(shape.n).map_err(|_| "ldc exceeds i32")?,
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(C)]
    struct TransposeParams {
        rows: i32,
        columns: i32,
        output_stride: i32,
    }
    unsafe impl DeviceRepr for TransposeParams {}

    impl TransposeParams {
        fn new(shape: Shape) -> Result<Self, String> {
            Ok(Self {
                rows: i32::try_from(shape.m).map_err(|_| "transpose rows exceed i32")?,
                columns: i32::try_from(shape.k).map_err(|_| "transpose columns exceed i32")?,
                output_stride: i32::try_from(shape.transpose_stride()?)
                    .map_err(|_| "transpose stride exceeds i32")?,
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(C)]
    struct CandidatePreprocessParams {
        rows: i32,
        columns: i32,
        output_stride: i32,
        b_elements: i32,
    }
    unsafe impl DeviceRepr for CandidatePreprocessParams {}

    impl CandidatePreprocessParams {
        fn new(shape: Shape) -> Result<Self, String> {
            let b_elements = shape
                .m
                .checked_mul(shape.n)
                .ok_or("B preprocess extent overflows usize")?;
            Ok(Self {
                rows: i32::try_from(shape.m).map_err(|_| "preprocess A rows exceed i32")?,
                columns: i32::try_from(shape.k).map_err(|_| "preprocess A columns exceed i32")?,
                output_stride: i32::try_from(shape.transpose_stride()?)
                    .map_err(|_| "preprocess A stride exceeds i32")?,
                b_elements: i32::try_from(b_elements)
                    .map_err(|_| "preprocess B elements exceed i32")?,
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Retained,
        Fast,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Candidate => "pre_rna_ab_n96",
                Self::Retained => "pre_rna_a_n96",
                Self::Fast => "cublas_fast_tf32",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    struct GuardedF32 {
        buffer: GpuBuffer,
        baseline: Vec<f32>,
        len: usize,
        label: &'static str,
    }

    impl GuardedF32 {
        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            label: &'static str,
        ) -> Result<Self, String> {
            let len = active.len();
            let mut baseline = vec![f32::from_bits(GUARD_BITS); GUARD + len + GUARD];
            baseline[GUARD..GUARD + len].copy_from_slice(&active);
            let buffer = GpuBuffer::from_cpu(stream, &baseline)?;
            let value = Self {
                buffer,
                baseline,
                len,
                label,
            };
            if value.buffer.cached_ptr() % ALIGNMENT != 0 || value.ptr(stream) % ALIGNMENT != 0 {
                return Err(format!(
                    "{} base/logical pointer is not 256B aligned",
                    value.label
                ));
            }
            Ok(value)
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, GUARD)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.baseline)
        }

        fn bits(&self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            if values[..GUARD]
                .iter()
                .chain(&values[GUARD + self.len..])
                .any(|value| value.to_bits() != GUARD_BITS)
            {
                return Err(format!("{} red zone changed", self.label));
            }
            Ok(values[GUARD..GUARD + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn unchanged(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            let values = self.buffer.to_cpu(stream)?;
            if values
                .iter()
                .zip(&self.baseline)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("{} input or guard changed", self.label));
            }
            Ok(())
        }

        fn baseline_bits(&self) -> Vec<u32> {
            self.baseline[GUARD..GUARD + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }
    }

    struct Fixture {
        a: GuardedF32,
        b: GuardedF32,
        candidate_scratch: GuardedF32,
        candidate_b_scratch: GuardedF32,
        retained_scratch: GuardedF32,
        candidate: GuardedF32,
        retained: GuardedF32,
        fast: GuardedF32,
        exceptional: bool,
    }

    impl Fixture {
        fn new(runtime: &Runtime, case: Case, exceptional: bool) -> Result<Self, String> {
            let shape = case.shape;
            let mut a = full_mantissa::finite_full_mantissa_values(shape.m * shape.k, 0xb196_a001);
            let mut b = full_mantissa::finite_full_mantissa_values(shape.m * shape.n, 0xb196_b002);
            if exceptional && shape.m > 0 {
                a.fill(0.0);
                b.fill(0.0);
                for (column, bits) in [
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0001,
                    0x8000_0001,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7f80_0001,
                    0x7fc1_2345,
                    0x7fa1_2345,
                ]
                .into_iter()
                .enumerate()
                {
                    if column < shape.k {
                        a[column] = f32::from_bits(bits);
                    }
                }
                if shape.n > 0 {
                    b[0] = 1.0;
                }
            }
            let output = full_mantissa::finite_full_mantissa_values(shape.k * shape.n, 0xb196_c003);
            let scratch_len = shape
                .k
                .checked_mul(shape.transpose_stride()?)
                .ok_or("transpose scratch extent overflows usize")?;
            let scratch = vec![f32::from_bits(POISON_BITS); scratch_len];
            let b_scratch = vec![f32::from_bits(POISON_BITS); b.len()];
            Ok(Self {
                a: GuardedF32::new(&runtime.ctx.stream, a, "A")?,
                b: GuardedF32::new(&runtime.ctx.stream, b, "B")?,
                candidate_scratch: GuardedF32::new(
                    &runtime.ctx.stream,
                    scratch.clone(),
                    "candidate transposed A",
                )?,
                candidate_b_scratch: GuardedF32::new(
                    &runtime.ctx.stream,
                    b_scratch,
                    "candidate pre-rounded B",
                )?,
                retained_scratch: GuardedF32::new(
                    &runtime.ctx.stream,
                    scratch,
                    "retained transposed A",
                )?,
                candidate: GuardedF32::new(&runtime.ctx.stream, output.clone(), "candidate C")?,
                retained: GuardedF32::new(&runtime.ctx.stream, output.clone(), "retained C")?,
                fast: GuardedF32::new(&runtime.ctx.stream, output, "Fast C")?,
                exceptional,
            })
        }

        fn output(&self, arm: Arm) -> &GuardedF32 {
            match arm {
                Arm::Candidate => &self.candidate,
                Arm::Retained => &self.retained,
                Arm::Fast => &self.fast,
            }
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedF32 {
            match arm {
                Arm::Candidate => &mut self.candidate,
                Arm::Retained => &mut self.retained,
                Arm::Fast => &mut self.fast,
            }
        }

        fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
            self.a.unchanged(&runtime.ctx.stream)?;
            self.b.unchanged(&runtime.ctx.stream)
        }

        fn scratch(&self, arm: Arm) -> Result<&GuardedF32, String> {
            match arm {
                Arm::Candidate => Ok(&self.candidate_scratch),
                Arm::Retained => Ok(&self.retained_scratch),
                Arm::Fast => Err("Fast has no transpose scratch".into()),
            }
        }

        fn scratch_mut(&mut self, arm: Arm) -> Result<&mut GuardedF32, String> {
            match arm {
                Arm::Candidate => Ok(&mut self.candidate_scratch),
                Arm::Retained => Ok(&mut self.retained_scratch),
                Arm::Fast => Err("Fast has no transpose scratch".into()),
            }
        }

        fn candidate_b_scratch(&self) -> &GuardedF32 {
            &self.candidate_b_scratch
        }

        fn candidate_b_scratch_mut(&mut self) -> &mut GuardedF32 {
            &mut self.candidate_b_scratch
        }

        fn reset_workspaces(&mut self, stream: &Arc<CudaStream>, arm: Arm) -> Result<(), String> {
            if arm != Arm::Fast {
                self.scratch_mut(arm)?.reset(stream)?;
            }
            if arm == Arm::Candidate {
                self.candidate_b_scratch_mut().reset(stream)?;
            }
            Ok(())
        }
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _retained_module: Arc<CudaModule>,
        _candidate_module: Arc<CudaModule>,
        retained: CudaFunction,
        candidate: CudaFunction,
        retained_transpose: CudaFunction,
        candidate_transpose: CudaFunction,
        retained_source_sha: String,
        candidate_source_sha: String,
        retained_ptx_sha: String,
        candidate_ptx_sha: String,
    }

    fn module_source(body: &str) -> String {
        let prelude = include_str!("../kernels/_typed_prelude.cuh");
        let common = include_str!("../kernels/gemm_bi_fixed/common.cuh")
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n");
        let tf32 = include_str!("../kernels/gemm_bi_fixed/tf32.cu");
        [prelude, &common, tf32, body].join("\n")
    }

    fn compile_module(
        device: &GpuDevice,
        source: String,
        symbol: &str,
    ) -> Result<(Arc<CudaModule>, CudaFunction, String, String), String> {
        let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec![
                    "--fmad=true".into(),
                    "--extra-device-vectorization".into(),
                    "-DNDEBUG".into(),
                ],
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("compile {symbol}: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load {symbol} module: {error:?}"))?;
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SHARED as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        Ok((module, function, source_sha, ptx_sha))
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var(ENV).as_deref() != Ok("1") {
            return Err(format!("set {ENV}=1"));
        }
        if cfg!(debug_assertions) {
            return Err("TN joint A/B pre-RNA N96 timing requires --release".into());
        }
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err(format!(
                "requires CC8.9/142 SM, found {:?}/{}",
                device.compute_capability,
                device.multiprocessor_count()
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        let mut math = blas::cublasMath_t::CUBLAS_DEFAULT_MATH;
        let mut pointer = blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
        unsafe {
            if blas::cublasGetMathMode(*ctx.blas.handle(), &mut math)
                != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || blas::cublasGetPointerMode_v2(*ctx.blas.handle(), &mut pointer)
                    != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || math == blas::cublasMath_t::CUBLAS_PEDANTIC_MATH
                || pointer != blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST
            {
                return Err("cuBLAS Fast handle mode is not supported".into());
            }
        }
        let compiler = ctx.kernels.compiler_identity();
        if compiler.nvrtc_version != (13, 2)
            || compiler.target.as_str() != "sm_89"
            || !compiler.nvrtc_library_known
        {
            return Err(format!("wrong N96 discovery compiler: {compiler:?}"));
        }
        let raw_retained_body = raw_retained_source::candidate_source(FIXED_N96_SOURCE)?;
        let retained_body = retained_source::compose_candidate_source(&raw_retained_body)?;
        let candidate_body = candidate_source::compose_candidate_source(&retained_body)?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) = compile_module(
            &device,
            module_source(&retained_body),
            candidate_source::RETAINED_GEMM_SYMBOL,
        )?;
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_module(
                &device,
                module_source(&candidate_body),
                candidate_source::CANDIDATE_GEMM_SYMBOL,
            )?;
        let retained_transpose = retained_module
            .load_function(candidate_source::RETAINED_PREPROCESS_SYMBOL)
            .map_err(|error| format!("load retained transpose: {error:?}"))?;
        let candidate_transpose = candidate_module
            .load_function(candidate_source::CANDIDATE_PREPROCESS_SYMBOL)
            .map_err(|error| format!("load candidate transpose: {error:?}"))?;
        let runtime = Runtime {
            _device: device,
            ctx,
            _retained_module: retained_module,
            _candidate_module: candidate_module,
            retained,
            candidate,
            retained_transpose,
            candidate_transpose,
            retained_source_sha,
            candidate_source_sha,
            retained_ptx_sha,
            candidate_ptx_sha,
        };
        resource_gate(
            &runtime.retained,
            candidate_source::RETAINED_GEMM_SYMBOL,
            128,
        )?;
        resource_gate(
            &runtime.candidate,
            candidate_source::CANDIDATE_GEMM_SYMBOL,
            128,
        )?;
        transpose_resource_gate(
            &runtime.retained_transpose,
            candidate_source::RETAINED_PREPROCESS_SYMBOL,
        )?;
        transpose_resource_gate(
            &runtime.candidate_transpose,
            candidate_source::CANDIDATE_PREPROCESS_SYMBOL,
        )?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutIdentityV1",
                "retained_symbol":candidate_source::RETAINED_GEMM_SYMBOL,
                "candidate_symbol":candidate_source::CANDIDATE_GEMM_SYMBOL,
                "retained_transform_symbol":candidate_source::RETAINED_PREPROCESS_SYMBOL,
                "candidate_transform_symbol":candidate_source::CANDIDATE_PREPROCESS_SYMBOL,
                "retained_source_sha":runtime.retained_source_sha,
                "candidate_source_sha":runtime.candidate_source_sha,
                "retained_ptx_sha":runtime.retained_ptx_sha,
                "candidate_ptx_sha":runtime.candidate_ptx_sha,
                "conversion":"add_half_ulp_tf32_v1",
                "change":"preprocess_A_and_B_RNA_once_before_GEMM",
            })
        );
        Ok(runtime)
    }

    fn resource_gate(
        function: &CudaFunction,
        symbol: &str,
        register_limit: i32,
    ) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|e| format!("{symbol} regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("{symbol} local: {e:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("{symbol} shared: {e:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|e| format!("{symbol} max threads: {e:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|e| format!("{symbol} max dynamic: {e:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(256, SHARED, None)
            .map_err(|e| format!("{symbol} occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutResourceV1","symbol":symbol,
                "threads":256,"registers":registers,"local_bytes":local,
                "static_shared_bytes":static_shared,"dynamic_shared_bytes":SHARED,
                "max_threads_per_block":max_threads,"occupancy":occupancy,
            })
        );
        if registers <= 0
            || registers > register_limit
            || local != 0
            || static_shared != 0
            || max_threads < 256
            || max_dynamic < SHARED as i32
            || occupancy != 1
        {
            return Err(format!(
                "{symbol} resource gate failed: regs={registers} local={local} static={static_shared} max_dynamic={max_dynamic} occupancy={occupancy}"
            ));
        }
        Ok(())
    }

    fn transpose_resource_gate(function: &CudaFunction, symbol: &str) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|e| format!("{symbol} regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("{symbol} local: {e:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("{symbol} shared: {e:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|e| format!("{symbol} max threads: {e:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutTransformResourceV1",
                "symbol":symbol,"threads":256,"registers":registers,"local_bytes":local,
                "static_shared_bytes":static_shared,"dynamic_shared_bytes":0,"max_threads_per_block":max_threads,
            })
        );
        if registers <= 0
            || registers > 64
            || local != 0
            || static_shared != 4_224
            || max_threads < 256
        {
            return Err(format!(
                "{symbol} transform resource gate failed: regs={registers} local={local} static={static_shared} max_threads={max_threads}"
            ));
        }
        Ok(())
    }

    fn launch(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
    ) -> Result<(), String> {
        let shape = case.shape;
        let output = fixture.output(arm).ptr(&runtime.ctx.stream);
        let (a, b) = if shape.m == 0 {
            (0, 0)
        } else {
            (
                fixture.a.ptr(&runtime.ctx.stream),
                fixture.b.ptr(&runtime.ctx.stream),
            )
        };
        match arm {
            Arm::Candidate => {
                let a_scratch = fixture.scratch(arm)?.ptr(&runtime.ctx.stream);
                let b_scratch = fixture.candidate_b_scratch().ptr(&runtime.ctx.stream);
                let preprocess_params = CandidatePreprocessParams::new(shape)?;
                let mut preprocess_builder = runtime
                    .ctx
                    .stream
                    .launch_builder(&runtime.candidate_transpose);
                preprocess_builder
                    .arg(&a)
                    .arg(&a_scratch)
                    .arg(&b)
                    .arg(&b_scratch)
                    .arg(&preprocess_params);
                unsafe { preprocess_builder.launch(shape.transpose_config()?) }
                    .map_err(|e| format!("launch {} preprocess: {e:?}", arm.name()))?;
                let bias = 0_u64;
                let params = Params::new(shape)?;
                let mut builder = runtime.ctx.stream.launch_builder(&runtime.candidate);
                builder
                    .arg(&output)
                    .arg(&a_scratch)
                    .arg(&b_scratch)
                    .arg(&bias)
                    .arg(&params);
                unsafe { builder.launch(shape.config()?) }
                    .map(|_| ())
                    .map_err(|e| format!("launch {}: {e:?}", arm.name()))
            }
            Arm::Retained => {
                let scratch = fixture.scratch(arm)?.ptr(&runtime.ctx.stream);
                let transpose_params = TransposeParams::new(shape)?;
                let mut transpose_builder = runtime
                    .ctx
                    .stream
                    .launch_builder(&runtime.retained_transpose);
                transpose_builder
                    .arg(&a)
                    .arg(&scratch)
                    .arg(&transpose_params);
                unsafe { transpose_builder.launch(shape.transpose_config()?) }
                    .map_err(|e| format!("launch {} preprocess: {e:?}", arm.name()))?;
                let bias = 0_u64;
                let params = Params::new(shape)?;
                let mut builder = runtime.ctx.stream.launch_builder(&runtime.retained);
                builder
                    .arg(&output)
                    .arg(&scratch)
                    .arg(&b)
                    .arg(&bias)
                    .arg(&params);
                unsafe { builder.launch(shape.config()?) }
                    .map(|_| ())
                    .map_err(|e| format!("launch {}: {e:?}", arm.name()))
            }
            Arm::Fast => {
                if shape.m == 0 {
                    return Err("Fast comparator excludes zero reduction".into());
                }
                let alpha = 1.0_f32;
                let beta = 1.0_f32;
                let dtype = WeightDtype::F32.cuda_data_type();
                unsafe {
                    blas_result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        blas::cublasOperation_t::CUBLAS_OP_T,
                        shape.n as i32,
                        shape.k as i32,
                        shape.m as i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        b as *const c_void,
                        dtype,
                        shape.n as i32,
                        a as *const c_void,
                        dtype,
                        shape.k as i32,
                        (&beta as *const f32).cast::<c_void>(),
                        output as *mut c_void,
                        dtype,
                        shape.n as i32,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|e| format!("cuBLAS Fast TF32: {e:?}"))
            }
        }
    }

    fn capture(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        launch(runtime, fixture, case, arm)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} warmup sync: {e:?}", arm.name()))?;
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        fixture.reset_workspaces(&runtime.ctx.stream, arm)?;
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, case, arm)) }
    }

    fn graph_identity(graph: &CudaGraph, case: Case, arm: Arm) -> Result<(), String> {
        unsafe {
            let mut count = 0;
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count == 0
            {
                return Err(format!("{} graph is empty/unqueryable", arm.name()));
            }
            if arm == Arm::Fast {
                println!(
                    "{}",
                    json!({"schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutGraphV1","arm":arm.name(),"nodes":count,"abi":"opaque","timing":"whole_graph"})
                );
                return Ok(());
            }
            if count != 2 {
                return Err(format!("{} graph has {count} nodes", arm.name()));
            }
            let mut nodes = [std::ptr::null_mut(); 2];
            if sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
            {
                return Err(format!("{} graph nodes query failed", arm.name()));
            }
            let expected_gemm = if arm == Arm::Candidate {
                candidate_source::CANDIDATE_GEMM_SYMBOL
            } else {
                candidate_source::RETAINED_GEMM_SYMBOL
            };
            let expected_transpose = if arm == Arm::Candidate {
                candidate_source::CANDIDATE_PREPROCESS_SYMBOL
            } else {
                candidate_source::RETAINED_PREPROCESS_SYMBOL
            };
            let mut saw_gemm = false;
            let mut saw_transpose = false;
            for node in nodes {
                let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                if sys::cuGraphKernelNodeGetParams_v2(node, &mut params)
                    != sys::CUresult::CUDA_SUCCESS
                {
                    return Err(format!("{} graph params unavailable", arm.name()));
                }
                let mut name_ptr = std::ptr::null();
                if sys::cuFuncGetName(&mut name_ptr, params.func) != sys::CUresult::CUDA_SUCCESS
                    || name_ptr.is_null()
                {
                    return Err(format!("{} graph symbol unavailable", arm.name()));
                }
                let symbol = CStr::from_ptr(name_ptr).to_string_lossy();
                if symbol == expected_gemm {
                    let config = case.shape.config()?;
                    if (params.gridDimX, params.gridDimY, params.gridDimZ) != config.grid_dim
                        || (params.blockDimX, params.blockDimY, params.blockDimZ)
                            != config.block_dim
                        || params.sharedMemBytes != config.shared_mem_bytes
                    {
                        return Err(format!("{} GEMM graph geometry changed", arm.name()));
                    }
                    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
                        .into_iter()
                        .enumerate()
                    {
                        let (mut offset, mut size) = (0, 0);
                        if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                            != sys::CUresult::CUDA_SUCCESS
                            || (offset, size) != expected
                        {
                            return Err(format!(
                                "{} GEMM ABI parameter {index} changed: {:?}",
                                arm.name(),
                                (offset, size)
                            ));
                        }
                    }
                    saw_gemm = true;
                } else if symbol == expected_transpose {
                    let config = case.shape.transpose_config()?;
                    if (params.gridDimX, params.gridDimY, params.gridDimZ) != config.grid_dim
                        || (params.blockDimX, params.blockDimY, params.blockDimZ)
                            != config.block_dim
                        || params.sharedMemBytes != 0
                    {
                        return Err(format!("{} transpose graph geometry changed", arm.name()));
                    }
                    let expected_abi: &[(usize, usize)] = if arm == Arm::Candidate {
                        &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 16)]
                    } else {
                        &[(0, 8), (8, 8), (16, 12)]
                    };
                    for (index, &(expected_offset, expected_size)) in
                        expected_abi.iter().enumerate()
                    {
                        let (mut offset, mut size) = (0, 0);
                        if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                            != sys::CUresult::CUDA_SUCCESS
                            || (offset, size) != (expected_offset, expected_size)
                        {
                            return Err(format!(
                                "{} transpose ABI parameter {index} changed: {:?}",
                                arm.name(),
                                (offset, size)
                            ));
                        }
                    }
                    saw_transpose = true;
                } else {
                    return Err(format!(
                        "{} graph has unexpected symbol {symbol}",
                        arm.name()
                    ));
                }
            }
            if !saw_gemm || !saw_transpose {
                return Err(format!("{} graph lost required node", arm.name()));
            }
            println!(
                "{}",
                json!({"schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutGraphV1","arm":arm.name(),"nodes":2,"gemm_symbol":expected_gemm,"transpose_symbol":expected_transpose,"gemm_grid":case.shape.grid()?,"transpose_grid":case.shape.transpose_config()?.grid_dim,"timing":"whole_two_node_graph"})
            );
        }
        Ok(())
    }

    fn output_after(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
        graph: Option<&CudaGraph>,
        repeats: usize,
    ) -> Result<Vec<u32>, String> {
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        fixture.reset_workspaces(&runtime.ctx.stream, arm)?;
        for _ in 0..repeats {
            if let Some(graph) = graph {
                graph
                    .launch()
                    .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?;
            } else {
                launch(runtime, fixture, case, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} sync: {e:?}", arm.name()))?;
        let bits = fixture.output(arm).bits(&runtime.ctx.stream)?;
        fixture.validate_inputs(runtime)?;
        if arm != Arm::Fast && !fixture.exceptional {
            let input = fixture.a.baseline_bits();
            let scratch = fixture.scratch(arm)?.bits(&runtime.ctx.stream)?;
            let stride = case.shape.transpose_stride()?;
            if arm == Arm::Candidate {
                candidate_source::validate_rna_transposed_words(
                    &input,
                    case.shape.m,
                    case.shape.k,
                    stride,
                    &scratch,
                )?;
                candidate_source::validate_rna_words(
                    &fixture.b.baseline_bits(),
                    &fixture.candidate_b_scratch().bits(&runtime.ctx.stream)?,
                )?;
            } else {
                retained_source::validate_rna_transposed_words(
                    &input,
                    case.shape.m,
                    case.shape.k,
                    stride,
                    &scratch,
                )?;
            }
        }
        Ok(bits)
    }

    fn check_exact_case(runtime: &Runtime, case: Case, exceptional: bool) -> Result<(), String> {
        let mut fixture = Fixture::new(runtime, case, exceptional)?;
        let golden = output_after(runtime, &mut fixture, case, Arm::Retained, None, 1)?;
        let candidate_graph = capture(runtime, &mut fixture, case, Arm::Candidate)?;
        let retained_graph = capture(runtime, &mut fixture, case, Arm::Retained)?;
        graph_identity(&candidate_graph, case, Arm::Candidate)?;
        graph_identity(&retained_graph, case, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (Arm::Candidate, &candidate_graph),
                (Arm::Retained, &retained_graph),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, case, arm, path, 1)? != golden {
                        return Err(format!(
                            "{} {} repeat {repeat} changed exact bits",
                            case.label,
                            arm.name()
                        ));
                    }
                }
            }
        }
        println!(
            "{}",
            json!({"schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutBitsV1","case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],"exceptional":exceptional,"candidate_retained_exact":true,"eager_repeats":2,"graph_repeats":2,"guards":true})
        );
        Ok(())
    }

    struct Prepared {
        fixture: Fixture,
        candidate_graph: CudaGraph,
        retained_graph: CudaGraph,
        fast_graph: Option<CudaGraph>,
        exact_bits: Vec<u32>,
        fast_bits: Option<Vec<u32>>,
    }

    fn prepare_target(runtime: &Runtime, case: Case) -> Result<Prepared, String> {
        let mut fixture = Fixture::new(runtime, case, false)?;
        let exact_bits = output_after(runtime, &mut fixture, case, Arm::Retained, None, 1)?;
        if output_after(runtime, &mut fixture, case, Arm::Candidate, None, 1)? != exact_bits {
            return Err(format!(
                "{} candidate differs from retained N96",
                case.label
            ));
        }
        let candidate_graph = capture(runtime, &mut fixture, case, Arm::Candidate)?;
        let retained_graph = capture(runtime, &mut fixture, case, Arm::Retained)?;
        graph_identity(&candidate_graph, case, Arm::Candidate)?;
        graph_identity(&retained_graph, case, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (Arm::Candidate, &candidate_graph),
                (Arm::Retained, &retained_graph),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, case, arm, path, 1)? != exact_bits {
                        return Err(format!(
                            "{} {} {} repeat {repeat} changed bits",
                            case.label,
                            arm.name(),
                            if path.is_some() { "graph" } else { "eager" }
                        ));
                    }
                }
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutTargetBitsV1",
                "case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],
                "candidate_retained_exact":true,"fast_deferred_until_retained_win":true,
                "eager_repeats":2,"graph_repeats":2,"guards":true,
            })
        );
        Ok(Prepared {
            fixture,
            candidate_graph,
            retained_graph,
            fast_graph: None,
            exact_bits,
            fast_bits: None,
        })
    }

    fn prepare_fast(runtime: &Runtime, case: Case, prepared: &mut Prepared) -> Result<(), String> {
        let fast_bits = output_after(runtime, &mut prepared.fixture, case, Arm::Fast, None, 1)?;
        if fast_bits.is_empty()
            || fast_bits.iter().all(|word| word & 0x7fff_ffff == 0)
            || fast_bits
                .iter()
                .any(|word| !f32::from_bits(*word).is_finite())
        {
            return Err(format!("{} Fast comparator output invalid", case.label));
        }
        let fast_graph = capture(runtime, &mut prepared.fixture, case, Arm::Fast)?;
        graph_identity(&fast_graph, case, Arm::Fast)?;
        for repeat in 0..2 {
            for path in [None, Some(&fast_graph)] {
                if output_after(runtime, &mut prepared.fixture, case, Arm::Fast, path, 1)?
                    != fast_bits
                {
                    return Err(format!(
                        "{} Fast {} repeat {repeat} changed bits",
                        case.label,
                        if path.is_some() { "graph" } else { "eager" }
                    ));
                }
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutFastBitsV1",
                "case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],
                "fast_self_consistent":true,"eager_repeats":2,"graph_repeats":2,
                "guards":true,"after_retained_win":true,
            })
        );
        prepared.fast_bits = Some(fast_bits);
        prepared.fast_graph = Some(fast_graph);
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
        graph: &CudaGraph,
        path: Path,
        golden: &[u32],
    ) -> Result<f64, String> {
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        fixture.reset_workspaces(&runtime.ctx.stream, arm)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("{} start event: {e:?}", arm.name()))?;
        for _ in 0..OPS {
            match path {
                Path::Eager => launch(runtime, fixture, case, arm)?,
                Path::Graph => graph
                    .launch()
                    .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?,
            }
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("{} end event: {e:?}", arm.name()))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("{} elapsed: {e:?}", arm.name()))?,
        ) * 1_000.0
            / OPS as f64;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} timed sync: {e:?}", arm.name()))?;
        if fixture.output(arm).bits(&runtime.ctx.stream)? != golden {
            return Err(format!("{} timed bits changed", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("{} invalid sample {us}", arm.name()));
        }
        Ok(us)
    }

    fn quantile(values: &[f64], q: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1)]
    }

    fn screen(
        runtime: &Runtime,
        case: Case,
        prepared: &mut Prepared,
        comparator: Arm,
        path: Path,
        order: BracketOrder,
        windows: usize,
    ) -> Result<[f64; 2], String> {
        let (comparator_graph, comparator_bits) = match comparator {
            Arm::Retained => (&prepared.retained_graph, prepared.exact_bits.as_slice()),
            Arm::Fast => (
                prepared
                    .fast_graph
                    .as_ref()
                    .ok_or("Fast graph requested before retained win")?,
                prepared
                    .fast_bits
                    .as_deref()
                    .ok_or("Fast bits requested before retained win")?,
            ),
            Arm::Candidate => return Err("candidate cannot be its own comparator".into()),
        };
        for _ in 0..WARMUPS {
            measure(
                runtime,
                &mut prepared.fixture,
                case,
                Arm::Candidate,
                &prepared.candidate_graph,
                path,
                &prepared.exact_bits,
            )?;
            measure(
                runtime,
                &mut prepared.fixture,
                case,
                comparator,
                comparator_graph,
                path,
                comparator_bits,
            )?;
        }
        let candidate_slots = order.candidate_slots();
        let mut raw = Vec::with_capacity(windows);
        let mut ratios = Vec::with_capacity(windows);
        for _ in 0..windows {
            let mut observation = [0.0; 4];
            for (index, candidate_slot) in candidate_slots.into_iter().enumerate() {
                let (arm, graph, bits) = if candidate_slot {
                    (
                        Arm::Candidate,
                        &prepared.candidate_graph,
                        prepared.exact_bits.as_slice(),
                    )
                } else {
                    (comparator, comparator_graph, comparator_bits)
                };
                observation[index] =
                    measure(runtime, &mut prepared.fixture, case, arm, graph, path, bits)?;
            }
            ratios.push(candidate_over_comparator_ratio(order, observation)?);
            raw.push(observation);
        }
        let result = [quantile(&ratios, 0.5), quantile(&ratios, 0.95)];
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutScreenV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid()?,
                "candidate":"pre_rna_ab_n96","comparator":comparator.name(),"path":path.name(),
                "order":order.name(),"windows":windows,"warmups_per_arm":WARMUPS,
                "logical_gemms_per_observation":OPS,"raw_observations_us":raw,
                "ratio_direction":"candidate_over_comparator","ratio_p50":result[0],"ratio_p95":result[1],
            })
        );
        Ok(result)
    }

    fn screen_all(
        runtime: &Runtime,
        case: Case,
        prepared: &mut Prepared,
        comparator: Arm,
        windows: usize,
    ) -> Result<Vec<[f64; 2]>, String> {
        let mut strata = Vec::with_capacity(4);
        for path in [Path::Eager, Path::Graph] {
            for order in [BracketOrder::Abba, BracketOrder::Baab] {
                strata.push(screen(
                    runtime, case, prepared, comparator, path, order, windows,
                )?);
            }
        }
        Ok(strata)
    }

    #[test]
    #[ignore = "requires exclusive quiet CC8.9/142-SM CUDA13.2; isolated TF32 TN joint A/B pre-RNA N96 d768-out once3 -> once7 -> Fast"]
    fn ada_tf32_tn_transpose_ab_rna_n96_d768_out_protocol() -> Result<(), String> {
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        let idle_resident_mode = common::gpu_quiet::idle_resident_mode_enabled();
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-tn-transpose-ab-rna-n96-d768-out/pre", 1_800)?;
        } else {
            quiet.require_pre_context("tf32-tn-transpose-ab-rna-n96-d768-out/pre")?;
        }
        let runtime = new_runtime()?;
        assert_eq!(D768_OUT.shape.grid()?, (96, 1, 1));
        assert_eq!(D768_OUT.shape.transpose_config()?.grid_dim, (48, 64, 1));
        check_exact_case(&runtime, FULL_TILE, false)?;
        check_exact_case(&runtime, FULL_TILE, true)?;
        check_exact_case(&runtime, TAIL, false)?;
        check_exact_case(&runtime, TAIL, true)?;
        check_exact_case(&runtime, K0, false)?;
        let case = D768_OUT;
        let mut prepared = prepare_target(&runtime, case)?;
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-tn-transpose-ab-rna-n96-d768-out/retained", 256)?;
        } else {
            quiet.require_cohort("tf32-tn-transpose-ab-rna-n96-d768-out/retained")?;
        }
        let scout_strata = screen_all(&runtime, case, &mut prepared, Arm::Retained, 3)?;
        let scout_win = all_strata_below(&scout_strata, FAST_THRESHOLD);
        let retained_strata = if scout_win {
            screen_all(&runtime, case, &mut prepared, Arm::Retained, 7)?
        } else {
            Vec::new()
        };
        let retained_win = scout_win && all_strata_below(&retained_strata, FAST_THRESHOLD);
        let fast_strata = if retained_win {
            prepare_fast(&runtime, case, &mut prepared)?;
            if idle_resident_mode {
                quiet.require_idle_resident("tf32-tn-transpose-ab-rna-n96-d768-out/fast", 256)?;
            } else {
                quiet.require_cohort("tf32-tn-transpose-ab-rna-n96-d768-out/fast")?;
            }
            screen_all(&runtime, case, &mut prepared, Arm::Fast, 7)?
        } else {
            Vec::new()
        };
        let fast_win = retained_win && all_strata_below(&fast_strata, FAST_THRESHOLD);
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeAbRnaN96D768OutDecisionV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid()?,
                "scout_windows":3,"scout_strata":scout_strata,"once7_windows":7,
                "retained_strata":retained_strata,"fast_strata":fast_strata,"threshold":FAST_THRESHOLD,
                "strata_order":["eager/ABBA","eager/BAAB","graph/ABBA","graph/BAAB"],
                "scout_win":scout_win,"retained_win":retained_win,"fast_win":fast_win,
                "decision":if !scout_win { "stop_after_once3" } else if !retained_win {
                    "stop_after_once7"
                } else if fast_win { "shortlist_strict_fast_win" } else { "retain_candidate_fast_miss" },
                "fast_screened":retained_win,"fast_qualified":fast_win,
                "idle_resident_mode":idle_resident_mode,"promotion":false,
            })
        );
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-tn-transpose-ab-rna-n96-d768-out/post", 256)?;
        } else {
            quiet.verify_post_cohort("tf32-tn-transpose-ab-rna-n96-d768-out/post")?;
        }
        Ok(())
    }
}
