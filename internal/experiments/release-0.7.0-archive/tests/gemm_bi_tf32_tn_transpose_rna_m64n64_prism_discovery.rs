//! Test-only Ada TF32 TN canonical-Prism M64N64/S3 wave-quantization discovery.
//! Production kernels and dispatch remain untouched.

#[path = "support/triad_tf32_tn_transpose_rna_m64n64_source.rs"]
#[allow(dead_code)]
mod candidate_source;
#[path = "support/triad_tn_transpose_n96_source.rs"]
mod raw_source;
#[path = "support/triad_tf32_tn_transpose_rna_n96_source.rs"]
#[allow(dead_code)]
mod retained_source;

const FIXED_N96_SOURCE: &str = include_str!("../kernels/gemm_bi_inference/tf32_rna_n96.cu");

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Target {
    label: &'static str,
    shape: (usize, usize, usize),
    candidate_grid: (u32, u32, u32),
    retained_grid: (u32, u32, u32),
    transpose_grid: (u32, u32, u32),
}

const PRISM_TARGET: Target = Target {
    label: "canonical_prism_in_proj",
    shape: (4_621, 384, 1_928),
    candidate_grid: (186, 1, 1),
    retained_grid: (63, 1, 1),
    transpose_grid: (12, 145, 1),
};

#[test]
fn prism_target_has_independently_derived_geometries() {
    assert_eq!(
        PRISM_TARGET,
        Target {
            label: "canonical_prism_in_proj",
            shape: (4_621, 384, 1_928),
            candidate_grid: (186, 1, 1),
            retained_grid: (63, 1, 1),
            transpose_grid: (12, 145, 1),
        }
    );
    assert_eq!(candidate_source::K8_ISSUE_OFFSETS, [0, 8, 16, 24]);
    assert_eq!(candidate_source::BLOCK_THREADS, 256);
    assert_eq!(candidate_source::DYNAMIC_SHARED_BYTES, 49_152);
    assert_eq!(candidate_source::MAX_REGISTERS, 128);
    assert_eq!(candidate_source::REQUIRED_OCCUPANCY, 2);
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
        assert_eq!(retained_source::tf32_rna_bits(input), expected);
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
    assert!(retained_source::validate_rna_transposed_words(&input, 2, 3, 4, &expected).is_ok());
    let mut wrong = expected;
    wrong[4] ^= 0x2000;
    assert!(retained_source::validate_rna_transposed_words(&input, 2, 3, 4, &wrong).is_err());
}

#[test]
fn adapter_preserves_pre_rna_contract_and_changes_tile_materially() {
    let raw = raw_source::candidate_source(FIXED_N96_SOURCE).unwrap();
    let retained = retained_source::compose_candidate_source(&raw).unwrap();
    let candidate = candidate_source::candidate_source(&raw).unwrap();
    assert_eq!(candidate.matches(candidate_source::GEMM_SYMBOL).count(), 1);
    assert_eq!(
        candidate
            .matches(candidate_source::TRANSPOSE_SYMBOL)
            .count(),
        1
    );
    assert!(!candidate.contains(retained_source::CANDIDATE_GEMM_SYMBOL));
    assert!(candidate.contains("float acc[2][2][4];"));
    assert!(candidate.contains("int warp_m = (warp >> 2) * 32;"));
    assert!(candidate.contains("int warp_n = (warp & 3) * 16;"));
    assert!(candidate.contains("fragments.a[m_atom][0] = raw0;"));
    assert_eq!(
        candidate
            .matches("tf32m64n64_round(__float_as_uint(b_step")
            .count(),
        2
    );
    assert!(
        candidate.contains("tf32m64n64_round(tile[(int)threadIdx.x][(int)threadIdx.y + offset])")
    );
    assert_eq!(
        candidate_source::retained_parent_source(&raw).unwrap(),
        retained
    );
    assert_ne!(candidate_source::retained_parent_fnv64(&raw).unwrap(), 0);
}

#[test]
fn eight_warps_own_each_m64n64_output_exactly_once() {
    let mut ownership = vec![0_u8; 64 * 64];
    for warp in 0..8 {
        let warp_m = (warp >> 2) * 32;
        let warp_n = (warp & 3) * 16;
        for row in warp_m..warp_m + 32 {
            for column in warp_n..warp_n + 16 {
                ownership[row * 64 + column] += 1;
            }
        }
    }
    assert!(ownership.into_iter().all(|owners| owners == 1));
}

#[test]
fn two_copy_slices_own_a_and_b_float4_vectors_exactly_once() {
    let mut a_ownership = vec![0_u8; 64 * 8];
    let mut b_ownership = vec![0_u8; 32 * 16];
    for thread in 0..256 {
        for slice in 0..2 {
            let linear = thread + slice * 256;
            a_ownership[(linear >> 3) * 8 + (linear & 7)] += 1;
            b_ownership[(linear / 16) * 16 + (linear % 16)] += 1;
        }
    }
    assert!(a_ownership.into_iter().all(|owners| owners == 1));
    assert!(b_ownership.into_iter().all(|owners| owners == 1));
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
            sys, CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig,
            PushKernelArg,
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
        ffi::{c_void, CStr},
        sync::Arc,
    };

    const ENV: &str = "MAMBA_TRIAD_TF32_TN_TRANSPOSE_RNA_M64N64_PRISM_DISCOVERY";
    const CANDIDATE_SHARED: usize = candidate_source::DYNAMIC_SHARED_BYTES;
    const RETAINED_SHARED: usize = 86_016;
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
        fn grid(self, arm: Arm) -> Result<(u32, u32, u32), String> {
            let (tile_m, tile_n) = match arm {
                Arm::Candidate => (64, 64),
                Arm::Retained => (128, 96),
                Arm::Fast => return Err("Fast has no custom GEMM grid".into()),
            };
            let blocks = self
                .k
                .div_ceil(tile_m)
                .checked_mul(self.n.div_ceil(tile_n))
                .ok_or("custom GEMM grid overflows usize")?;
            Ok((
                u32::try_from(blocks).map_err(|_| "custom GEMM grid exceeds u32")?,
                1,
                1,
            ))
        }

        fn config(self, arm: Arm) -> Result<LaunchConfig, String> {
            let shared = match arm {
                Arm::Candidate => CANDIDATE_SHARED,
                Arm::Retained => RETAINED_SHARED,
                Arm::Fast => return Err("Fast has no custom launch config".into()),
            };
            Ok(LaunchConfig {
                grid_dim: self.grid(arm)?,
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared as u32,
            })
        }

        fn transpose_stride(self) -> Result<usize, String> {
            raw_source::padded_stride(self.m)
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

    const PRISM: Case = Case {
        label: PRISM_TARGET.label,
        shape: Shape {
            m: PRISM_TARGET.shape.0,
            k: PRISM_TARGET.shape.1,
            n: PRISM_TARGET.shape.2,
        },
    };
    const FULL_TILE: Case = Case {
        label: "full_tile_tn_32x128x192",
        shape: Shape {
            m: 32,
            k: 128,
            n: 192,
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
    enum Arm {
        Candidate,
        Retained,
        Fast,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Candidate => "pre_rna_transpose_m64n64_s3_wave",
                Self::Retained => "pre_rna_transpose_n96",
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
            Ok(Self {
                a: GuardedF32::new(&runtime.ctx.stream, a, "A")?,
                b: GuardedF32::new(&runtime.ctx.stream, b, "B")?,
                candidate_scratch: GuardedF32::new(
                    &runtime.ctx.stream,
                    scratch.clone(),
                    "candidate transposed A",
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
        let common = include_str!("../kernels/gemm_bi_inference/common.cuh")
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n");
        let tf32 = include_str!("../kernels/gemm_bi_inference/tf32.cu");
        [prelude, &common, tf32, body].join("\n")
    }

    fn compile_module(
        device: &GpuDevice,
        source: String,
        symbol: &str,
        shared: usize,
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
                shared as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        Ok((module, function, source_sha, ptx_sha))
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var(ENV).as_deref() != Ok("1") {
            return Err(format!("set {ENV}=1"));
        }
        if cfg!(debug_assertions) {
            return Err("TN pre-RNA transpose N96 timing requires --release".into());
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
        let raw_body = raw_source::candidate_source(FIXED_N96_SOURCE)?;
        let retained_body = retained_source::compose_candidate_source(&raw_body)?;
        let candidate_body = candidate_source::candidate_source(&raw_body)?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) = compile_module(
            &device,
            module_source(&retained_body),
            retained_source::CANDIDATE_GEMM_SYMBOL,
            RETAINED_SHARED,
        )?;
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_module(
                &device,
                module_source(&candidate_body),
                candidate_source::GEMM_SYMBOL,
                CANDIDATE_SHARED,
            )?;
        let retained_transpose = retained_module
            .load_function(retained_source::CANDIDATE_TRANSPOSE_SYMBOL)
            .map_err(|error| format!("load retained transpose: {error:?}"))?;
        let candidate_transpose = candidate_module
            .load_function(candidate_source::TRANSPOSE_SYMBOL)
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
            retained_source::CANDIDATE_GEMM_SYMBOL,
            128,
            RETAINED_SHARED,
            1,
        )?;
        resource_gate(
            &runtime.candidate,
            candidate_source::GEMM_SYMBOL,
            candidate_source::MAX_REGISTERS,
            CANDIDATE_SHARED,
            candidate_source::REQUIRED_OCCUPANCY,
        )?;
        transpose_resource_gate(
            &runtime.retained_transpose,
            retained_source::CANDIDATE_TRANSPOSE_SYMBOL,
        )?;
        transpose_resource_gate(
            &runtime.candidate_transpose,
            candidate_source::TRANSPOSE_SYMBOL,
        )?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismIdentityV1",
                "retained_symbol":retained_source::CANDIDATE_GEMM_SYMBOL,
                "candidate_symbol":candidate_source::GEMM_SYMBOL,
                "retained_transform_symbol":retained_source::CANDIDATE_TRANSPOSE_SYMBOL,
                "candidate_transform_symbol":candidate_source::TRANSPOSE_SYMBOL,
                "retained_source_sha":runtime.retained_source_sha,
                "candidate_source_sha":runtime.candidate_source_sha,
                "retained_ptx_sha":runtime.retained_ptx_sha,
                "candidate_ptx_sha":runtime.candidate_ptx_sha,
                "conversion":"add_half_ulp_tf32_v1",
                "change":"M128N96_occupancy1_to_M64N64_occupancy2_wave_quantization",
            })
        );
        Ok(runtime)
    }

    fn resource_gate(
        function: &CudaFunction,
        symbol: &str,
        register_limit: i32,
        shared: usize,
        required_occupancy: u32,
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
            .occupancy_max_active_blocks_per_multiprocessor(256, shared, None)
            .map_err(|e| format!("{symbol} occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismResourceV1","symbol":symbol,
                "threads":256,"registers":registers,"local_bytes":local,
                "static_shared_bytes":static_shared,"dynamic_shared_bytes":shared,
                "max_threads_per_block":max_threads,"occupancy":occupancy,
            })
        );
        if registers <= 0
            || registers > register_limit
            || local != 0
            || static_shared != 0
            || max_threads < 256
            || max_dynamic < shared as i32
            || occupancy < required_occupancy
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
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismTransformResourceV1",
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
            Arm::Candidate | Arm::Retained => {
                let (function, transpose) = if arm == Arm::Candidate {
                    (&runtime.candidate, &runtime.candidate_transpose)
                } else {
                    (&runtime.retained, &runtime.retained_transpose)
                };
                let scratch = fixture.scratch(arm)?.ptr(&runtime.ctx.stream);
                let transpose_params = TransposeParams::new(shape)?;
                let mut transpose_builder = runtime.ctx.stream.launch_builder(transpose);
                transpose_builder
                    .arg(&a)
                    .arg(&scratch)
                    .arg(&transpose_params);
                unsafe { transpose_builder.launch(shape.transpose_config()?) }
                    .map_err(|e| format!("launch {} transpose: {e:?}", arm.name()))?;
                let bias = 0_u64;
                let params = Params::new(shape)?;
                let mut builder = runtime.ctx.stream.launch_builder(function);
                builder
                    .arg(&output)
                    .arg(&scratch)
                    .arg(&b)
                    .arg(&bias)
                    .arg(&params);
                unsafe { builder.launch(shape.config(arm)?) }
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
        if arm != Arm::Fast {
            fixture.scratch_mut(arm)?.reset(&runtime.ctx.stream)?;
        }
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
                    json!({"schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismGraphV1","arm":arm.name(),"nodes":count,"abi":"opaque","timing":"whole_graph"})
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
                candidate_source::GEMM_SYMBOL
            } else {
                retained_source::CANDIDATE_GEMM_SYMBOL
            };
            let expected_transpose = if arm == Arm::Candidate {
                candidate_source::TRANSPOSE_SYMBOL
            } else {
                retained_source::CANDIDATE_TRANSPOSE_SYMBOL
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
                    let config = case.shape.config(arm)?;
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
                    for (index, expected) in [(0, 8), (8, 8), (16, 12)].into_iter().enumerate() {
                        let (mut offset, mut size) = (0, 0);
                        if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                            != sys::CUresult::CUDA_SUCCESS
                            || (offset, size) != expected
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
                json!({"schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismGraphV1","arm":arm.name(),"nodes":2,"gemm_symbol":expected_gemm,"transpose_symbol":expected_transpose,"gemm_grid":case.shape.grid(arm)?,"transpose_grid":case.shape.transpose_config()?.grid_dim,"timing":"whole_two_node_graph"})
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
        if arm != Arm::Fast {
            fixture.scratch_mut(arm)?.reset(&runtime.ctx.stream)?;
        }
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
            retained_source::validate_rna_transposed_words(
                &input,
                case.shape.m,
                case.shape.k,
                stride,
                &scratch,
            )?;
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
            json!({"schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismBitsV1","case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],"exceptional":exceptional,"candidate_retained_exact":true,"eager_repeats":2,"graph_repeats":2,"guards":true})
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
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismTargetBitsV1",
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
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismFastBitsV1",
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
        if arm != Arm::Fast {
            fixture.scratch_mut(arm)?.reset(&runtime.ctx.stream)?;
        }
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
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismScreenV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid(Arm::Candidate)?,
                "candidate":"pre_rna_transpose_m64n64_s3_wave","comparator":comparator.name(),"path":path.name(),
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

    fn run_target_protocol(
        runtime: &Runtime,
        quiet: &common::gpu_quiet::QuietGpu,
        idle_resident_mode: bool,
        case: Case,
    ) -> Result<(), String> {
        let retained_label = format!("tf32-tn-transpose-rna-m64n64-prism/{}/retained", case.label);
        let fast_label = format!("tf32-tn-transpose-rna-m64n64-prism/{}/fast", case.label);
        let post_label = format!("tf32-tn-transpose-rna-m64n64-prism/{}/post", case.label);
        let mut prepared = prepare_target(runtime, case)?;
        if idle_resident_mode {
            quiet.require_idle_resident(&retained_label, 256)?;
        } else {
            quiet.require_cohort(&retained_label)?;
        }
        let scout_strata = screen_all(runtime, case, &mut prepared, Arm::Retained, 3)?;
        let scout_win = all_strata_below(&scout_strata, FAST_THRESHOLD);
        let retained_strata = if scout_win {
            screen_all(runtime, case, &mut prepared, Arm::Retained, 7)?
        } else {
            Vec::new()
        };
        let retained_win = scout_win && all_strata_below(&retained_strata, FAST_THRESHOLD);
        let fast_strata = if retained_win {
            prepare_fast(runtime, case, &mut prepared)?;
            if idle_resident_mode {
                quiet.require_idle_resident(&fast_label, 256)?;
            } else {
                quiet.require_cohort(&fast_label)?;
            }
            screen_all(runtime, case, &mut prepared, Arm::Fast, 7)?
        } else {
            Vec::new()
        };
        let fast_win = retained_win && all_strata_below(&fast_strata, FAST_THRESHOLD);
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32TnTransposeRnaM64N64PrismDecisionV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid(Arm::Candidate)?,
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
        drop(prepared);
        if idle_resident_mode {
            quiet.require_idle_resident(&post_label, 256)?;
        } else {
            quiet.verify_post_cohort(&post_label)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive quiet CC8.9/142-SM CUDA13.2; isolated TF32 TN pre-RNA M64N64/S3 canonical Prism once3 -> once7 -> Fast"]
    fn ada_tf32_tn_transpose_rna_m64n64_prism_protocol() -> Result<(), String> {
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        let idle_resident_mode = common::gpu_quiet::idle_resident_mode_enabled();
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-tn-transpose-rna-m64n64-prism/pre", 1_800)?;
        } else {
            quiet.require_pre_context("tf32-tn-transpose-rna-m64n64-prism/pre")?;
        }
        let runtime = new_runtime()?;
        assert_eq!(
            PRISM.shape.grid(Arm::Candidate)?,
            PRISM_TARGET.candidate_grid
        );
        assert_eq!(PRISM.shape.grid(Arm::Retained)?, PRISM_TARGET.retained_grid);
        assert_eq!(
            PRISM.shape.transpose_config()?.grid_dim,
            PRISM_TARGET.transpose_grid
        );
        check_exact_case(&runtime, FULL_TILE, false)?;
        check_exact_case(&runtime, FULL_TILE, true)?;
        check_exact_case(&runtime, TAIL, false)?;
        check_exact_case(&runtime, TAIL, true)?;
        check_exact_case(&runtime, K0, false)?;
        run_target_protocol(&runtime, &quiet, idle_resident_mode, PRISM)?;
        Ok(())
    }
}
