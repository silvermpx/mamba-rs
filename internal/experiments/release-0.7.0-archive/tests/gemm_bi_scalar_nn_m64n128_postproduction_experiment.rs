use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
mod common;

const DIMS: (usize, usize, usize) = (2_048, 768, 3_072);
const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";
const CANDIDATE_SYMBOL: &str = "gemm_bi_nn_m64n128_bk16_s2_post_exp_v1";
const CUDA_SOURCE_FILE: &str = "gemm_bi_scalar_nn_m64n128_postproduction_experiment.cu";
const SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const MIN_SPEEDUP_P05: f64 = 1.01;
const MIN_SPEEDUP_P50: f64 = 1.01;

fn candidate_source_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(CUDA_SOURCE_FILE)
}

fn read_candidate_source() -> Result<String, String> {
    let path = candidate_source_path();
    std::fs::read_to_string(&path)
        .map_err(|error| format!("read candidate CUDA source {}: {error}", path.display()))
}

fn exact_windows(value: Option<&OsStr>, required: usize) -> Result<usize, String> {
    if !matches!(required, SCREEN_WINDOWS | OFFICIAL_WINDOWS) {
        return Err(format!("unsupported qualification window count {required}"));
    }
    let actual = match value {
        Some(value) => value
            .to_str()
            .ok_or_else(|| "window override is not UTF-8".to_string())?
            .parse::<usize>()
            .map_err(|error| format!("parse qualification window override: {error}"))?,
        None => required,
    };
    if actual != required {
        return Err(format!(
            "qualification requires exactly {required} windows, received {actual}"
        ));
    }
    Ok(actual)
}

fn percentile(values: &[f64], quantile: f64) -> Result<f64, String> {
    if values.is_empty() || !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return Err("invalid percentile request".into());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("percentile samples must be finite and positive".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).floor() as usize;
    Ok(sorted[index])
}

fn validate_speedup(windows: usize, ratios: &[f64]) -> Result<(f64, f64, f64), String> {
    if !matches!(windows, SCREEN_WINDOWS | OFFICIAL_WINDOWS) || ratios.len() != windows {
        return Err("speedup sample count is not a frozen qualification window".into());
    }
    let stats = (
        percentile(ratios, 0.05)?,
        percentile(ratios, 0.50)?,
        percentile(ratios, 0.95)?,
    );
    if stats.0 < MIN_SPEEDUP_P05 || stats.1 < MIN_SPEEDUP_P50 {
        return Err(format!(
            "candidate speedup failed: p05={:.9} p50={:.9} requires p05>={MIN_SPEEDUP_P05:.3} p50>={MIN_SPEEDUP_P50:.3}",
            stats.0, stats.1
        ));
    }
    Ok(stats)
}

fn validate_exact_dims(dims: (usize, usize, usize)) -> Result<(), String> {
    if dims != DIMS {
        return Err(format!(
            "M64N128 post-production experiment only accepts {DIMS:?}, received {dims:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResourceArm {
    Production,
    Candidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PtxasResourceContract {
    registers: u64,
}

fn ptxas_resource_contract(
    target: &str,
    arm: ResourceArm,
) -> Result<PtxasResourceContract, String> {
    let registers = match (target, arm) {
        ("sm_80", ResourceArm::Production) => 123,
        ("sm_80", ResourceArm::Candidate) => 103,
        ("sm_120", ResourceArm::Production) => 101,
        ("sm_120", ResourceArm::Candidate) => 94,
        _ => return Err(format!("unsupported PTXAS resource target {target}")),
    };
    Ok(PtxasResourceContract { registers })
}

fn validate_ptxas_register_observation(
    contract: PtxasResourceContract,
    registers: &[u64],
) -> Result<(), String> {
    if registers != [contract.registers] {
        return Err(format!(
            "register census is {registers:?}, expected={}",
            contract.registers
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DriverResourceContract {
    registers: i32,
    occupancy: u32,
}

fn sm120_driver_resource_contract(arm: ResourceArm) -> DriverResourceContract {
    match arm {
        ResourceArm::Production => DriverResourceContract {
            registers: 103,
            occupancy: 4,
        },
        ResourceArm::Candidate => DriverResourceContract {
            registers: 96,
            occupancy: 2,
        },
    }
}

fn validate_driver_resource_observation(
    contract: DriverResourceContract,
    registers: i32,
    occupancy: u32,
    local: i32,
    static_shared: i32,
) -> Result<(), String> {
    if registers != contract.registers
        || occupancy != contract.occupancy
        || local != 0
        || static_shared != 0
    {
        return Err(format!(
            "registers={registers} expected_registers={} occupancy={occupancy} expected_occupancy={} local={local} static_shared={static_shared}",
            contract.registers, contract.occupancy
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExceptionalCase {
    reduction: usize,
    a_bits: u32,
    b_bits: u32,
    frozen_output_bits: Option<u32>,
}

const EXCEPTIONAL_CASES: [ExceptionalCase; 8] = [
    ExceptionalCase {
        reduction: 0,
        a_bits: 0x0000_0001,
        b_bits: 1.0_f32.to_bits(),
        frozen_output_bits: Some(0x0000_0001),
    },
    ExceptionalCase {
        reduction: 15,
        a_bits: 0x007f_ffff,
        b_bits: 1.0_f32.to_bits(),
        frozen_output_bits: Some(0x007f_ffff),
    },
    ExceptionalCase {
        reduction: 16,
        a_bits: 1.0_f32.to_bits(),
        b_bits: f32::INFINITY.to_bits(),
        frozen_output_bits: Some(0x7f80_0000),
    },
    ExceptionalCase {
        reduction: 31,
        a_bits: (-1.0_f32).to_bits(),
        b_bits: f32::INFINITY.to_bits(),
        frozen_output_bits: Some(0xff80_0000),
    },
    ExceptionalCase {
        reduction: 32,
        a_bits: 0x7fc2_3456,
        b_bits: 1.0_f32.to_bits(),
        frozen_output_bits: Some(0x7fff_ffff),
    },
    ExceptionalCase {
        reduction: 63,
        a_bits: 0x7fa1_2345,
        b_bits: 1.0_f32.to_bits(),
        frozen_output_bits: Some(0x7fff_ffff),
    },
    ExceptionalCase {
        reduction: 128,
        a_bits: 0.0_f32.to_bits(),
        b_bits: (-1.0_f32).to_bits(),
        frozen_output_bits: Some(0x0000_0000),
    },
    ExceptionalCase {
        reduction: 767,
        a_bits: (-0.0_f32).to_bits(),
        b_bits: 1.0_f32.to_bits(),
        frozen_output_bits: Some(0x0000_0000),
    },
];

#[test]
fn candidate_source_is_exact_one_owner_and_never_enters_production() {
    let source = read_candidate_source().expect("candidate CUDA source must exist");
    assert_eq!(
        source.matches(&format!("void {CANDIDATE_SYMBOL}(")).count(),
        1
    );
    for required in [
        "#define POST_NN_M64N128_BM 64",
        "#define POST_NN_M64N128_BN 128",
        "#define POST_NN_M64N128_BK 16",
        "#define POST_NN_M64N128_THREADS 256",
        "__launch_bounds__(POST_NN_M64N128_THREADS, 2)",
        "TOTAL_SMEM_BYTES == 25600",
        "SgbNnM64N128PostParams params",
        "__fmaf_rn(",
        "cp.async",
    ] {
        assert!(source.contains(required), "candidate omitted {required}");
    }
    let lower = source.to_ascii_lowercase();
    for forbidden in ["atomic", "mma.sync", "split_k", "--use_fast_math"] {
        assert!(!lower.contains(forbidden), "candidate contains {forbidden}");
    }
    let signature = source
        .split_once(&format!("void {CANDIDATE_SYMBOL}("))
        .and_then(|(_, tail)| tail.split_once(") {").map(|(parameters, _)| parameters))
        .expect("candidate signature");
    assert!(signature.matches(',').count() < 7);

    let modules = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert!(!modules.contains(CANDIDATE_SYMBOL));
    let production = include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu");
    assert_eq!(
        production
            .matches(&format!("void {PRODUCTION_SYMBOL}("))
            .count(),
        1
    );
}

#[test]
fn exact_shape_gate_rejects_every_neighbor_before_launch() {
    assert!(validate_exact_dims(DIMS).is_ok());
    for neighbor in [
        (DIMS.0 - 1, DIMS.1, DIMS.2),
        (DIMS.0 + 1, DIMS.1, DIMS.2),
        (DIMS.0, DIMS.1 - 1, DIMS.2),
        (DIMS.0, DIMS.1 + 1, DIMS.2),
        (DIMS.0, DIMS.1, DIMS.2 - 1),
        (DIMS.0, DIMS.1, DIMS.2 + 1),
    ] {
        assert!(
            validate_exact_dims(neighbor).is_err(),
            "accepted {neighbor:?}"
        );
    }
}

#[test]
fn resource_contracts_are_arch_specific_strict_and_fail_closed() {
    assert_eq!(
        ptxas_resource_contract("sm_80", ResourceArm::Production)
            .unwrap()
            .registers,
        123
    );
    assert_eq!(
        ptxas_resource_contract("sm_80", ResourceArm::Candidate)
            .unwrap()
            .registers,
        103
    );
    assert_eq!(
        ptxas_resource_contract("sm_120", ResourceArm::Production)
            .unwrap()
            .registers,
        101
    );
    assert_eq!(
        ptxas_resource_contract("sm_120", ResourceArm::Candidate)
            .unwrap()
            .registers,
        94
    );
    assert!(ptxas_resource_contract("sm_89", ResourceArm::Candidate).is_err());

    assert_eq!(
        sm120_driver_resource_contract(ResourceArm::Production),
        DriverResourceContract {
            registers: 103,
            occupancy: 4,
        }
    );
    assert_eq!(
        sm120_driver_resource_contract(ResourceArm::Candidate),
        DriverResourceContract {
            registers: 96,
            occupancy: 2,
        }
    );

    let ptxas = ptxas_resource_contract("sm_120", ResourceArm::Candidate).unwrap();
    assert!(validate_ptxas_register_observation(ptxas, &[94]).is_ok());
    for invalid in [vec![], vec![0], vec![93], vec![95], vec![94, 94]] {
        assert!(validate_ptxas_register_observation(ptxas, &invalid).is_err());
    }

    let driver = sm120_driver_resource_contract(ResourceArm::Candidate);
    assert!(validate_driver_resource_observation(driver, 96, 2, 0, 0).is_ok());
    for invalid in [
        (95, 2, 0, 0),
        (97, 2, 0, 0),
        (96, 1, 0, 0),
        (96, 3, 0, 0),
        (96, 2, 1, 0),
        (96, 2, 0, 1),
    ] {
        assert!(
            validate_driver_resource_observation(
                driver, invalid.0, invalid.1, invalid.2, invalid.3
            )
            .is_err()
        );
    }
}

#[test]
fn exceptional_literals_are_frozen_from_live_cuda_13_2_evidence() {
    assert_eq!(EXCEPTIONAL_CASES.len(), 8);
    assert_eq!(
        EXCEPTIONAL_CASES.map(|case| case.frozen_output_bits.unwrap()),
        [
            0x0000_0001,
            0x007f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fff_ffff,
            0x7fff_ffff,
            0x0000_0000,
            0x0000_0000,
        ]
    );
}

#[test]
fn windows_percentiles_and_promotion_thresholds_are_fail_closed() {
    assert_eq!(exact_windows(None, 21).unwrap(), 21);
    assert_eq!(exact_windows(Some(OsStr::new("101")), 101).unwrap(), 101);
    for invalid in ["0", "1", "20", "22", "100"] {
        assert!(exact_windows(Some(OsStr::new(invalid)), 21).is_err());
    }
    assert!(percentile(&[], 0.5).is_err());
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(percentile(&[1.0, invalid], 0.5).is_err());
    }
    assert!(validate_speedup(21, &[1.009; 21]).is_err());
    assert!(validate_speedup(21, &[1.01; 21]).is_ok());
    assert!(validate_speedup(101, &vec![1.02; 101]).is_ok());
}

#[cfg(feature = "cuda")]
mod cuda_experiment {
    use std::ffi::CStr;
    use std::process::Command;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, ResolvedGemmOp};
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;
    use super::{
        CANDIDATE_SYMBOL, DIMS, EXCEPTIONAL_CASES, OFFICIAL_WINDOWS, PRODUCTION_SYMBOL,
        ResourceArm, SCREEN_WINDOWS, exact_windows, percentile, ptxas_resource_contract,
        read_candidate_source, sm120_driver_resource_contract,
        validate_driver_resource_observation, validate_exact_dims,
        validate_ptxas_register_observation, validate_speedup,
    };

    const GUARD_ELEMENTS: usize = 64;
    const INPUT_GUARD_BITS: u32 = 0x7fc1_a128;
    const OUTPUT_GUARD_BITS: u32 = 0x7fc1_c128;
    const CORRECTNESS_REPEATS: usize = 3;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const PRODUCTION_CONFIG: LaunchConfig = LaunchConfig {
        grid_dim: (1_536, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 17_408,
    };
    const CANDIDATE_CONFIG: LaunchConfig = LaunchConfig {
        grid_dim: (768, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 25_600,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        Candidate,
    }

    impl Arm {
        fn symbol(self) -> &'static str {
            match self {
                Self::Production => PRODUCTION_SYMBOL,
                Self::Candidate => CANDIDATE_SYMBOL,
            }
        }

        fn config(self) -> LaunchConfig {
            match self {
                Self::Production => PRODUCTION_CONFIG,
                Self::Candidate => CANDIDATE_CONFIG,
            }
        }

        fn resource_arm(self) -> ResourceArm {
            match self {
                Self::Production => ResourceArm::Production,
                Self::Candidate => ResourceArm::Candidate,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PathKind {
        Eager,
        Graph,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Order {
        Abba,
        Baab,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct KernelParams {
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for KernelParams {}

    const _: [(); 32] = [(); std::mem::size_of::<KernelParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<KernelParams>()];

    struct Runtime {
        production_ctx: GpuCtx,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    struct Kernels {
        production: CudaFunction,
        candidate: CudaFunction,
    }

    impl Kernels {
        fn function(&self, arm: Arm) -> &CudaFunction {
            match arm {
                Arm::Production => &self.production,
                Arm::Candidate => &self.candidate,
            }
        }
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        active_offset: usize,
        active_len: usize,
        guard_bits: u32,
    }

    impl GuardedBuffer {
        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_offset = GUARD_ELEMENTS;
            let active_len = active.len();
            let total = active_offset
                .checked_add(active_len)
                .and_then(|value| value.checked_add(GUARD_ELEMENTS))
                .ok_or_else(|| "guarded allocation extent overflow".to_string())?;
            let mut expected = vec![f32::from_bits(guard_bits); total];
            expected[active_offset..active_offset + active_len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                active_offset,
                active_len,
                guard_bits,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.active_offset)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.expected)
        }

        fn snapshot(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<f32>, String> {
            let actual = self.buffer.to_cpu(stream)?;
            for (index, value) in actual[..self.active_offset]
                .iter()
                .chain(&actual[self.active_offset + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != self.guard_bits {
                    return Err(format!(
                        "{label} red zone {index} changed to 0x{:08x}",
                        value.to_bits()
                    ));
                }
            }
            Ok(actual)
        }

        fn bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.snapshot(stream, label)?;
            Ok(
                actual[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }

        fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.snapshot(stream, label)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} read-only data changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        production_output: GuardedBuffer,
        candidate_output: GuardedBuffer,
        a_host: Vec<f32>,
        b_host: Vec<f32>,
        kind: usize,
        params: KernelParams,
    }

    impl Fixture {
        fn output(&self, arm: Arm) -> &GuardedBuffer {
            match arm {
                Arm::Production => &self.production_output,
                Arm::Candidate => &self.candidate_output,
            }
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedBuffer {
            match arm {
                Arm::Production => &mut self.production_output,
                Arm::Candidate => &mut self.candidate_output,
            }
        }
    }

    fn compose_source() -> Result<String, String> {
        let candidate = read_candidate_source()?;
        Ok([
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
            candidate.as_str(),
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n"))
    }

    fn compile_ptx(target: &'static str) -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some(target),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "-DGEMM_BI_GROUP_M=16".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(compose_source()?, options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile M64N128 post-production experiment: {error:?}"))
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        )
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "M64N128 post-production experiment requires CC12.0/170SM, found CC{}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let production_ctx = GpuCtx::new(&device)?;
        production_ctx.set_batch_invariant(true);
        production_ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        production_ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        presize_physical_qualification_suite(&production_ctx, &[production_request()])?;

        let ptx = compile_ptx(device.nvrtc_target())?;
        validate_ptx(&ptx)?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load M64N128 experiment module: {error:?}"))?;
        Ok(Runtime {
            production_ctx,
            stream,
            module,
        })
    }

    fn production_holder(runtime: &Runtime) -> Result<QualifiedPhysicalLaunch<'_>, String> {
        let mut holder = qualify_physical_launch(&runtime.production_ctx, production_request())?;
        holder.seed_f32_operands(&runtime.production_ctx, 0x4d64_1280)?;
        let evidence = holder.evidence();
        let nodes = evidence.nodes();
        if evidence.launch_count() != 1
            || nodes.len() != 1
            || nodes[0].symbol != PRODUCTION_SYMBOL
            || nodes[0].module_kind != ModuleKind::TriadScalar
            || nodes[0].launch.grid_dim != PRODUCTION_CONFIG.grid_dim
            || nodes[0].launch.block_dim != PRODUCTION_CONFIG.block_dim
            || nodes[0].launch.shared_mem_bytes != PRODUCTION_CONFIG.shared_mem_bytes
            || nodes[0].launch.arguments_digest == [0; 32]
            || evidence.launch_digest() == [0; 32]
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "public production M64N64 physical identity drifted: evidence={evidence:?}"
            ));
        }
        Ok(holder)
    }

    fn load_kernels(runtime: &Runtime) -> Result<Kernels, String> {
        let load = |arm: Arm| -> Result<CudaFunction, String> {
            let function = runtime
                .module
                .load_function(arm.symbol())
                .map_err(|error| format!("load {}: {error:?}", arm.symbol()))?;
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    arm.config().shared_mem_bytes as i32,
                )
                .map_err(|error| format!("set {} dynamic shared: {error:?}", arm.symbol()))?;
            Ok(function)
        };
        Ok(Kernels {
            production: load(Arm::Production)?,
            candidate: load(Arm::Candidate)?,
        })
    }

    fn seeded_values(len: usize, mut state: u64, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4093 == 0 {
                    if index.is_multiple_of(2) { 0.0 } else { -0.0 }
                } else {
                    let signed = ((state >> 32) as u32 % 2049) as i32 - 1024;
                    signed as f32 * (scale / 1024.0)
                }
            })
            .collect()
    }

    fn make_values(kind: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (m, k, n) = DIMS;
        if kind == 0 {
            return (
                seeded_values(m * k, 0xa128_0001, 0.25),
                seeded_values(k * n, 0xb128_0002, 0.125),
                vec![0.0; m * n],
            );
        }
        let mut a = vec![0.0_f32; m * k];
        let mut b = vec![0.0_f32; k * n];
        let output = vec![0.0_f32; m * n];
        if kind == 1 {
            for (reduction, value) in [
                (0, 2.0_f32.powi(25)),
                (15, -2.0_f32.powi(25)),
                (16, 1.0),
                (31, 2.0_f32.powi(-20)),
                (767, -2.0_f32.powi(-20)),
            ] {
                a[reduction] = value;
                b[reduction * n] = 1.0;
            }
        } else {
            let case = EXCEPTIONAL_CASES[kind - 2];
            a[case.reduction] = f32::from_bits(case.a_bits);
            b[case.reduction * n] = f32::from_bits(case.b_bits);
        }
        (a, b, output)
    }

    fn new_fixture(runtime: &Runtime, kind: usize) -> Result<Fixture, String> {
        let (a_host, b_host, output) = make_values(kind);
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.stream, a_host.clone(), INPUT_GUARD_BITS)?,
            b: GuardedBuffer::new(&runtime.stream, b_host.clone(), INPUT_GUARD_BITS)?,
            production_output: GuardedBuffer::new(
                &runtime.stream,
                output.clone(),
                OUTPUT_GUARD_BITS,
            )?,
            candidate_output: GuardedBuffer::new(&runtime.stream, output, OUTPUT_GUARD_BITS)?,
            a_host,
            b_host,
            kind,
            params: KernelParams {
                alpha: 1.0,
                beta: 0.0,
                m: DIMS.0 as i32,
                n: DIMS.2 as i32,
                k: DIMS.1 as i32,
                lda: DIMS.1 as i32,
                ldb: DIMS.2 as i32,
                ldc: DIMS.2 as i32,
            },
        })
    }

    fn launch(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        validate_exact_dims((
            fixture.params.m as usize,
            fixture.params.k as usize,
            fixture.params.n as usize,
        ))?;
        if fixture.params.alpha.to_bits() != 1.0_f32.to_bits()
            || fixture.params.beta.to_bits() != 0.0_f32.to_bits()
            || fixture.params.lda != DIMS.1 as i32
            || fixture.params.ldb != DIMS.2 as i32
            || fixture.params.ldc != DIMS.2 as i32
        {
            return Err("M64N128 launch operands or contiguous strides drifted".into());
        }
        let output = fixture.output(arm).ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
        let bias = 0_u64;
        if [output, a, b]
            .into_iter()
            .any(|pointer| pointer == 0 || pointer & 15 != 0)
        {
            return Err("M64N128 launch requires non-null 16-byte-aligned C/A/B".into());
        }
        let mut builder = runtime.stream.launch_builder(kernels.function(arm));
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        unsafe { builder.launch(arm.config()) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", arm.symbol()))
    }

    fn capture(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch(runtime, kernels, fixture, arm)) }
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    fn graph_identity(graph: &CudaGraph, arm: Arm) -> Result<[u8; 32], String> {
        let raw = graph.cu_graph();
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut count) },
            "query graph nodes",
        )?;
        if count != 1 {
            return Err(format!("{} graph has {count} nodes", arm.symbol()));
        }
        let mut edges = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edges,
                )
            },
            "query graph edges",
        )?;
        if edges != 0 {
            return Err(format!("{} graph has {edges} edges", arm.symbol()));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, &mut node, &mut count) },
            "read graph node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "read graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("{} graph node is {kind:?}", arm.symbol()));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "read graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "read graph function name",
        )?;
        if name.is_null() || params.kernelParams.is_null() {
            return Err(format!(
                "{} graph omitted function or arguments",
                arm.symbol()
            ));
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph symbol is not UTF-8: {error}"))?;
        let config = arm.config();
        if symbol != arm.symbol()
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != config.grid_dim
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != config.block_dim
            || params.sharedMemBytes != config.shared_mem_bytes
        {
            return Err(format!("{} graph physical identity drifted", arm.symbol()));
        }
        let mut digest = Sha256::new();
        digest.update(b"scalar-nn-m64n128-post-graph-args.v1");
        for (index, size) in [8_usize, 8, 8, 8, 32].into_iter().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{} graph argument {index} is null", arm.symbol()));
            }
            digest.update((size as u64).to_le_bytes());
            digest.update(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        let digest: [u8; 32] = digest.finalize().into();
        if digest == [0; 32] {
            return Err(format!("{} graph argument digest is zero", arm.symbol()));
        }
        Ok(digest)
    }

    fn dot_bits(fixture: &Fixture, row: usize, column: usize) -> u32 {
        let mut sum = 0.0_f32;
        for reduction in 0..DIMS.1 {
            sum = fixture.a_host[row * DIMS.1 + reduction]
                .mul_add(fixture.b_host[reduction * DIMS.2 + column], sum);
        }
        sum.to_bits()
    }

    fn validate_witnesses(fixture: &Fixture, production: &[u32]) -> Result<(), String> {
        if fixture.kind == 1 {
            let mut expected = vec![0.0_f32.to_bits(); DIMS.0 * DIMS.2];
            expected[0] = dot_bits(fixture, 0, 0);
            if production != expected {
                return Err("production tree-killer output differs from independent oracle".into());
            }
        } else if fixture.kind >= 2 {
            let case = EXCEPTIONAL_CASES[fixture.kind - 2];
            if let Some(expected) = case.frozen_output_bits {
                if production[0] != expected {
                    return Err(format!(
                        "production exceptional witness {} drifted: expected 0x{expected:08x}, received 0x{:08x}",
                        fixture.kind - 2,
                        production[0]
                    ));
                }
                return Ok(());
            }
            let witness = f32::from_bits(production[0]);
            let valid = match fixture.kind {
                2 | 3 => witness.is_subnormal(),
                4 => witness == f32::INFINITY,
                5 => witness == f32::NEG_INFINITY,
                6 | 7 => witness.is_nan(),
                8 | 9 => witness == 0.0,
                _ => false,
            };
            if !valid {
                return Err(format!(
                    "production exceptional witness {} drifted: 0x{:08x}",
                    fixture.kind - 2,
                    production[0]
                ));
            }
        }
        Ok(())
    }

    fn validate_correctness(
        runtime: &Runtime,
        kernels: &Kernels,
        kind: usize,
    ) -> Result<(), String> {
        let mut fixture = new_fixture(runtime, kind)?;
        let production_graph = capture(runtime, kernels, &fixture, Arm::Production)?;
        let candidate_graph = capture(runtime, kernels, &fixture, Arm::Candidate)?;
        let production_digest = graph_identity(&production_graph, Arm::Production)?;
        let candidate_digest = graph_identity(&candidate_graph, Arm::Candidate)?;
        if production_digest == candidate_digest {
            return Err("production and candidate physical argument digests collided".into());
        }

        fixture.output_mut(Arm::Production).reset(&runtime.stream)?;
        launch(runtime, kernels, &fixture, Arm::Production)?;
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize production reference: {error:?}"))?;
        let reference = fixture
            .output(Arm::Production)
            .bits(&runtime.stream, "production reference")?;
        validate_witnesses(&fixture, &reference)?;

        for (arm, graph) in [
            (Arm::Production, &production_graph),
            (Arm::Candidate, &candidate_graph),
        ] {
            for path in [PathKind::Eager, PathKind::Graph] {
                for repeat in 0..CORRECTNESS_REPEATS {
                    fixture.output_mut(arm).reset(&runtime.stream)?;
                    match path {
                        PathKind::Eager => launch(runtime, kernels, &fixture, arm)?,
                        PathKind::Graph => graph
                            .launch()
                            .map_err(|error| format!("launch {} graph: {error:?}", arm.symbol()))?,
                    }
                    runtime
                        .stream
                        .synchronize()
                        .map_err(|error| format!("synchronize {}: {error:?}", arm.symbol()))?;
                    let actual = fixture.output(arm).bits(&runtime.stream, arm.symbol())?;
                    if actual != reference {
                        return Err(format!(
                            "{} {path:?} repeat {repeat} differs from production bits",
                            arm.symbol()
                        ));
                    }
                    fixture.a.unchanged(&runtime.stream, "A")?;
                    fixture.b.unchanged(&runtime.stream, "B")?;
                }
            }
        }
        Ok(())
    }

    fn validate_driver_resources(kernels: &Kernels) -> Result<(), String> {
        let mut failures = Vec::new();
        for arm in [Arm::Production, Arm::Candidate] {
            let function = kernels.function(arm);
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", arm.symbol()))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", arm.symbol()))?;
            let static_shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} static shared: {error:?}", arm.symbol()))?;
            let config = arm.config();
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    config.block_dim.0,
                    config.shared_mem_bytes as usize,
                    None,
                )
                .map_err(|error| format!("{} occupancy: {error:?}", arm.symbol()))?;
            eprintln!(
                "scalar_nn_m64n128 resource symbol={} threads={} dynamic_shared={} static_shared={} registers={} local={} occupancy={}",
                arm.symbol(),
                config.block_dim.0,
                config.shared_mem_bytes,
                static_shared,
                registers,
                local,
                occupancy
            );
            let expected = sm120_driver_resource_contract(arm.resource_arm());
            if let Err(detail) = validate_driver_resource_observation(
                expected,
                registers,
                occupancy,
                local,
                static_shared,
            ) {
                failures.push(format!(
                    "{} violates exact SM120 resource contract: {detail}",
                    arm.symbol()
                ));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    fn ptx_entry<'a>(ptx: &'a str, symbol: &str) -> Result<&'a str, String> {
        let marker = format!(".entry {symbol}(");
        let (_, tail) = ptx
            .split_once(&marker)
            .ok_or_else(|| format!("PTX omitted {symbol}"))?;
        Ok(tail
            .split_once(".visible .entry ")
            .map_or(tail, |(entry, _)| entry))
    }

    fn validate_ptx(ptx: &str) -> Result<(), String> {
        for arm in [Arm::Production, Arm::Candidate] {
            let entry = ptx_entry(ptx, arm.symbol())?;
            let parameters = entry
                .split_once("\n)")
                .map(|(parameters, _)| parameters)
                .ok_or_else(|| format!("{} PTX parameter list is malformed", arm.symbol()))?;
            if parameters.matches(".param").count() != 5 || !entry.contains("fma.rn.f32") {
                return Err(format!("{} PTX ABI or FFMA contract changed", arm.symbol()));
            }
            for token in [
                "atom.", "atom::", "red.", "red::", "redux.", "mma.", ".ftz", "call.",
            ] {
                if entry
                    .split_ascii_whitespace()
                    .any(|field| field.starts_with(token) || field.contains(".ftz"))
                {
                    return Err(format!("{} PTX contains forbidden {token}", arm.symbol()));
                }
            }
        }
        Ok(())
    }

    fn metric_before(line: &str, suffix: &str) -> Option<u64> {
        line.split_once(suffix)?
            .0
            .split(|character: char| !character.is_ascii_digit())
            .rfind(|field| !field.is_empty())?
            .parse()
            .ok()
    }

    fn command_output(mut command: Command, label: &str) -> Result<std::process::Output, String> {
        let output = command
            .output()
            .map_err(|error| format!("run {label}: {error}"))?;
        if output.status.success() {
            Ok(output)
        } else {
            Err(format!(
                "{label} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn line_offset(source: &str, mut predicate: impl FnMut(&str) -> bool) -> Option<usize> {
        let mut offset = 0;
        for line in source.split_inclusive('\n') {
            let content = line.trim_end_matches(['\n', '\r']);
            if predicate(content) {
                return Some(offset);
            }
            offset += line.len();
        }
        None
    }

    fn nvdisasm_symbol(line: &str) -> Option<&str> {
        let line = line
            .trim()
            .strip_prefix("//")
            .map(str::trim)
            .unwrap_or(line.trim());
        let symbol = line.strip_prefix("Function : ")?;
        (!symbol.is_empty() && !symbol.contains(char::is_whitespace)).then_some(symbol)
    }

    fn sass_entry<'a>(sass: &'a str, symbol: &str) -> Result<&'a str, String> {
        if let Some(start) = line_offset(sass, |line| nvdisasm_symbol(line) == Some(symbol)) {
            let tail = &sass[start..];
            let body_start = tail
                .find('\n')
                .map(|offset| offset + 1)
                .unwrap_or(tail.len());
            let end = line_offset(&tail[body_start..], |line| nvdisasm_symbol(line).is_some())
                .map(|offset| body_start + offset)
                .unwrap_or(tail.len());
            return Ok(&tail[..end]);
        }
        let label = format!("\n{symbol}:\n");
        let start = sass
            .find(&label)
            .map(|offset| offset + 1)
            .ok_or_else(|| format!("SASS omitted {symbol}"))?;
        let tail = &sass[start..];
        let end = tail
            .find("\n//--------------------- .text.")
            .or_else(|| tail.find("\n\t.section\t.text."))
            .unwrap_or(tail.len());
        Ok(&tail[..end])
    }

    fn validate_ptxas_and_sass(target: &str, ptx: &str) -> Result<(), String> {
        validate_ptx(ptx)?;
        let nonce = std::process::id();
        let stem = format!("mamba-nn-m64n128-post-{target}-{nonce}");
        let directory = std::env::temp_dir();
        let ptx_path = directory.join(format!("{stem}.ptx"));
        let cubin_path = directory.join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
        let output = command_output(
            {
                let mut command = Command::new("ptxas");
                command
                    .arg(format!("--gpu-name={target}"))
                    .arg("--verbose")
                    .arg(&ptx_path)
                    .arg("--output-file")
                    .arg(&cubin_path);
                command
            },
            "ptxas",
        );
        let result = (|| {
            let output = output?;
            let report = format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let disassembly = command_output(
                {
                    let mut command = Command::new("nvdisasm");
                    command.arg(&cubin_path);
                    command
                },
                "nvdisasm",
            )?;
            let sass = String::from_utf8_lossy(&disassembly.stdout);
            for arm in [Arm::Production, Arm::Candidate] {
                let marker = format!("Compiling entry function '{}'", arm.symbol());
                let block = report
                    .split_once(&marker)
                    .and_then(|(_, tail)| tail.split("Compiling entry function '").next())
                    .ok_or_else(|| format!("ptxas omitted {} resource record", arm.symbol()))?;
                for metric in [
                    " bytes stack frame",
                    " bytes spill stores",
                    " bytes spill loads",
                ] {
                    let values = block
                        .lines()
                        .filter_map(|line| metric_before(line, metric))
                        .collect::<Vec<_>>();
                    if values != [0] {
                        return Err(format!(
                            "{} {target} has nonzero or ambiguous {metric}: {values:?}",
                            arm.symbol()
                        ));
                    }
                }
                let registers = block
                    .lines()
                    .filter_map(|line| metric_before(line, " registers"))
                    .collect::<Vec<_>>();
                let contract = ptxas_resource_contract(target, arm.resource_arm())?;
                validate_ptxas_register_observation(contract, &registers)
                    .map_err(|detail| format!("{} {target} {detail}", arm.symbol()))?;
                eprintln!(
                    "scalar_nn_m64n128 ptxas target={target} symbol={} registers={}",
                    arm.symbol(),
                    registers[0]
                );
                let entry = sass_entry(&sass, arm.symbol())?;
                if !entry.contains("FFMA") || !entry.contains("LDGSTS") {
                    return Err(format!("{} SASS omitted FFMA or LDGSTS", arm.symbol()));
                }
                for forbidden in [" LDL", " STL", " ATOM", " RED", " REDUX"] {
                    if entry.contains(forbidden) {
                        return Err(format!("{} SASS contains {forbidden}", arm.symbol()));
                    }
                }
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(ptx_path);
        let _ = std::fs::remove_file(cubin_path);
        result
    }

    struct TimingFixture {
        fixture: Fixture,
        production_graph: CudaGraph,
        candidate_graph: CudaGraph,
    }

    fn timing_fixture(runtime: &Runtime, kernels: &Kernels) -> Result<TimingFixture, String> {
        let fixture = new_fixture(runtime, 0)?;
        let production_graph = capture(runtime, kernels, &fixture, Arm::Production)?;
        let candidate_graph = capture(runtime, kernels, &fixture, Arm::Candidate)?;
        graph_identity(&production_graph, Arm::Production)?;
        graph_identity(&candidate_graph, Arm::Candidate)?;
        Ok(TimingFixture {
            fixture,
            production_graph,
            candidate_graph,
        })
    }

    fn measure(
        runtime: &Runtime,
        kernels: &Kernels,
        timing: &TimingFixture,
        arm: Arm,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be nonzero".into());
        }
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record start event: {error:?}"))?;
        for _ in 0..iterations {
            match (arm, path) {
                (_, PathKind::Eager) => launch(runtime, kernels, &timing.fixture, arm)?,
                (Arm::Production, PathKind::Graph) => timing
                    .production_graph
                    .launch()
                    .map_err(|error| format!("launch production graph: {error:?}"))?,
                (Arm::Candidate, PathKind::Graph) => timing
                    .candidate_graph
                    .launch()
                    .map_err(|error| format!("launch candidate graph: {error:?}"))?,
            }
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record end event: {error:?}"))?;
        let us = start
            .elapsed_ms(&end)
            .map(|milliseconds| f64::from(milliseconds) * 1_000.0 / iterations as f64)
            .map_err(|error| format!("measure {}: {error:?}", arm.symbol()))?;
        if us.is_finite() && us > 0.0 {
            Ok(us)
        } else {
            Err(format!("{} returned invalid timing {us}", arm.symbol()))
        }
    }

    fn adjacent(
        runtime: &Runtime,
        kernels: &Kernels,
        timing: &TimingFixture,
        path: PathKind,
        order: Order,
        iterations: usize,
    ) -> Result<(f64, f64), String> {
        let sequence = match order {
            Order::Abba => [
                Arm::Production,
                Arm::Candidate,
                Arm::Candidate,
                Arm::Production,
            ],
            Order::Baab => [
                Arm::Candidate,
                Arm::Production,
                Arm::Production,
                Arm::Candidate,
            ],
        };
        let mut production = Vec::with_capacity(2);
        let mut candidate = Vec::with_capacity(2);
        for arm in sequence {
            let value = measure(runtime, kernels, timing, arm, path, iterations)?;
            match arm {
                Arm::Production => production.push(value),
                Arm::Candidate => candidate.push(value),
            }
        }
        Ok((
            production.iter().sum::<f64>() / 2.0,
            candidate.iter().sum::<f64>() / 2.0,
        ))
    }

    fn paired(
        runtime: &Runtime,
        kernels: &Kernels,
        timing: &TimingFixture,
        path: PathKind,
        order: Order,
        windows: usize,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        let production_probe = measure(runtime, kernels, timing, Arm::Production, path, 4)?;
        let candidate_probe = measure(runtime, kernels, timing, Arm::Candidate, path, 4)?;
        let iterations = (TARGET_WINDOW_US / production_probe.min(candidate_probe)).ceil() as usize;
        if iterations == 0 {
            return Err("timing calibration returned zero iterations".into());
        }
        let label = format!("scalar-nn-m64n128-post/{path:?}/{order:?}/{windows}");
        quiet.require_cohort(&label)?;
        let body = (|| {
            let mut production_us = Vec::with_capacity(windows);
            let mut candidate_us = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (production, candidate) =
                    adjacent(runtime, kernels, timing, path, order, iterations)?;
                if !production.is_finite()
                    || production <= 0.0
                    || !candidate.is_finite()
                    || candidate <= 0.0
                {
                    return Err(format!(
                        "invalid paired sample production={production} candidate={candidate}"
                    ));
                }
                production_us.push(production);
                candidate_us.push(candidate);
                ratios.push(production / candidate);
            }
            let stats = validate_speedup(windows, &ratios)?;
            eprintln!(
                "scalar_nn_m64n128 path={path:?} order={order:?} windows={windows} production_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
                percentile(&production_us, 0.50)?,
                percentile(&candidate_us, 0.50)?,
                stats.0,
                stats.1,
                stats.2
            );
            Ok(())
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(&label).map(drop))
    }

    fn run_timing(windows: usize) -> Result<(), String> {
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let label = format!("scalar-nn-m64n128-post-{windows}");
        quiet.require_pre_context(&label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            let mut holder = production_holder(&runtime)?;
            holder.measure_graph_window_ms(&runtime.production_ctx, 1)?;
            holder.measure_eager_window_ms(&runtime.production_ctx, 1)?;
            drop(holder);
            let kernels = load_kernels(&runtime)?;
            validate_driver_resources(&kernels)?;
            let timing = timing_fixture(&runtime, &kernels)?;
            for path in [PathKind::Eager, PathKind::Graph] {
                for order in [Order::Abba, Order::Baab] {
                    paired(&runtime, &kernels, &timing, path, order, windows, &quiet)?;
                }
            }
            Ok(())
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(&label).map(drop))
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC but launches no GPU work"]
    fn candidate_compiles_for_compute80_and_sm120_with_exact_ptx() -> Result<(), String> {
        validate_ptx(&compile_ptx("compute_80")?)?;
        validate_ptx(&compile_ptx("compute_120")?)
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC, ptxas, and nvdisasm but launches no GPU work"]
    fn candidate_meets_sm80_sm120_ptxas_sass_and_resource_contracts() -> Result<(), String> {
        validate_ptxas_and_sass("sm_80", &compile_ptx("compute_80")?)?;
        validate_ptxas_and_sass("sm_120", &compile_ptx("compute_120")?)
    }

    #[test]
    #[ignore = "requires an exclusive exact SM120 CUDA13.2 GPU"]
    fn candidate_is_exact_guarded_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        let _holder = production_holder(&runtime)?;
        let kernels = load_kernels(&runtime)?;
        validate_driver_resources(&kernels)?;
        for kind in 0..10 {
            validate_correctness(&runtime, &kernels, kind)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet exact SM120 CUDA13.2 GPU and release build"]
    fn candidate_screen_21_windows_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("timing requires --release".into());
        }
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_NN_M64N128_POST_WINDOWS").as_deref(),
            SCREEN_WINDOWS,
        )?;
        run_timing(windows)
    }

    #[test]
    #[ignore = "requires an exclusive quiet exact SM120 CUDA13.2 GPU and release build"]
    fn candidate_official_101_windows_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("official timing requires --release".into());
        }
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_NN_M64N128_POST_WINDOWS").as_deref(),
            OFFICIAL_WINDOWS,
        )?;
        run_timing(windows)
    }
}

fn verify_post_cohort_even_on_error(
    body: Result<(), String>,
    postflight: Result<(), String>,
) -> Result<(), String> {
    match (body, postflight) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(body), Ok(())) => Err(body),
        (Ok(()), Err(postflight)) => Err(postflight),
        (Err(body), Err(postflight)) => Err(format!("{body}; quiet-GPU postflight: {postflight}")),
    }
}

#[test]
fn postflight_errors_are_never_hidden_by_body_results() {
    assert!(verify_post_cohort_even_on_error(Ok(()), Ok(())).is_ok());
    assert_eq!(
        verify_post_cohort_even_on_error(Err("body".into()), Ok(())).unwrap_err(),
        "body"
    );
    assert_eq!(
        verify_post_cohort_even_on_error(Ok(()), Err("post".into())).unwrap_err(),
        "post"
    );
    let both =
        verify_post_cohort_even_on_error(Err("body".into()), Err("post".into())).unwrap_err();
    assert!(both.contains("body") && both.contains("post"));
}
