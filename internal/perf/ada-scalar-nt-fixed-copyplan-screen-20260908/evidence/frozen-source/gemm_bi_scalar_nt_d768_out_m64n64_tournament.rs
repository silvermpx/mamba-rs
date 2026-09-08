const TRANSPOSE_32X16: &str = "gemm_bi_transpose_f32_32x16_d768_exp";
const FIXED_COPYPLAN_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
const TEST_SOURCE: &str = include_str!("gemm_bi_scalar_nt_d768_transpose_tournament.cu");
#[cfg(feature = "cuda")]
const PRODUCTION_TRANSPOSE_SOURCE: &str =
    include_str!("../kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu");
#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod fixed_full_mantissa;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaBracketOrder {
    Abba,
    Baab,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct NnParams {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaCopyPlanContract {
    symbol: &'static str,
    params_size: usize,
    params_align: usize,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    static_shared: usize,
    dynamic_shared: usize,
    minimum_occupancy: u32,
}

const fn ada_copyplan_contract() -> AdaCopyPlanContract {
    AdaCopyPlanContract {
        symbol: FIXED_COPYPLAN_SYMBOL,
        params_size: size_of::<NnParams>(),
        params_align: align_of::<NnParams>(),
        grid: (768, 1, 1),
        block: (128, 1, 1),
        static_shared: 32_768,
        dynamic_shared: 0,
        minimum_occupancy: 1,
    }
}

fn validate_ada_copyplan_contract(actual: AdaCopyPlanContract) -> Result<(), String> {
    let expected = ada_copyplan_contract();
    if actual != expected {
        return Err(format!(
            "Fixed CopyPlan launch contract changed: actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}

#[test]
fn ada_fixed_copyplan_contract_is_exact_and_distinct_from_portable_m64() {
    let contract = ada_copyplan_contract();
    assert_eq!(contract.symbol, FIXED_COPYPLAN_SYMBOL);
    assert_eq!((contract.params_size, contract.params_align), (32, 4));
    assert_eq!(contract.grid, (768, 1, 1));
    assert_eq!(contract.block, (128, 1, 1));
    assert_eq!(
        (contract.static_shared, contract.dynamic_shared),
        (32_768, 0)
    );
    assert_eq!(contract.minimum_occupancy, 1);
    validate_ada_copyplan_contract(contract).unwrap();
    for changed in [
        AdaCopyPlanContract {
            params_size: 40,
            ..contract
        },
        AdaCopyPlanContract {
            grid: (767, 1, 1),
            ..contract
        },
        AdaCopyPlanContract {
            static_shared: 0,
            dynamic_shared: 32_768,
            ..contract
        },
        AdaCopyPlanContract {
            minimum_occupancy: 0,
            ..contract
        },
    ] {
        assert!(validate_ada_copyplan_contract(changed).is_err());
    }
}

fn ada_candidate_over_auto(raw: [f64; 4], order: AdaBracketOrder) -> f64 {
    let (candidate, auto) = match order {
        AdaBracketOrder::Abba => (0.5 * (raw[1] + raw[2]), 0.5 * (raw[0] + raw[3])),
        AdaBracketOrder::Baab => (0.5 * (raw[0] + raw[3]), 0.5 * (raw[1] + raw[2])),
    };
    candidate / auto
}

fn ada_percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

fn ada_retain_stratum(ratios: &[f64]) -> bool {
    if ratios.len() != 7
        || ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
    {
        return false;
    }
    matches!(
        (
            ada_percentile(ratios, 0.50),
            ada_percentile(ratios, 0.95),
        ),
        (Some(p50), Some(p95)) if p50 < 0.99 && p95 < 0.99
    )
}

fn ada_retain_decision(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < 0.99
                && *p95 < 0.99
        })
}

#[test]
fn ada_once7_decision_uses_raw_order_and_requires_both_percentiles_below_point99() {
    let abba = ada_candidate_over_auto([10.0, 8.0, 8.0, 10.0], AdaBracketOrder::Abba);
    let baab = ada_candidate_over_auto([8.0, 10.0, 10.0, 8.0], AdaBracketOrder::Baab);
    assert_eq!(abba, 0.8);
    assert_eq!(baab, 0.8);

    assert!(ada_retain_stratum(&[
        0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.989
    ]));
    assert!(!ada_retain_stratum(&[]));
    assert!(!ada_retain_stratum(&[0.98; 6]));
    assert!(!ada_retain_stratum(&[0.98; 8]));
    assert!(!ada_retain_stratum(&[0.0; 7]));
    assert!(!ada_retain_stratum(&[-0.1; 7]));
    assert!(!ada_retain_stratum(&[f64::NAN; 7]));
    assert!(!ada_retain_stratum(&[
        0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99
    ]));
    assert!(!ada_retain_stratum(&[
        0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 1.01
    ]));
    assert!(ada_retain_decision(&[[0.98, 0.989]; 4]));
    assert!(!ada_retain_decision(&[[0.98, 0.989]; 3]));
    assert!(!ada_retain_decision(&[[0.98, 0.989]; 5]));
    assert!(!ada_retain_decision(&[
        [0.98, 0.989],
        [0.98, 0.989],
        [0.98, 0.99],
        [0.98, 0.989],
    ]));
}

#[test]
fn d768_out_source_is_test_only_bounded_copy_kernel() {
    let registry = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    {
        let symbol = TRANSPOSE_32X16;
        let marker = format!("void {symbol}(");
        assert_eq!(TEST_SOURCE.matches(&marker).count(), 1);
        let (_, signature) = TEST_SOURCE
            .split_once(&marker)
            .expect("candidate signature");
        let (parameters, _) = signature
            .split_once(") {")
            .expect("candidate parameter list");
        assert!(parameters.matches(',').count() < 7);
        assert!(
            !registry.contains(symbol),
            "{symbol} escaped into production"
        );
    }
    for required in [
        "__shared__ float tile[32][33]",
        "transpose_d768_body<16>",
        "__launch_bounds__(512, 2)",
    ] {
        assert!(TEST_SOURCE.contains(required), "candidate lost {required}");
    }
    for forbidden in ["atomic", "mma.sync"] {
        assert!(!TEST_SOURCE.contains(forbidden));
    }
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use std::ffi::CStr;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dx_raw;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;
    use super::{
        AdaBracketOrder, FIXED_COPYPLAN_SYMBOL, NnParams, PRODUCTION_TRANSPOSE_SOURCE, TEST_SOURCE,
        TRANSPOSE_32X16, ada_candidate_over_auto, ada_copyplan_contract, ada_percentile,
        ada_retain_decision, fixed_full_mantissa, validate_ada_copyplan_contract,
    };

    const DIMS: (usize, usize, usize) = (2_048, 1_536, 768);
    const M64_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";
    const GENERIC_NT_SYMBOL: &str = "gemm_bi_nt";
    const TRANSPOSE_32X32: &str = "gemm_bi_transpose_f32_2d";
    const PROMOTED_TRANSPOSE: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
    const GUARD: usize = 64;
    const INPUT_GUARD: u32 = 0x7fc0_a768;
    const OUTPUT_GUARD: u32 = 0x7fc0_c768;
    const M64_SHARED: usize = 17_408;
    const NT_SHARED: usize = 33_376;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const TRANSPOSE_STATIC_SHARED: usize = 32 * 33 * size_of::<f32>();
    const BASELINE_MIN_P05_SPEEDUP: f64 = 1.45;
    const BASELINE_MIN_P50_SPEEDUP: f64 = 1.50;
    const TRANSPOSE16_MIN_P05_SPEEDUP: f64 = 0.97;
    const TRANSPOSE16_MIN_P50_SPEEDUP: f64 = 1.002;
    const PARITY_MIN_P05: f64 = 0.97;
    const PARITY_MIN_P50: f64 = 0.985;
    const PARITY_MAX_P50: f64 = 1.015;
    const PARITY_MAX_P95: f64 = 1.03;
    const ADA_WINDOWS: usize = 7;
    const ADA_WARMUPS: usize = 4;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        GenericNt,
        Transpose32,
        Transpose16,
        Transpose16FixedCopyPlan,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Production => "production_nt",
                Self::GenericNt => "generic_nt",
                Self::Transpose32 => "transpose32_plus_m64n64",
                Self::Transpose16 => "transpose32x16_plus_m64n64",
                Self::Transpose16FixedCopyPlan => "transpose32x16_plus_fixed_copyplan",
            }
        }

        const fn is_candidate(self) -> bool {
            matches!(
                self,
                Self::Transpose32 | Self::Transpose16 | Self::Transpose16FixedCopyPlan
            )
        }
    }

    unsafe impl DeviceRepr for NnParams {}

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
        symbol: &'static str,
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _module: Arc<CudaModule>,
        m64: Kernel,
        generic_nt: Kernel,
        transpose32: Kernel,
        transpose16: Kernel,
        fixed_copyplan: Option<Kernel>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CandidateNodeIdentity {
        symbol: String,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        arguments_digest: [u8; 32],
    }

    fn validate_candidate_identity(
        nodes: &[CandidateNodeIdentity],
        arm: Arm,
    ) -> Result<(), String> {
        validate_candidate_identity_with_transpose16(nodes, arm, TRANSPOSE_32X16)
    }

    fn validate_candidate_identity_with_transpose16(
        nodes: &[CandidateNodeIdentity],
        arm: Arm,
        transpose16_symbol: &str,
    ) -> Result<(), String> {
        if nodes.len() != 2 {
            return Err(format!(
                "candidate must contain two ordered nodes, found {}",
                nodes.len()
            ));
        }
        let copyplan = ada_copyplan_contract();
        let (transpose_symbol, transpose_block, inner_symbol, inner_shared) = match arm {
            Arm::Transpose32 => (TRANSPOSE_32X32, (32, 32, 1), M64_SYMBOL, M64_SHARED as u32),
            Arm::Transpose16 => (
                transpose16_symbol,
                (32, 16, 1),
                M64_SYMBOL,
                M64_SHARED as u32,
            ),
            Arm::Transpose16FixedCopyPlan => (
                transpose16_symbol,
                (32, 16, 1),
                copyplan.symbol,
                copyplan.dynamic_shared as u32,
            ),
            _ => return Err(format!("{arm:?} is not a transpose candidate")),
        };
        if nodes[0].symbol != transpose_symbol || nodes[1].symbol != inner_symbol {
            return Err(format!("candidate node order changed: {nodes:?}"));
        }
        let expected_configs = [
            ((24, 48, 1), transpose_block, 0),
            ((768, 1, 1), (128, 1, 1), inner_shared),
        ];
        for (node, (grid, block, shared)) in nodes.iter().zip(expected_configs) {
            if (node.grid_dim, node.block_dim, node.shared_mem_bytes) != (grid, block, shared) {
                return Err(format!("{} launch config changed: {node:?}", node.symbol));
            }
        }
        for node in nodes {
            if node.arguments_digest == [0; 32] {
                return Err(format!("{} has a zero argument digest", node.symbol));
            }
        }
        if nodes[0].arguments_digest == nodes[1].arguments_digest {
            return Err("candidate node argument digests collided".into());
        }
        Ok(())
    }

    fn validate_qualified_environment(
        compute_capability: (u32, u32),
        multiprocessor_count: u32,
        nvrtc_target: &str,
        compiler: CompilerIdentity,
        artifact: ArtifactIdentity,
    ) -> Result<(), String> {
        if compute_capability != (12, 0)
            || multiprocessor_count != 170
            || nvrtc_target != "compute_120"
            || compiler.target.as_str() != "compute_120"
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.header_manifest_digest == [0; 32]
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.output_kind != ArtifactKind::Ptx
            || compiler.composer_revision != COMPOSER_REVISION
            || compiler.compiler_revision != COMPILER_REVISION
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || artifact.module_kind != ModuleKind::TriadScalar
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "d768-out tournament requires the qualified compute_120 NVRTC 13.2 TriadScalar artifact domain: cc={compute_capability:?} sms={multiprocessor_count} target={nvrtc_target} compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
    }

    fn validate_ada_qualified_environment(
        compute_capability: (u32, u32),
        multiprocessor_count: u32,
        nvrtc_target: &str,
        compiler: CompilerIdentity,
        artifact: ArtifactIdentity,
    ) -> Result<(), String> {
        let expected_target = GpuDevice::resolve_nvrtc_target(compute_capability)?;
        if compute_capability != (8, 9)
            || multiprocessor_count != 142
            || nvrtc_target != expected_target
            || compiler.target.as_str() != nvrtc_target
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.header_manifest_digest == [0; 32]
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.output_kind != ArtifactKind::Ptx
            || compiler.composer_revision != COMPOSER_REVISION
            || compiler.compiler_revision != COMPILER_REVISION
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || artifact.module_kind != ModuleKind::TriadScalar
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "Ada d768-out discovery requires the qualified device-target sm_89 NVRTC 13.2 TriadScalar artifact domain: cc={compute_capability:?} sms={multiprocessor_count} target={nvrtc_target} compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
    }

    fn validate_ada_fixed_copyplan_environment(ctx: &GpuCtx) -> Result<(), String> {
        let compiler = ctx.kernels.compiler_identity();
        let artifact = ctx.kernels.artifact_set_identity().fixed;
        if compiler.target.as_str() != "sm_89"
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.header_manifest_digest == [0; 32]
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.output_kind != ArtifactKind::Ptx
            || compiler.composer_revision != COMPOSER_REVISION
            || compiler.compiler_revision != COMPILER_REVISION
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || artifact.module_kind != ModuleKind::Fixed
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "Ada NT CopyPlan discovery requires the independently qualified production Fixed sm_89 artifact: compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        offset: usize,
        len: usize,
        guard: u32,
    }

    impl GuardedBuffer {
        fn new(stream: &Arc<CudaStream>, values: Vec<f32>, guard: u32) -> Result<Self, String> {
            let offset = GUARD;
            let len = values.len();
            let mut expected = vec![f32::from_bits(guard); offset + len + GUARD];
            expected[offset..offset + len].copy_from_slice(&values);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                offset,
                len,
                guard,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.offset)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.expected)?;
            stream
                .synchronize()
                .map_err(|error| format!("guarded reset sync: {error:?}"))
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            stream
                .synchronize()
                .map_err(|error| format!("{label} download sync: {error:?}"))?;
            for (index, value) in values[..self.offset]
                .iter()
                .chain(&values[self.offset + self.len..])
                .enumerate()
            {
                if value.to_bits() != self.guard {
                    return Err(format!("{label} red zone changed at {index}"));
                }
            }
            Ok(values[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.buffer.to_cpu(stream)?;
            stream
                .synchronize()
                .map_err(|error| format!("{label} download sync: {error:?}"))?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} or its red zones changed"));
            }
            Ok(())
        }

        fn expected_active_bits(&self) -> Vec<u32> {
            self.expected[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        production_a: GpuBuffer,
        production_a_expected: Vec<f32>,
        b: GuardedBuffer,
        scratch: GuardedBuffer,
        generic_output: GuardedBuffer,
        candidate_output: GuardedBuffer,
        production_output: GpuBuffer,
        production_output_seed: Vec<f32>,
        params: NnParams,
    }

    fn compose_source() -> String {
        [
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/scalar.cu"),
            include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
            TEST_SOURCE,
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
    }

    fn compose_ada_source() -> String {
        [
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/scalar.cu"),
            include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
            PRODUCTION_TRANSPOSE_SOURCE,
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
    }

    fn load_kernel(
        module: &Arc<CudaModule>,
        symbol: &'static str,
        config: LaunchConfig,
        dynamic_shared: usize,
    ) -> Result<Kernel, String> {
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        if dynamic_shared > 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared as i32,
                )
                .map_err(|error| format!("set {symbol} shared memory: {error:?}"))?;
        }
        Ok(Kernel {
            function,
            config,
            symbol,
        })
    }

    fn runtime_fixed_copyplan(
        ctx: &GpuCtx,
        contract: super::AdaCopyPlanContract,
    ) -> Result<Kernel, String> {
        let function = ctx
            .kernels
            .fixed_sm89_f32_n64_copyplan
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                ctx.kernels
                    .fixed_sm89_f32_n64_copyplan_rejection
                    .clone()
                    .unwrap_or_else(|| "production Fixed CopyPlan holder is unavailable".into())
            })?;
        Ok(Kernel {
            function,
            config: LaunchConfig {
                grid_dim: contract.grid,
                block_dim: contract.block,
                shared_mem_bytes: contract.dynamic_shared as u32,
            },
            symbol: contract.symbol,
        })
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "d768-out tournament requires CC12.0/170 SM, found {:?}/{} SM",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        validate_qualified_environment(
            identity.compute_capability,
            identity.multiprocessor_count,
            device.nvrtc_target(),
            ctx.kernels.triad_scalar_compiler_identity(),
            ctx.kernels.artifact_set_identity().triad_scalar,
        )?;
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "-DGEMM_BI_GROUP_M=16".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
            .map_err(|error| format!("compile d768-out tournament: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let candidate_artifact_digest: [u8; 32] = Sha256::digest(ptx_source.as_bytes()).into();
        if candidate_artifact_digest == [0; 32] {
            return Err("candidate PTX artifact digest is zero".into());
        }
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load d768-out tournament module: {error:?}"))?;
        let transpose_grid = (24, 48, 1);
        let m64 = load_kernel(
            &module,
            M64_SYMBOL,
            LaunchConfig {
                grid_dim: (768, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
            },
            M64_SHARED,
        )?;
        let generic_nt = load_kernel(
            &module,
            GENERIC_NT_SYMBOL,
            LaunchConfig {
                grid_dim: (192, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: NT_SHARED as u32,
            },
            NT_SHARED,
        )?;
        let transpose32 = load_kernel(
            &module,
            TRANSPOSE_32X32,
            LaunchConfig {
                grid_dim: transpose_grid,
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        let transpose16 = load_kernel(
            &module,
            TRANSPOSE_32X16,
            LaunchConfig {
                grid_dim: transpose_grid,
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            m64,
            generic_nt,
            transpose32,
            transpose16,
            fixed_copyplan: None,
        })
    }

    fn new_ada_runtime() -> Result<Runtime, String> {
        new_ada_runtime_with_copyplan(false)
    }

    fn new_ada_copyplan_runtime() -> Result<Runtime, String> {
        new_ada_runtime_with_copyplan(true)
    }

    fn new_ada_runtime_with_copyplan(load_copyplan: bool) -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
            return Err(format!(
                "Ada d768-out discovery requires CC8.9/142 SM, found {:?}/{} SM",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        validate_ada_qualified_environment(
            identity.compute_capability,
            identity.multiprocessor_count,
            device.nvrtc_target(),
            ctx.kernels.triad_scalar_compiler_identity(),
            ctx.kernels.artifact_set_identity().triad_scalar,
        )?;
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some(device.nvrtc_target()),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "-DGEMM_BI_GROUP_M=16".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_ada_source(), options)
            .map_err(|error| format!("compile Ada d768-out discovery: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let candidate_artifact_digest: [u8; 32] = Sha256::digest(ptx_source.as_bytes()).into();
        if candidate_artifact_digest == [0; 32] {
            return Err("Ada candidate PTX artifact digest is zero".into());
        }
        eprintln!(
            "ada_nt_d768_out source_sha256={:02x?} artifact_sha256={candidate_artifact_digest:02x?} target={}",
            <[u8; 32]>::from(Sha256::digest(compose_ada_source().as_bytes())),
            device.nvrtc_target(),
        );
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load Ada d768-out discovery module: {error:?}"))?;
        let transpose_grid = (24, 48, 1);
        let m64 = load_kernel(
            &module,
            M64_SYMBOL,
            LaunchConfig {
                grid_dim: (768, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
            },
            M64_SHARED,
        )?;
        let generic_nt = load_kernel(
            &module,
            GENERIC_NT_SYMBOL,
            LaunchConfig {
                grid_dim: (192, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: NT_SHARED as u32,
            },
            NT_SHARED,
        )?;
        let transpose32 = load_kernel(
            &module,
            TRANSPOSE_32X32,
            LaunchConfig {
                grid_dim: transpose_grid,
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        let transpose16 = load_kernel(
            &module,
            PROMOTED_TRANSPOSE,
            LaunchConfig {
                grid_dim: transpose_grid,
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        let fixed_copyplan = if load_copyplan {
            let contract = ada_copyplan_contract();
            validate_ada_copyplan_contract(contract)?;
            validate_ada_fixed_copyplan_environment(&ctx)?;
            Some(runtime_fixed_copyplan(&ctx, contract)?)
        } else {
            None
        };
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            m64,
            generic_nt,
            transpose32,
            transpose16,
            fixed_copyplan,
        })
    }

    fn values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4093 == 0 {
                    -0.0
                } else {
                    let sign = ((state >> 63) as u32) << 31;
                    let exponent = (123 + ((state >> 29) as u32 % 8)) << 23;
                    let mantissa = (state as u32 & 0x007f_ffff) | 1;
                    f32::from_bits(sign | exponent | mantissa)
                }
            })
            .collect()
    }

    fn new_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let a_expected = values(m * n, 0xa768_1001);
        let b_values = values(k_out * n, 0xb768_1002);
        new_fixture_with_values(runtime, a_expected, b_values)
    }

    fn new_ada_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let a_values = fixed_full_mantissa::finite_full_mantissa_values(m * n, 0xa89a_7681);
        let b_values = fixed_full_mantissa::finite_full_mantissa_values(k_out * n, 0xb89a_7682);
        let scratch_seed = fixed_full_mantissa::finite_full_mantissa_values(k_out * n, 0x589a_7683);
        let output_seed = fixed_full_mantissa::finite_full_mantissa_values(m * k_out, 0xc89a_7684);
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a_values.clone(), INPUT_GUARD)?,
            production_a: GpuBuffer::from_cpu(&runtime.ctx.stream, &a_values)?,
            production_a_expected: a_values,
            b: GuardedBuffer::new(&runtime.ctx.stream, b_values, INPUT_GUARD)?,
            scratch: GuardedBuffer::new(&runtime.ctx.stream, scratch_seed, OUTPUT_GUARD)?,
            generic_output: GuardedBuffer::new(
                &runtime.ctx.stream,
                output_seed.clone(),
                OUTPUT_GUARD,
            )?,
            candidate_output: GuardedBuffer::new(
                &runtime.ctx.stream,
                output_seed.clone(),
                OUTPUT_GUARD,
            )?,
            production_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &output_seed)?,
            production_output_seed: output_seed,
            params: NnParams {
                alpha: 1.0,
                beta: 0.0,
                m: m as i32,
                n: k_out as i32,
                k: n as i32,
                lda: n as i32,
                ldb: k_out as i32,
                ldc: k_out as i32,
            },
        })
    }

    fn new_fixture_with_values(
        runtime: &Runtime,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
    ) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        if a_values.len() != m * n || b_values.len() != k_out * n {
            return Err("d768-out fixture dimensions do not match inputs".into());
        }
        let output = vec![0.0; m * k_out];
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a_values.clone(), INPUT_GUARD)?,
            production_a: GpuBuffer::from_cpu(&runtime.ctx.stream, &a_values)?,
            production_a_expected: a_values,
            b: GuardedBuffer::new(&runtime.ctx.stream, b_values, INPUT_GUARD)?,
            scratch: GuardedBuffer::new(&runtime.ctx.stream, vec![0.0; k_out * n], OUTPUT_GUARD)?,
            generic_output: GuardedBuffer::new(&runtime.ctx.stream, output.clone(), OUTPUT_GUARD)?,
            candidate_output: GuardedBuffer::new(
                &runtime.ctx.stream,
                output.clone(),
                OUTPUT_GUARD,
            )?,
            production_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &output)?,
            production_output_seed: output,
            params: NnParams {
                alpha: 1.0,
                beta: 0.0,
                m: m as i32,
                n: k_out as i32,
                k: n as i32,
                lda: n as i32,
                ldb: k_out as i32,
                ldc: k_out as i32,
            },
        })
    }

    fn candidate_inner_kernel(runtime: &Runtime, arm: Arm) -> Result<&Kernel, String> {
        match arm {
            Arm::Transpose16FixedCopyPlan => runtime.fixed_copyplan.as_ref().ok_or_else(|| {
                "Ada Fixed CopyPlan candidate is missing its production Fixed holder".into()
            }),
            Arm::Transpose32 | Arm::Transpose16 => Ok(&runtime.m64),
            _ => Err(format!("{arm:?} is not a two-node transpose candidate")),
        }
    }

    fn launch_arm(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        let a = fixture.a.ptr(stream);
        let b = fixture.b.ptr(stream);
        match arm {
            Arm::Production => gpu_gemm_bi_backward_dx_raw(
                &runtime.ctx,
                &mut fixture.production_output,
                &fixture.production_a,
                b,
                DIMS.0,
                DIMS.1,
                DIMS.2,
            ),
            Arm::GenericNt => {
                let output = fixture.generic_output.ptr(stream);
                let alpha = 1.0_f32;
                let (m, k_out, n) = (DIMS.0 as i32, DIMS.1 as i32, DIMS.2 as i32);
                let mut builder = stream.launch_builder(&runtime.generic_nt.function);
                builder.arg(&output);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&alpha);
                builder.arg(&m);
                builder.arg(&n);
                builder.arg(&k_out);
                unsafe { builder.launch(runtime.generic_nt.config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch {}: {error:?}", runtime.generic_nt.symbol))
            }
            candidate => {
                let transpose = match candidate {
                    Arm::Transpose32 => &runtime.transpose32,
                    Arm::Transpose16 | Arm::Transpose16FixedCopyPlan => &runtime.transpose16,
                    _ => unreachable!(),
                };
                let scratch = fixture.scratch.ptr(stream);
                let rows = DIMS.1 as i32;
                let cols = DIMS.2 as i32;
                let mut transpose_builder = stream.launch_builder(&transpose.function);
                transpose_builder.arg(&scratch);
                transpose_builder.arg(&b);
                transpose_builder.arg(&rows);
                transpose_builder.arg(&cols);
                unsafe { transpose_builder.launch(transpose.config) }
                    .map_err(|error| format!("launch {}: {error:?}", transpose.symbol))?;
                let output = fixture.candidate_output.ptr(stream);
                let bias = 0_u64;
                let inner = candidate_inner_kernel(runtime, candidate)?;
                let mut m64_builder = stream.launch_builder(&inner.function);
                m64_builder.arg(&output);
                m64_builder.arg(&a);
                m64_builder.arg(&scratch);
                m64_builder.arg(&bias);
                m64_builder.arg(&fixture.params);
                unsafe { m64_builder.launch(inner.config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch {}: {error:?}", inner.symbol))
            }
        }
    }

    fn capture_arm(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch_arm(runtime, fixture, arm)) }
    }

    fn capture_ada_actual_auto(
        runtime: &Runtime,
        fixture: &mut Fixture,
    ) -> Result<CudaGraph, String> {
        // Prepared f32 Triad cache entries bind the exact operand pointers. The
        // qualification probe above uses its own buffers, so warm this fixture's
        // unchanged public AUTO call immediately before graph capture.
        reset_ada_arm(runtime, fixture, Arm::Production)?;
        launch_arm(runtime, fixture, Arm::Production)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("Ada actual AUTO capture warmup sync: {error:?}"))?;
        capture_arm(runtime, fixture, Arm::Production)
    }

    fn reset_ada_arm(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<(), String> {
        fixture.a.reset(&runtime.ctx.stream)?;
        fixture.b.reset(&runtime.ctx.stream)?;
        match arm {
            Arm::Production => {
                fixture
                    .production_a
                    .upload(&runtime.ctx.stream, &fixture.production_a_expected)?;
                fixture
                    .production_output
                    .upload(&runtime.ctx.stream, &fixture.production_output_seed)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("production reset sync: {error:?}"))?;
            }
            Arm::GenericNt => fixture.generic_output.reset(&runtime.ctx.stream)?,
            candidate if candidate.is_candidate() => {
                fixture.scratch.reset(&runtime.ctx.stream)?;
                fixture.candidate_output.reset(&runtime.ctx.stream)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"nt-d768-out-candidate-arguments.v1");
        for argument in arguments {
            digest.update((argument.len() as u64).to_le_bytes());
            digest.update(argument);
        }
        digest.finalize().into()
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn eager_candidate_identity(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
    ) -> Vec<CandidateNodeIdentity> {
        let scratch = fixture.scratch.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
        let rows = DIMS.1 as i32;
        let cols = DIMS.2 as i32;
        let output = fixture.candidate_output.ptr(&runtime.ctx.stream);
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let bias = 0_u64;
        let transpose = match arm {
            Arm::Transpose32 => &runtime.transpose32,
            Arm::Transpose16 | Arm::Transpose16FixedCopyPlan => &runtime.transpose16,
            _ => panic!("{arm:?} is not a transpose candidate"),
        };
        let inner = candidate_inner_kernel(runtime, arm)
            .expect("candidate identity requires a loaded two-node inner kernel");
        vec![
            CandidateNodeIdentity {
                symbol: transpose.symbol.into(),
                grid_dim: transpose.config.grid_dim,
                block_dim: transpose.config.block_dim,
                shared_mem_bytes: transpose.config.shared_mem_bytes,
                arguments_digest: digest_arguments(&[
                    bytes_of(&scratch),
                    bytes_of(&b),
                    bytes_of(&rows),
                    bytes_of(&cols),
                ]),
            },
            CandidateNodeIdentity {
                symbol: inner.symbol.into(),
                grid_dim: inner.config.grid_dim,
                block_dim: inner.config.block_dim,
                shared_mem_bytes: inner.config.shared_mem_bytes,
                arguments_digest: digest_arguments(&[
                    bytes_of(&output),
                    bytes_of(&a),
                    bytes_of(&scratch),
                    bytes_of(&bias),
                    bytes_of(&fixture.params),
                ]),
            },
        ]
    }

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    unsafe fn graph_node_identity(node: sys::CUgraphNode) -> Result<CandidateNodeIdentity, String> {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("candidate graph contains non-kernel node {kind:?}"));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "kernel name",
        )?;
        if name.is_null() {
            return Err("graph kernel name is null".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph kernel name is not UTF-8: {error}"))?
            .to_owned();
        let argument_sizes: &[usize] = match symbol.as_str() {
            TRANSPOSE_32X16 | PROMOTED_TRANSPOSE | TRANSPOSE_32X32 => &[8, 8, 4, 4],
            M64_SYMBOL | FIXED_COPYPLAN_SYMBOL => &[8, 8, 8, 8, size_of::<NnParams>()],
            _ => return Err(format!("unexpected candidate graph symbol {symbol}")),
        };
        if params.kernelParams.is_null() {
            return Err(format!("{symbol} graph kernelParams is null"));
        }
        let mut arguments = Vec::with_capacity(argument_sizes.len());
        for (index, size) in argument_sizes.iter().copied().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{symbol} argument {index} is null"));
            }
            arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        Ok(CandidateNodeIdentity {
            symbol,
            grid_dim: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block_dim: (params.blockDimX, params.blockDimY, params.blockDimZ),
            shared_mem_bytes: params.sharedMemBytes,
            arguments_digest: digest_arguments(&arguments),
        })
    }

    fn graph_candidate_identity(
        graph: &CudaGraph,
        arm: Arm,
    ) -> Result<Vec<CandidateNodeIdentity>, String> {
        graph_candidate_identity_with_transpose16(graph, arm, TRANSPOSE_32X16)
    }

    fn graph_candidate_identity_with_transpose16(
        graph: &CudaGraph,
        arm: Arm,
        transpose16_symbol: &str,
    ) -> Result<Vec<CandidateNodeIdentity>, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "graph node count",
        )?;
        if node_count != 2 {
            return Err(format!(
                "candidate graph has {node_count} nodes, expected two"
            ));
        }
        let mut edge_count = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                )
            },
            "graph edge count",
        )?;
        if edge_count != 1 {
            return Err(format!(
                "candidate graph has {edge_count} edges, expected one"
            ));
        }
        let mut from = [std::ptr::null_mut(); 1];
        let mut to = [std::ptr::null_mut(); 1];
        let mut edge_data: [sys::CUgraphEdgeData; 1] = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    edge_data.as_mut_ptr(),
                    &mut edge_count,
                )
            },
            "graph edges",
        )?;
        if edge_data[0].from_port != 0
            || edge_data[0].to_port != 0
            || edge_data[0].type_
                != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
            || edge_data[0].reserved != [0; 5]
        {
            return Err("candidate graph edge descriptor is not the exact CUDA default".into());
        }
        let ordered = vec![unsafe { graph_node_identity(from[0]) }?, unsafe {
            graph_node_identity(to[0])
        }?];
        validate_candidate_identity_with_transpose16(&ordered, arm, transpose16_symbol)?;
        Ok(ordered)
    }

    fn bits(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<Vec<u32>, String> {
        match arm {
            Arm::Production => {
                let values = fixture.production_output.to_cpu(&runtime.ctx.stream)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("production output download sync: {error:?}"))?;
                Ok(values.iter().map(|value| value.to_bits()).collect())
            }
            Arm::GenericNt => fixture
                .generic_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            candidate if candidate.is_candidate() => fixture
                .candidate_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            _ => unreachable!(),
        }
    }

    fn validate_scratch_transpose(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let (_, rows, cols) = DIMS;
        let input = fixture.b.expected_active_bits();
        let scratch = fixture
            .scratch
            .active_bits(&runtime.ctx.stream, "transpose scratch")?;
        for row in 0..rows {
            for col in 0..cols {
                let expected = input[row * cols + col];
                let actual = scratch[col * rows + row];
                if actual != expected {
                    return Err(format!(
                        "transpose scratch differs at input ({row},{col}): actual=0x{actual:08x} expected=0x{expected:08x}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_ada_inputs(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        fixture.a.validate_unchanged(&runtime.ctx.stream, "Ada A")?;
        fixture.b.validate_unchanged(&runtime.ctx.stream, "Ada B")?;
        let production_a = fixture.production_a.to_cpu(&runtime.ctx.stream)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("Ada production A download sync: {error:?}"))?;
        if production_a
            .iter()
            .zip(&fixture.production_a_expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("Ada production A changed".into());
        }
        Ok(())
    }

    fn exceptional_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let mut a = vec![0.0; m * n];
        for row in 0..m {
            a[row * n] = 1.0;
        }
        let mut b = vec![0.0; k_out * n];
        for (column, bits) in [
            0x8000_0000,
            0x0000_0001,
            0x7f80_0000,
            0x7fc0_1234,
            0xff80_0000,
        ]
        .into_iter()
        .enumerate()
        {
            b[column * n] = f32::from_bits(bits);
        }
        new_fixture_with_values(runtime, a, b)
    }

    fn validate_exceptional_oracle(bits: &[u32]) -> Result<(), String> {
        let (_, k_out, _) = DIMS;
        for row in [0, DIMS.0 - 1] {
            let base = row * k_out;
            if bits[base] != 0x0000_0000
                || bits[base + 1] != 0x0000_0001
                || bits[base + 2] != 0x7f80_0000
                || !f32::from_bits(bits[base + 3]).is_nan()
                || bits[base + 4] != 0xff80_0000
            {
                return Err(format!(
                    "isolated exceptional outputs changed at row {row}: {:08x?}",
                    &bits[base..base + 5]
                ));
            }
        }
        Ok(())
    }

    struct ResourceContract {
        threads: u32,
        static_shared: usize,
        dynamic_shared: usize,
        register_cap: i32,
        minimum_occupancy: u32,
    }

    fn check_resources(kernel: &Kernel, contract: ResourceContract) -> Result<(), String> {
        let threads = kernel.config.block_dim.0 * kernel.config.block_dim.1;
        let registers = kernel
            .function
            .num_regs()
            .map_err(|error| format!("regs: {error:?}"))?;
        let local = kernel
            .function
            .local_size_bytes()
            .map_err(|error| format!("local: {error:?}"))?;
        let static_shared = kernel
            .function
            .shared_size_bytes()
            .map_err(|error| format!("static shared: {error:?}"))?;
        let occupancy = kernel
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                kernel.config.shared_mem_bytes as usize,
                None,
            )
            .map_err(|error| format!("occupancy: {error:?}"))?;
        eprintln!(
            "nt_d768_out resource symbol={} threads={} registers={} local_bytes={} static_shared_bytes={} dynamic_shared_bytes={} active_blocks={}",
            kernel.symbol,
            threads,
            registers,
            local,
            static_shared,
            kernel.config.shared_mem_bytes,
            occupancy
        );
        if threads != contract.threads
            || static_shared as usize != contract.static_shared
            || kernel.config.shared_mem_bytes as usize != contract.dynamic_shared
            || local != 0
            || registers > contract.register_cap
            || occupancy < contract.minimum_occupancy
        {
            return Err(format!(
                "{} resource contract failed: expected threads={} static={} dynamic={} registers<={} occupancy>={}",
                kernel.symbol,
                contract.threads,
                contract.static_shared,
                contract.dynamic_shared,
                contract.register_cap,
                contract.minimum_occupancy
            ));
        }
        Ok(())
    }

    fn check_ada_resources(
        kernel: &Kernel,
        expected_threads: u32,
        expected_static_shared: usize,
        expected_dynamic_shared: usize,
    ) -> Result<(), String> {
        let threads = kernel.config.block_dim.0 * kernel.config.block_dim.1;
        let registers = kernel
            .function
            .num_regs()
            .map_err(|error| format!("{} regs: {error:?}", kernel.symbol))?;
        let local = kernel
            .function
            .local_size_bytes()
            .map_err(|error| format!("{} local: {error:?}", kernel.symbol))?;
        let static_shared = kernel
            .function
            .shared_size_bytes()
            .map_err(|error| format!("{} static shared: {error:?}", kernel.symbol))?;
        let max_threads = kernel
            .function
            .max_threads_per_block()
            .map_err(|error| format!("{} max threads: {error:?}", kernel.symbol))?;
        let occupancy = kernel
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                kernel.config.shared_mem_bytes as usize,
                None,
            )
            .map_err(|error| format!("{} occupancy: {error:?}", kernel.symbol))?;
        println!(
            "{{\"schema\":\"MambaBiScalarNtAdaDiscoveryResourceV1\",\"symbol\":\"{}\",\"threads\":{threads},\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{},\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":1}}",
            kernel.symbol, kernel.config.shared_mem_bytes,
        );
        if registers <= 0
            || local != 0
            || static_shared as usize != expected_static_shared
            || kernel.config.shared_mem_bytes as usize != expected_dynamic_shared
            || threads != expected_threads
            || max_threads < expected_threads as i32
            || occupancy < 1
        {
            return Err(format!(
                "{} Ada resource floor failed: threads={threads}/{expected_threads} registers={registers} local={local} static={static_shared}/{expected_static_shared} dynamic={}/{expected_dynamic_shared} max_threads={max_threads} occupancy={occupancy}/1",
                kernel.symbol, kernel.config.shared_mem_bytes,
            ));
        }
        Ok(())
    }

    fn validate_physical_production(runtime: &Runtime) -> Result<(), String> {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let qualified = qualify_physical_launch(&runtime.ctx, request)?;
        eprintln!(
            "nt_d768_out production nodes={} digest={:02x?}",
            qualified.evidence().launch_count(),
            qualified.evidence().launch_digest()
        );
        for (index, node) in qualified.evidence().nodes().iter().enumerate() {
            eprintln!(
                "nt_d768_out production node={} symbol={} grid={:?} block={:?} args={:02x?}",
                index,
                node.symbol,
                node.launch.grid_dim,
                node.launch.block_dim,
                node.launch.arguments_digest
            );
        }
        let evidence = qualified.evidence();
        if !evidence.eager_graph_equal() {
            return Err("production eager/graph physical nodes differ".into());
        }
        let nodes = evidence.nodes();
        if nodes.len() != 2 || evidence.launch_count() != 2 {
            return Err(format!(
                "production d768-out route must contain two nodes, found {}",
                nodes.len()
            ));
        }
        if [nodes[0].symbol, nodes[1].symbol] != [PROMOTED_TRANSPOSE, M64_SYMBOL] {
            return Err(format!(
                "production d768-out node order changed: [{}, {}]",
                nodes[0].symbol, nodes[1].symbol
            ));
        }
        let expected = [
            ((24, 48, 1), (32, 16, 1), 0),
            ((768, 1, 1), (128, 1, 1), M64_SHARED as u32),
        ];
        for (node, (grid, block, shared)) in nodes.iter().zip(expected) {
            if node.module_kind != ModuleKind::TriadScalar
                || node.logical_op != ResolvedGemmOp::Nt
                || node.shape != DIMS
                || node.strides != (768, 768, 1_536)
                || (
                    node.launch.grid_dim,
                    node.launch.block_dim,
                    node.launch.shared_mem_bytes,
                ) != (grid, block, shared)
                || node.launch.arguments_digest == [0; 32]
            {
                return Err(format!("production d768-out node changed: {node:?}"));
            }
        }
        if nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest
            || evidence.launch_digest() == [0; 32]
        {
            return Err("production d768-out physical identity digest is invalid".into());
        }
        qualified.validate_red_zones(&runtime.ctx)?;
        Ok(())
    }

    fn validate_ada_actual_auto(runtime: &Runtime) -> Result<(), String> {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let qualified = qualify_physical_launch(&runtime.ctx, request)?;
        let evidence = qualified.evidence();
        let nodes = evidence.nodes();
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || evidence.single_launch_symbol() != Some(GENERIC_NT_SYMBOL)
            || nodes.len() != 1
        {
            return Err(format!(
                "Ada actual AUTO is not the expected one-node portable exact NT route: {evidence:?}"
            ));
        }
        let node = &nodes[0];
        if node.module_kind != ModuleKind::TriadScalar
            || node.logical_op != ResolvedGemmOp::Nt
            || node.shape != DIMS
            || node.strides != (768, 768, 1_536)
            || (
                node.launch.grid_dim,
                node.launch.block_dim,
                node.launch.shared_mem_bytes,
            ) != ((192, 1, 1), (256, 1, 1), NT_SHARED as u32)
            || node.launch.arguments_digest == [0; 32]
            || evidence.launch_digest() == [0; 32]
        {
            return Err(format!(
                "Ada actual AUTO physical identity changed: {node:?}"
            ));
        }
        println!(
            "{{\"schema\":\"MambaBiScalarNtAdaAutoIdentityV1\",\"shape\":[2048,1536,768],\"symbol\":\"{}\",\"module\":\"TriadScalar\",\"grid\":[192,1,1],\"block\":[256,1,1],\"dynamic_shared_bytes\":{NT_SHARED},\"launch_digest\":\"{:02x?}\"}}",
            node.symbol,
            evidence.launch_digest(),
        );
        qualified.validate_red_zones(&runtime.ctx)?;
        Ok(())
    }

    #[test]
    fn candidate_identity_gate_rejects_count_order_zero_and_collisions() {
        let transpose = CandidateNodeIdentity {
            symbol: TRANSPOSE_32X16.into(),
            grid_dim: (24, 48, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
            arguments_digest: [1; 32],
        };
        let m64 = CandidateNodeIdentity {
            symbol: M64_SYMBOL.into(),
            grid_dim: (768, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: M64_SHARED as u32,
            arguments_digest: [2; 32],
        };
        assert!(
            validate_candidate_identity(&[transpose.clone(), m64.clone()], Arm::Transpose16)
                .is_ok()
        );
        assert!(
            validate_candidate_identity(std::slice::from_ref(&transpose), Arm::Transpose16)
                .is_err()
        );
        assert!(
            validate_candidate_identity(&[m64.clone(), transpose.clone()], Arm::Transpose16)
                .is_err()
        );
        let mut zero = transpose.clone();
        zero.arguments_digest = [0; 32];
        assert!(validate_candidate_identity(&[zero, m64.clone()], Arm::Transpose16).is_err());
        let mut collision = m64.clone();
        collision.arguments_digest = transpose.arguments_digest;
        assert!(validate_candidate_identity(&[transpose, collision], Arm::Transpose16).is_err());

        let mutations: [fn(&mut [CandidateNodeIdentity; 2]); 6] = [
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].grid_dim.0 += 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].block_dim.1 -= 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].shared_mem_bytes += 4,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].grid_dim.0 += 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].block_dim.0 -= 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].shared_mem_bytes -= 4,
        ];
        for mutate in mutations {
            let mut nodes = [
                CandidateNodeIdentity {
                    symbol: TRANSPOSE_32X16.into(),
                    grid_dim: (24, 48, 1),
                    block_dim: (32, 16, 1),
                    shared_mem_bytes: 0,
                    arguments_digest: [1; 32],
                },
                CandidateNodeIdentity {
                    symbol: M64_SYMBOL.into(),
                    grid_dim: (768, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: M64_SHARED as u32,
                    arguments_digest: [2; 32],
                },
            ];
            mutate(&mut nodes);
            assert!(validate_candidate_identity(&nodes, Arm::Transpose16).is_err());
        }

        let fixed = CandidateNodeIdentity {
            symbol: FIXED_COPYPLAN_SYMBOL.into(),
            grid_dim: (768, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
            arguments_digest: [2; 32],
        };
        let transpose = CandidateNodeIdentity {
            symbol: PROMOTED_TRANSPOSE.into(),
            grid_dim: (24, 48, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
            arguments_digest: [1; 32],
        };
        validate_candidate_identity_with_transpose16(
            &[transpose.clone(), fixed.clone()],
            Arm::Transpose16FixedCopyPlan,
            PROMOTED_TRANSPOSE,
        )
        .unwrap();
        assert!(
            validate_candidate_identity_with_transpose16(
                &[transpose, m64],
                Arm::Transpose16FixedCopyPlan,
                PROMOTED_TRANSPOSE,
            )
            .is_err()
        );
    }

    #[test]
    fn post_promotion_performance_gate_separates_speedup_parity_and_t16_win() {
        assert!(
            validate_performance_gate(
                Arm::GenericNt,
                Arm::Production,
                "test",
                BASELINE_MIN_P05_SPEEDUP,
                BASELINE_MIN_P50_SPEEDUP,
                1.70,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::GenericNt,
                Arm::Production,
                "test",
                BASELINE_MIN_P05_SPEEDUP - 0.001,
                BASELINE_MIN_P50_SPEEDUP,
                1.70,
            )
            .is_err()
        );

        assert!(
            validate_performance_gate(Arm::Production, Arm::Transpose16, "test", 0.99, 1.0, 1.01,)
                .is_ok()
        );
        for (p05, p50, p95) in [
            (PARITY_MIN_P05 - 0.001, 1.0, 1.01),
            (0.99, PARITY_MIN_P50 - 0.001, 1.01),
            (0.99, PARITY_MAX_P50 + 0.001, 1.01),
            (0.99, 1.0, PARITY_MAX_P95 + 0.001),
            (1.45, 1.50, 1.55),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::Production,
                    Arm::Transpose16,
                    "test",
                    p05,
                    p50,
                    p95,
                )
                .is_err()
            );
        }

        assert!(
            validate_performance_gate(
                Arm::Transpose32,
                Arm::Transpose16,
                "test",
                TRANSPOSE16_MIN_P05_SPEEDUP,
                TRANSPOSE16_MIN_P50_SPEEDUP,
                1.03,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::Transpose32,
                Arm::Transpose16,
                "test",
                TRANSPOSE16_MIN_P05_SPEEDUP,
                TRANSPOSE16_MIN_P50_SPEEDUP - 0.001,
                1.03,
            )
            .is_err()
        );
        assert!(
            validate_performance_gate(Arm::Production, Arm::GenericNt, "test", 1.0, 1.0, 1.0,)
                .is_err()
        );
    }

    #[test]
    fn qualified_environment_rejects_scalar_compiler_and_artifact_mismatches() {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: compiler.invocation_digest,
            artifact_digest: [5; 32],
        };
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, artifact).is_ok()
        );
        assert!(
            validate_qualified_environment((8, 9), 128, "compute_89", compiler, artifact).is_err()
        );
        let mut wrong_compiler = compiler;
        wrong_compiler.nvrtc_version = (13, 1);
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", wrong_compiler, artifact)
                .is_err()
        );
        let mut unknown_library = compiler;
        unknown_library.nvrtc_library_known = false;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", unknown_library, artifact)
                .is_err()
        );
        let compiler_mutations: [fn(&mut CompilerIdentity); 8] = [
            |identity: &mut CompilerIdentity| identity.source_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.invocation_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.header_manifest_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.nvrtc_library_domain = [0; 32],
            |identity: &mut CompilerIdentity| identity.composer_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.compiler_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.numeric_abi_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.schedule_revision ^= 1,
        ];
        for mutate in compiler_mutations {
            let mut mutated = compiler;
            mutate(&mut mutated);
            assert!(
                validate_qualified_environment((12, 0), 170, "compute_120", mutated, artifact)
                    .is_err()
            );
        }
        let mut wrong_artifact = artifact;
        wrong_artifact.compile_key = [9; 32];
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.artifact_kind = ArtifactKind::Cubin;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.module_kind = ModuleKind::TriadSm80;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.artifact_digest = [0; 32];
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );

        let mut ada_compiler = compiler;
        ada_compiler.target = CudaTarget::new("sm_89").unwrap();
        assert!(
            validate_ada_qualified_environment((8, 9), 142, "sm_89", ada_compiler, artifact,)
                .is_ok()
        );
        assert!(
            validate_ada_qualified_environment((8, 9), 142, "compute_89", ada_compiler, artifact,)
                .is_err()
        );
        let mut wrong_ada_compiler = ada_compiler;
        wrong_ada_compiler.target = CudaTarget::new("compute_89").unwrap();
        assert!(
            validate_ada_qualified_environment((8, 9), 142, "sm_89", wrong_ada_compiler, artifact,)
                .is_err()
        );
    }

    fn parsed_registers(resource_block: &str) -> usize {
        let (_, tail) = resource_block
            .split_once("Used ")
            .expect("ptxas register line");
        tail.split_whitespace()
            .next()
            .expect("ptxas register count")
            .parse()
            .expect("numeric ptxas register count")
    }

    #[test]
    #[ignore = "requires CUDA NVRTC and ptxas but launches no GPU work"]
    fn d768_out_sources_compile_with_bounded_sm89_and_sm120_resources() {
        for (arch, sm, contracts) in [
            (
                "compute_89",
                "sm_89",
                [
                    (GENERIC_NT_SYMBOL, 128, 0),
                    (M64_SYMBOL, 120, 0),
                    (TRANSPOSE_32X32, 12, TRANSPOSE_STATIC_SHARED),
                    (TRANSPOSE_32X16, 18, TRANSPOSE_STATIC_SHARED),
                ],
            ),
            (
                "compute_120",
                "sm_120",
                [
                    (GENERIC_NT_SYMBOL, 123, 0),
                    (M64_SYMBOL, 101, 0),
                    (TRANSPOSE_32X32, 12, TRANSPOSE_STATIC_SHARED),
                    (TRANSPOSE_32X16, 24, TRANSPOSE_STATIC_SHARED),
                ],
            ),
        ] {
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some(arch),
                options: vec![
                    "--fmad=true".to_owned(),
                    "--extra-device-vectorization".to_owned(),
                    "-DNDEBUG".to_owned(),
                    "-DGEMM_BI_GROUP_M=16".to_owned(),
                ],
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
                .unwrap_or_else(|error| panic!("{arch} compile failed: {error:?}"))
                .to_src();
            assert!(ptx.contains("fma.rn.f32"));
            let stem = format!("mamba-rs-d768-out-{}-{sm}", std::process::id());
            let ptx_path = std::env::temp_dir().join(format!("{stem}.ptx"));
            let cubin_path = std::env::temp_dir().join(format!("{stem}.cubin"));
            std::fs::write(&ptx_path, &ptx).expect("write temporary PTX");
            let ptxas = std::env::var("CUDA_HOME")
                .map(|root| std::path::PathBuf::from(root).join("bin/ptxas"))
                .unwrap_or_else(|_| "ptxas".into());
            let output = std::process::Command::new(ptxas)
                .arg("--verbose")
                .arg(format!("--gpu-name={sm}"))
                .arg(&ptx_path)
                .arg("--output-file")
                .arg(&cubin_path)
                .output()
                .unwrap_or_else(|error| panic!("launch ptxas for {sm}: {error}"));
            let log = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.status.success(), "ptxas {sm} failed:\n{log}");
            for (symbol, register_cap, static_shared) in contracts {
                let marker = format!("Compiling entry function '{symbol}'");
                let block = log
                    .split_once(&marker)
                    .unwrap_or_else(|| panic!("{sm} resource log lost {symbol}"))
                    .1
                    .split("Compiling entry function '")
                    .next()
                    .expect("resource block");
                assert!(
                    block
                        .contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"),
                    "{sm} {symbol} local/spill regression:\n{block}"
                );
                assert!(
                    parsed_registers(block) <= register_cap,
                    "{sm} {symbol} exceeds {register_cap} registers:\n{block}"
                );
                if static_shared == 0 {
                    assert!(!block.contains("bytes smem"), "{sm} {symbol}:\n{block}");
                } else {
                    assert!(
                        block.contains(&format!("{static_shared} bytes smem")),
                        "{sm} {symbol}:\n{block}"
                    );
                }
            }
            let _ = std::fs::remove_file(ptx_path);
            let _ = std::fs::remove_file(cubin_path);
        }
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM GPU"]
    fn d768_out_transpose_candidates_are_exact_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        validate_physical_production(&runtime)?;
        check_resources(
            &runtime.generic_nt,
            ResourceContract {
                threads: 256,
                static_shared: 0,
                dynamic_shared: NT_SHARED,
                register_cap: 123,
                minimum_occupancy: 2,
            },
        )?;
        check_resources(
            &runtime.m64,
            ResourceContract {
                threads: 128,
                static_shared: 0,
                dynamic_shared: M64_SHARED,
                register_cap: 103,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.transpose32,
            ResourceContract {
                threads: 1_024,
                static_shared: TRANSPOSE_STATIC_SHARED,
                dynamic_shared: 0,
                register_cap: 14,
                minimum_occupancy: 1,
            },
        )?;
        check_resources(
            &runtime.transpose16,
            ResourceContract {
                threads: 512,
                static_shared: TRANSPOSE_STATIC_SHARED,
                dynamic_shared: 0,
                register_cap: 24,
                minimum_occupancy: 2,
            },
        )?;
        let mut fixture = new_fixture(&runtime)?;
        launch_arm(&runtime, &mut fixture, Arm::GenericNt)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("generic sync: {error:?}"))?;
        let exact = bits(&runtime, &fixture, Arm::GenericNt)?;

        for arm in [Arm::Transpose32, Arm::Transpose16] {
            let transpose = match arm {
                Arm::Transpose32 => &runtime.transpose32,
                Arm::Transpose16 => &runtime.transpose16,
                _ => unreachable!(),
            };
            eprintln!(
                "candidate route arm={} nodes=[{},{}]",
                arm.name(),
                transpose.symbol,
                runtime.m64.symbol
            );
            for repeat in 0..3 {
                launch_arm(&runtime, &mut fixture, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} eager: {error:?}"))?;
                if bits(&runtime, &fixture, arm)? != exact {
                    return Err(format!(
                        "{arm:?} eager repeat {repeat} differs from ascending generic NT"
                    ));
                }
                validate_scratch_transpose(&runtime, &fixture)?;
            }
            let graph = capture_arm(&runtime, &mut fixture, arm)?;
            let eager_identity = eager_candidate_identity(&runtime, &fixture, arm);
            validate_candidate_identity(&eager_identity, arm)?;
            let graph_identity = graph_candidate_identity(&graph, arm)?;
            if eager_identity != graph_identity {
                return Err(format!(
                    "candidate eager/graph physical identity differs: eager={eager_identity:?} graph={graph_identity:?}"
                ));
            }
            eprintln!("candidate eager/graph identity={graph_identity:?}");
            for repeat in 0..3 {
                graph
                    .launch()
                    .map_err(|error| format!("{arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} graph sync: {error:?}"))?;
                if bits(&runtime, &fixture, arm)? != exact {
                    return Err(format!("{arm:?} graph repeat {repeat} differs from eager"));
                }
                validate_scratch_transpose(&runtime, &fixture)?;
            }
        }

        launch_arm(&runtime, &mut fixture, Arm::Production)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("production sync: {error:?}"))?;
        let production = bits(&runtime, &fixture, Arm::Production)?;
        if production != exact {
            return Err("production output differs from ascending generic NT".into());
        }
        for arm in [Arm::GenericNt, Arm::Production] {
            let reference = if arm == Arm::GenericNt {
                &exact
            } else {
                &production
            };
            let graph = capture_arm(&runtime, &mut fixture, arm)?;
            for repeat in 0..3 {
                launch_arm(&runtime, &mut fixture, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} repeat: {error:?}"))?;
                if &bits(&runtime, &fixture, arm)? != reference {
                    return Err(format!("{arm:?} eager repeat {repeat} changed"));
                }
                graph
                    .launch()
                    .map_err(|error| format!("{arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} graph sync: {error:?}"))?;
                if &bits(&runtime, &fixture, arm)? != reference {
                    return Err(format!("{arm:?} graph repeat {repeat} changed"));
                }
            }
        }
        fixture.a.validate_unchanged(&runtime.ctx.stream, "A")?;
        fixture.b.validate_unchanged(&runtime.ctx.stream, "B")?;
        let production_a = fixture.production_a.to_cpu(&runtime.ctx.stream)?;
        if production_a
            .iter()
            .zip(&fixture.production_a_expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("production A changed".into());
        }

        let mut exceptional = exceptional_fixture(&runtime)?;
        launch_arm(&runtime, &mut exceptional, Arm::GenericNt)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("exceptional generic sync: {error:?}"))?;
        let exceptional_exact = bits(&runtime, &exceptional, Arm::GenericNt)?;
        validate_exceptional_oracle(&exceptional_exact)?;
        for arm in [
            Arm::Production,
            Arm::GenericNt,
            Arm::Transpose32,
            Arm::Transpose16,
        ] {
            launch_arm(&runtime, &mut exceptional, arm)?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("exceptional {arm:?} warmup sync: {error:?}"))?;
            if bits(&runtime, &exceptional, arm)? != exceptional_exact {
                return Err(format!(
                    "exceptional {arm:?} eager warmup differs from generic NT"
                ));
            }
            if arm.is_candidate() {
                validate_scratch_transpose(&runtime, &exceptional)?;
            }
            let graph = capture_arm(&runtime, &mut exceptional, arm)?;
            let mut repeated: Option<Vec<u32>> = None;
            for repeat in 0..3 {
                launch_arm(&runtime, &mut exceptional, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("exceptional {arm:?} eager sync: {error:?}"))?;
                let eager = bits(&runtime, &exceptional, arm)?;
                if eager != exceptional_exact {
                    return Err(format!(
                        "exceptional {arm:?} eager repeat {repeat} differs from generic NT"
                    ));
                }
                if arm.is_candidate() {
                    validate_scratch_transpose(&runtime, &exceptional)?;
                }
                graph
                    .launch()
                    .map_err(|error| format!("exceptional {arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("exceptional {arm:?} graph sync: {error:?}"))?;
                let graph_bits = bits(&runtime, &exceptional, arm)?;
                if graph_bits != eager {
                    return Err(format!(
                        "exceptional {arm:?} graph repeat {repeat} differs from eager"
                    ));
                }
                if repeated.as_ref().is_some_and(|prior| prior != &eager) {
                    return Err(format!("exceptional {arm:?} repeated bits changed"));
                }
                repeated.get_or_insert(eager);
                if arm.is_candidate() {
                    validate_scratch_transpose(&runtime, &exceptional)?;
                }
            }
        }
        exceptional
            .a
            .validate_unchanged(&runtime.ctx.stream, "exceptional A")?;
        exceptional
            .b
            .validate_unchanged(&runtime.ctx.stream, "exceptional B")?;
        let exceptional_production_a = exceptional.production_a.to_cpu(&runtime.ctx.stream)?;
        if exceptional_production_a
            .iter()
            .zip(&exceptional.production_a_expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("exceptional production A changed".into());
        }
        Ok(())
    }

    fn percentile(values: &[f64], fraction: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[index]
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("start: {error:?}"))?;
        for _ in 0..iterations {
            launch_arm(runtime, fixture, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("end: {error:?}"))?;
        Ok(f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("elapsed: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64)
    }

    fn calibrated_iterations(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<usize, String> {
        let pilot = measure(runtime, fixture, arm, 3)?;
        Ok((TARGET_WINDOW_US / pilot).round().clamp(3.0, 500.0) as usize)
    }

    fn validate_performance_gate(
        baseline: Arm,
        candidate: Arm,
        order: &str,
        speedup_p05: f64,
        speedup_p50: f64,
        speedup_p95: f64,
    ) -> Result<(), String> {
        match (baseline, candidate) {
            (Arm::GenericNt, Arm::Production) => {
                if speedup_p05 < BASELINE_MIN_P05_SPEEDUP || speedup_p50 < BASELINE_MIN_P50_SPEEDUP
                {
                    return Err(format!(
                        "production speedup over generic NT in {order} failed: p05={speedup_p05:.9} p50={speedup_p50:.9}; required p05>={BASELINE_MIN_P05_SPEEDUP:.2} p50>={BASELINE_MIN_P50_SPEEDUP:.2}"
                    ));
                }
            }
            (Arm::Production, Arm::Transpose16) => {
                if speedup_p05 < PARITY_MIN_P05
                    || !(PARITY_MIN_P50..=PARITY_MAX_P50).contains(&speedup_p50)
                    || speedup_p95 > PARITY_MAX_P95
                {
                    return Err(format!(
                        "production/direct T16 parity gate failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9} p95={speedup_p95:.9}; required p05>={PARITY_MIN_P05:.3}, {PARITY_MIN_P50:.3}<=p50<={PARITY_MAX_P50:.3}, p95<={PARITY_MAX_P95:.3}"
                    ));
                }
            }
            (Arm::Transpose32, Arm::Transpose16) => {
                if speedup_p05 < TRANSPOSE16_MIN_P05_SPEEDUP
                    || speedup_p50 < TRANSPOSE16_MIN_P50_SPEEDUP
                {
                    return Err(format!(
                        "Transpose16 gate failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9}"
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "unsupported d768-out performance pair: {} -> {}",
                    baseline.name(),
                    candidate.name()
                ));
            }
        }
        Ok(())
    }

    fn paired(
        runtime: &Runtime,
        fixture: &mut Fixture,
        baseline: Arm,
        candidate: Arm,
        windows: usize,
    ) -> Result<(), String> {
        let baseline_iterations = calibrated_iterations(runtime, fixture, baseline)?;
        let candidate_iterations = calibrated_iterations(runtime, fixture, candidate)?;
        for (order, baseline_first) in [("ABBA", true), ("BAAB", false)] {
            let mut baseline_samples = Vec::with_capacity(windows);
            let mut candidate_samples = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (b0, b1, c0, c1) = if baseline_first {
                    let b0 = measure(runtime, fixture, baseline, baseline_iterations)?;
                    let c0 = measure(runtime, fixture, candidate, candidate_iterations)?;
                    let c1 = measure(runtime, fixture, candidate, candidate_iterations)?;
                    let b1 = measure(runtime, fixture, baseline, baseline_iterations)?;
                    (b0, b1, c0, c1)
                } else {
                    let c0 = measure(runtime, fixture, candidate, candidate_iterations)?;
                    let b0 = measure(runtime, fixture, baseline, baseline_iterations)?;
                    let b1 = measure(runtime, fixture, baseline, baseline_iterations)?;
                    let c1 = measure(runtime, fixture, candidate, candidate_iterations)?;
                    (b0, b1, c0, c1)
                };
                let baseline_us = 0.5 * (b0 + b1);
                let candidate_us = 0.5 * (c0 + c1);
                baseline_samples.push(baseline_us);
                candidate_samples.push(candidate_us);
                ratios.push(baseline_us / candidate_us);
            }
            let baseline_p50 = percentile(&baseline_samples, 0.50);
            let candidate_p50 = percentile(&candidate_samples, 0.50);
            let speedup_p05 = percentile(&ratios, 0.05);
            let speedup_p50 = percentile(&ratios, 0.50);
            let speedup_p95 = percentile(&ratios, 0.95);
            eprintln!(
                "nt_d768_out baseline={} candidate={} order={} windows={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
                baseline.name(),
                candidate.name(),
                order,
                windows,
                baseline_p50,
                candidate_p50,
                speedup_p05,
                speedup_p50,
                speedup_p95,
            );
            validate_performance_gate(
                baseline,
                candidate,
                order,
                speedup_p05,
                speedup_p50,
                speedup_p95,
            )?;
        }
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum AdaPath {
        Eager,
        Graph,
    }

    impl AdaPath {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    fn launch_ada_path(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: &CudaGraph,
        path: AdaPath,
    ) -> Result<(), String> {
        match path {
            AdaPath::Eager => launch_arm(runtime, fixture, arm),
            AdaPath::Graph => graph
                .launch()
                .map_err(|error| format!("launch Ada {} graph: {error:?}", arm.name())),
        }
    }

    fn check_ada_observation(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: &CudaGraph,
        path: AdaPath,
        expected: Option<&[u32]>,
    ) -> Result<Vec<u32>, String> {
        reset_ada_arm(runtime, fixture, arm)?;
        launch_ada_path(runtime, fixture, arm, graph, path)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("Ada {} {} sync: {error:?}", arm.name(), path.name()))?;
        let actual = bits(runtime, fixture, arm)?;
        if let Some(expected) = expected
            && actual != expected
        {
            let mismatch = actual
                .iter()
                .zip(expected)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(actual.len());
            return Err(format!(
                "Ada {} {} differs from actual AUTO at output {mismatch}",
                arm.name(),
                path.name()
            ));
        }
        if arm.is_candidate() {
            validate_scratch_transpose(runtime, fixture)?;
        }
        validate_ada_inputs(runtime, fixture)?;
        Ok(actual)
    }

    fn prepare_ada_pair(
        runtime: &Runtime,
    ) -> Result<(Fixture, CudaGraph, CudaGraph, Vec<u32>), String> {
        prepare_ada_pair_for(runtime, Arm::Transpose16)
    }

    fn prepare_ada_pair_for(
        runtime: &Runtime,
        candidate_arm: Arm,
    ) -> Result<(Fixture, CudaGraph, CudaGraph, Vec<u32>), String> {
        validate_ada_actual_auto(runtime)?;
        check_ada_resources(&runtime.generic_nt, 256, 0, NT_SHARED)?;
        let inner = candidate_inner_kernel(runtime, candidate_arm)?;
        let contract = ada_copyplan_contract();
        if candidate_arm == Arm::Transpose16FixedCopyPlan {
            validate_ada_fixed_copyplan_environment(&runtime.ctx)?;
            validate_ada_copyplan_contract(contract)?;
            check_ada_resources(
                inner,
                contract.block.0,
                contract.static_shared,
                contract.dynamic_shared,
            )?;
        } else {
            check_ada_resources(inner, 128, 0, M64_SHARED)?;
        }
        check_ada_resources(&runtime.transpose16, 512, TRANSPOSE_STATIC_SHARED, 0)?;
        let mut fixture = new_ada_fixture(runtime)?;
        let auto_graph = capture_ada_actual_auto(runtime, &mut fixture)?;
        let candidate_graph = capture_arm(runtime, &mut fixture, candidate_arm)?;
        let eager_identity = eager_candidate_identity(runtime, &fixture, candidate_arm);
        validate_candidate_identity_with_transpose16(
            &eager_identity,
            candidate_arm,
            PROMOTED_TRANSPOSE,
        )?;
        let graph_identity = graph_candidate_identity_with_transpose16(
            &candidate_graph,
            candidate_arm,
            PROMOTED_TRANSPOSE,
        )?;
        if graph_identity != eager_identity {
            return Err(format!(
                "Ada candidate eager/graph identity differs: eager={eager_identity:?} graph={graph_identity:?}"
            ));
        }
        if candidate_arm == Arm::Transpose16FixedCopyPlan {
            let artifact = runtime.ctx.kernels.artifact_set_identity();
            println!(
                "{{\"schema\":\"MambaBiScalarNtAdaCandidateIdentityV2\",\"shape\":[2048,1536,768],\"candidate\":\"{}\",\"symbols\":[\"{PROMOTED_TRANSPOSE}\",\"{}\"],\"modules\":[\"TriadScalar\",\"Fixed\"],\"artifact_digests\":[\"{:02x?}\",\"{:02x?}\"],\"launch_count\":2,\"node_configs\":[[[24,48,1],[32,16,1],0],[[768,1,1],[128,1,1],{}]],\"argument_digests_nonzero_and_distinct\":true}}",
                candidate_arm.name(),
                inner.symbol,
                artifact.triad_scalar.artifact_digest,
                artifact.fixed.artifact_digest,
                inner.config.shared_mem_bytes,
            );
        } else {
            println!(
                "{{\"schema\":\"MambaBiScalarNtAdaCandidateIdentityV1\",\"shape\":[2048,1536,768],\"symbols\":[\"{PROMOTED_TRANSPOSE}\",\"{M64_SYMBOL}\"],\"launch_count\":2,\"node_configs\":[[[24,48,1],[32,16,1],0],[[768,1,1],[128,1,1],{M64_SHARED}]],\"argument_digests_nonzero_and_distinct\":true}}"
            );
        }

        let golden = check_ada_observation(
            runtime,
            &mut fixture,
            Arm::Production,
            &auto_graph,
            AdaPath::Eager,
            None,
        )?;
        for path in [AdaPath::Eager, AdaPath::Graph] {
            for repeat in 0..2 {
                for (arm, graph) in [
                    (Arm::Production, &auto_graph),
                    (candidate_arm, &candidate_graph),
                ] {
                    let actual = check_ada_observation(
                        runtime,
                        &mut fixture,
                        arm,
                        graph,
                        path,
                        Some(&golden),
                    )?;
                    println!(
                        "{{\"schema\":\"MambaBiScalarNtAdaBitsV1\",\"shape\":[2048,1536,768],\"arm\":\"{}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{},\"digest\":\"{:02x?}\"}}",
                        arm.name(),
                        path.name(),
                        actual.len(),
                        <[u8; 32]>::from(Sha256::digest(
                            actual
                                .iter()
                                .flat_map(|word| word.to_le_bytes())
                                .collect::<Vec<_>>()
                        )),
                    );
                }
            }
        }
        Ok((fixture, auto_graph, candidate_graph, golden))
    }

    fn measure_ada_observation(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: &CudaGraph,
        path: AdaPath,
        expected: &[u32],
    ) -> Result<f64, String> {
        reset_ada_arm(runtime, fixture, arm)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record Ada {} start: {error:?}", arm.name()))?;
        launch_ada_path(runtime, fixture, arm, graph, path)?;
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record Ada {} end: {error:?}", arm.name()))?;
        let elapsed_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure Ada {}: {error:?}", arm.name()))?,
        ) * 1_000.0;
        if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
            return Err(format!("invalid Ada {} sample {elapsed_us}", arm.name()));
        }
        let actual = bits(runtime, fixture, arm)?;
        if actual != expected {
            return Err(format!(
                "Ada {} {} timing observation changed output bits",
                arm.name(),
                path.name()
            ));
        }
        if arm.is_candidate() {
            validate_scratch_transpose(runtime, fixture)?;
        }
        validate_ada_inputs(runtime, fixture)?;
        Ok(elapsed_us)
    }

    fn json_raw_observations(observations: &[[f64; 4]]) -> String {
        format!(
            "[{}]",
            observations
                .iter()
                .map(|raw| format!("[{:.9},{:.9},{:.9},{:.9}]", raw[0], raw[1], raw[2], raw[3]))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn screen_ada_stratum(
        runtime: &Runtime,
        fixture: &mut Fixture,
        auto_graph: &CudaGraph,
        candidate_graph: &CudaGraph,
        expected: &[u32],
        path: AdaPath,
        order: AdaBracketOrder,
    ) -> Result<[f64; 2], String> {
        screen_ada_stratum_for(
            runtime,
            fixture,
            auto_graph,
            candidate_graph,
            expected,
            path,
            order,
            Arm::Transpose16,
        )
    }

    fn screen_ada_stratum_for(
        runtime: &Runtime,
        fixture: &mut Fixture,
        auto_graph: &CudaGraph,
        candidate_graph: &CudaGraph,
        expected: &[u32],
        path: AdaPath,
        order: AdaBracketOrder,
        candidate_arm: Arm,
    ) -> Result<[f64; 2], String> {
        for _ in 0..ADA_WARMUPS {
            measure_ada_observation(
                runtime,
                fixture,
                Arm::Production,
                auto_graph,
                path,
                expected,
            )?;
            measure_ada_observation(
                runtime,
                fixture,
                candidate_arm,
                candidate_graph,
                path,
                expected,
            )?;
        }
        let arms = match order {
            AdaBracketOrder::Abba => [
                (Arm::Production, auto_graph),
                (candidate_arm, candidate_graph),
                (candidate_arm, candidate_graph),
                (Arm::Production, auto_graph),
            ],
            AdaBracketOrder::Baab => [
                (candidate_arm, candidate_graph),
                (Arm::Production, auto_graph),
                (Arm::Production, auto_graph),
                (candidate_arm, candidate_graph),
            ],
        };
        let mut raw_observations = Vec::with_capacity(ADA_WINDOWS);
        let mut ratios = Vec::with_capacity(ADA_WINDOWS);
        for _ in 0..ADA_WINDOWS {
            let mut raw = [0.0; 4];
            for (index, (arm, graph)) in arms.into_iter().enumerate() {
                raw[index] = measure_ada_observation(runtime, fixture, arm, graph, path, expected)?;
            }
            ratios.push(ada_candidate_over_auto(raw, order));
            raw_observations.push(raw);
        }
        let p50 = ada_percentile(&ratios, 0.50).ok_or("invalid Ada p50 samples")?;
        let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid Ada p95 samples")?;
        let order_name = match order {
            AdaBracketOrder::Abba => "ABBA",
            AdaBracketOrder::Baab => "BAAB",
        };
        if candidate_arm == Arm::Transpose16FixedCopyPlan {
            println!(
                "{{\"schema\":\"MambaBiScalarNtAdaDiscoveryScreenV2\",\"op\":\"NT\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"{}\",\"candidate_symbols\":[\"{PROMOTED_TRANSPOSE}\",\"{}\"],\"candidate_modules\":[\"TriadScalar\",\"Fixed\"],\"comparator\":\"actual_auto\",\"comparator_symbol\":\"{GENERIC_NT_SYMBOL}\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{ADA_WINDOWS},\"warmups_per_arm\":{ADA_WARMUPS},\"logical_pipelines_per_observation\":1,\"candidate_nodes_per_observation\":2,\"auto_nodes_per_observation\":1,\"raw_observations_us\":{},\"ratio_direction\":\"candidate_over_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
                candidate_arm.name(),
                candidate_inner_kernel(runtime, candidate_arm)?.symbol,
                path.name(),
                json_raw_observations(&raw_observations),
            );
        } else {
            println!(
                "{{\"schema\":\"MambaBiScalarNtAdaDiscoveryScreenV1\",\"op\":\"NT\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"transpose16_plus_m64n64\",\"candidate_symbols\":[\"{PROMOTED_TRANSPOSE}\",\"{M64_SYMBOL}\"],\"comparator\":\"actual_auto\",\"comparator_symbol\":\"{GENERIC_NT_SYMBOL}\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{ADA_WINDOWS},\"warmups_per_arm\":{ADA_WARMUPS},\"logical_pipelines_per_observation\":1,\"candidate_nodes_per_observation\":2,\"auto_nodes_per_observation\":1,\"raw_observations_us\":{},\"ratio_direction\":\"candidate_over_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
                path.name(),
                json_raw_observations(&raw_observations),
            );
        }
        Ok([p50, p95])
    }

    #[test]
    #[ignore = "requires an exclusive CC8.9/142-SM CUDA13.2 Ada GPU"]
    fn ada_d768_out_transpose16_m64n64_is_exact_and_graph_stable() -> Result<(), String> {
        assert!(!cfg!(debug_assertions), "Ada discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let _pre = quiet.require_pre_context("scalar-nt-d768-out-ada-bits/pre-context")?;
        let runtime = new_ada_runtime()?;
        let _cohort = quiet.require_cohort("scalar-nt-d768-out-ada-bits/cohort")?;
        let (_fixture, _auto_graph, _candidate_graph, _golden) = prepare_ada_pair(&runtime)?;
        drop(runtime);
        quiet.verify_post_cohort("scalar-nt-d768-out-ada-bits/post")?;
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC8.9/142-SM CUDA13.2 Ada GPU"]
    fn ada_d768_out_transpose16_m64n64_discovery_once7() -> Result<(), String> {
        assert!(!cfg!(debug_assertions), "Ada discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let _pre = quiet.require_pre_context("scalar-nt-d768-out-ada-once7/pre-context")?;
        let runtime = new_ada_runtime()?;
        let _cohort = quiet.require_cohort("scalar-nt-d768-out-ada-once7/cohort")?;
        let (mut fixture, auto_graph, candidate_graph, golden) = prepare_ada_pair(&runtime)?;
        let mut strata = Vec::with_capacity(4);
        for path in [AdaPath::Eager, AdaPath::Graph] {
            for order in [AdaBracketOrder::Abba, AdaBracketOrder::Baab] {
                strata.push(screen_ada_stratum(
                    &runtime,
                    &mut fixture,
                    &auto_graph,
                    &candidate_graph,
                    &golden,
                    path,
                    order,
                )?);
            }
        }
        let retain = ada_retain_decision(&strata);
        let strata_json = format!(
            "[{}]",
            strata
                .iter()
                .map(|[p50, p95]| format!("[{p50:.9},{p95:.9}]"))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "{{\"schema\":\"MambaBiScalarNtAdaDiscoveryDecisionV1\",\"op\":\"NT\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"transpose16_plus_m64n64\",\"comparator\":\"actual_auto\",\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"strata_fields\":[\"ratio_p50\",\"ratio_p95\"],\"strata\":{strata_json},\"ratio_direction\":\"candidate_over_auto\",\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
            if retain {
                "advance_to_full_qualification"
            } else {
                "stop_no_retry"
            },
        );
        drop(fixture);
        drop(auto_graph);
        drop(candidate_graph);
        drop(runtime);
        quiet.verify_post_cohort("scalar-nt-d768-out-ada-once7/post")?;
        if !retain {
            return Err("Ada d768-out transpose16+M64N64 failed one or more <0.99 strata".into());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC8.9/142-SM CUDA13.2 Ada GPU"]
    fn ada_d768_out_transpose16_fixed_copyplan_discovery_once7() -> Result<(), String> {
        assert!(!cfg!(debug_assertions), "Ada discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let _pre =
            quiet.require_pre_context("scalar-nt-d768-out-copyplan-ada-once7/pre-context")?;
        let runtime = new_ada_copyplan_runtime()?;
        let _cohort = quiet.require_cohort("scalar-nt-d768-out-copyplan-ada-once7/cohort")?;
        let candidate = Arm::Transpose16FixedCopyPlan;
        let (mut fixture, auto_graph, candidate_graph, golden) =
            prepare_ada_pair_for(&runtime, candidate)?;
        let mut strata = Vec::with_capacity(4);
        for path in [AdaPath::Eager, AdaPath::Graph] {
            for order in [AdaBracketOrder::Abba, AdaBracketOrder::Baab] {
                strata.push(screen_ada_stratum_for(
                    &runtime,
                    &mut fixture,
                    &auto_graph,
                    &candidate_graph,
                    &golden,
                    path,
                    order,
                    candidate,
                )?);
            }
        }
        let retain = ada_retain_decision(&strata);
        let strata_json = format!(
            "[{}]",
            strata
                .iter()
                .map(|[p50, p95]| format!("[{p50:.9},{p95:.9}]"))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "{{\"schema\":\"MambaBiScalarNtAdaDiscoveryDecisionV2\",\"op\":\"NT\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"transpose32x16_plus_fixed_copyplan\",\"candidate_symbols\":[\"{PROMOTED_TRANSPOSE}\",\"{FIXED_COPYPLAN_SYMBOL}\"],\"candidate_modules\":[\"TriadScalar\",\"Fixed\"],\"comparator\":\"actual_auto\",\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"strata_fields\":[\"ratio_p50\",\"ratio_p95\"],\"strata\":{strata_json},\"ratio_direction\":\"candidate_over_auto\",\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
            if retain {
                "advance_to_full_qualification"
            } else {
                "stop_no_retry"
            },
        );
        drop(fixture);
        drop(auto_graph);
        drop(candidate_graph);
        drop(runtime);
        quiet.verify_post_cohort("scalar-nt-d768-out-copyplan-ada-once7/post")?;
        if !retain {
            return Err(
                "Ada d768-out transpose16+Fixed CopyPlan failed one or more <0.99 strata".into(),
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM GPU"]
    fn d768_out_transpose_tournament_abba_baab() -> Result<(), String> {
        let runtime = new_runtime()?;
        let mut fixture = new_fixture(&runtime)?;
        for arm in [
            Arm::Production,
            Arm::GenericNt,
            Arm::Transpose32,
            Arm::Transpose16,
        ] {
            for _ in 0..5 {
                launch_arm(&runtime, &mut fixture, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("warmup: {error:?}"))?;
        let mut screen_p50 = Vec::new();
        for arm in [Arm::Transpose32, Arm::Transpose16] {
            let iterations = calibrated_iterations(&runtime, &mut fixture, arm)?;
            let samples = (0..11)
                .map(|_| measure(&runtime, &mut fixture, arm, iterations))
                .collect::<Result<Vec<_>, _>>()?;
            let p50 = percentile(&samples, 0.50);
            eprintln!("nt_d768_out screen arm={} p50_us={p50:.6}", arm.name());
            screen_p50.push((arm, p50));
        }
        eprintln!("nt_d768_out candidate screen={screen_p50:?}");
        let winner = Arm::Transpose16;
        let windows = std::env::var("MAMBA_RS_NT_D768_OUT_WINDOWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(101);
        paired(
            &runtime,
            &mut fixture,
            Arm::GenericNt,
            Arm::Production,
            windows,
        )?;
        paired(&runtime, &mut fixture, Arm::Production, winner, windows)?;
        paired(&runtime, &mut fixture, Arm::Transpose32, winner, windows)?;
        Ok(())
    }
}
