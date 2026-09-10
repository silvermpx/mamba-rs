use std::path::PathBuf;

#[cfg(feature = "cuda")]
mod common;

const CANDIDATE_SOURCE_PATH: &str = "gemm_bi_scalar_nt_prism_experiment.cu";
const TRANSPOSE_SYMBOL: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_prism_m64n64_bk16_s2_v1";
const CANDIDATE1_SYMBOL: &str = "gemm_bi_nn_prism_m64n64_bk16_s2_occ5_experiment_v1";
const CANDIDATE2_SYMBOL: &str = "gemm_bi_nn_prism_m64n128_bk16_s2_experiment_v1";
const MIN_SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const MIN_P05_SPEEDUP: f64 = 1.005;
const MIN_P50_SPEEDUP: f64 = 1.01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSpec {
    symbol: &'static str,
    block: u32,
    grid: (u32, u32, u32),
    shared: u32,
    minimum_blocks: u32,
    sm89_register_cap: u32,
    sm120_register_cap: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeIdentity {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    arguments_digest: [u8; 32],
}

fn validate_two_node_identity(
    nodes: &[NodeIdentity],
    candidate: CandidateSpec,
) -> Result<(), String> {
    if nodes.len() != 2 {
        return Err(format!(
            "candidate has {} nodes instead of two",
            nodes.len()
        ));
    }
    let expected = [
        (TRANSPOSE_SYMBOL, (61, 12, 1), (32, 16, 1), 0),
        (
            candidate.symbol,
            candidate.grid,
            (candidate.block, 1, 1),
            candidate.shared,
        ),
    ];
    for (node, (symbol, grid, block, shared)) in nodes.iter().zip(expected) {
        if (node.symbol.as_str(), node.grid, node.block, node.shared)
            != (symbol, grid, block, shared)
        {
            return Err(format!("candidate node identity changed: {node:?}"));
        }
        if node.arguments_digest == [0; 32] {
            return Err(format!("{symbol} argument digest is zero"));
        }
    }
    if nodes[0].arguments_digest == nodes[1].arguments_digest {
        return Err("candidate argument digests collided".into());
    }
    Ok(())
}

const CANDIDATE1: CandidateSpec = CandidateSpec {
    symbol: CANDIDATE1_SYMBOL,
    block: 128,
    grid: (73, 6, 1),
    shared: 17_408,
    minimum_blocks: 5,
    sm89_register_cap: 102,
    sm120_register_cap: 102,
};

const CANDIDATE2: CandidateSpec = CandidateSpec {
    symbol: CANDIDATE2_SYMBOL,
    block: 256,
    grid: (73, 3, 1),
    shared: 25_600,
    minimum_blocks: 2,
    sm89_register_cap: 128,
    sm120_register_cap: 128,
};

fn candidate_source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(CANDIDATE_SOURCE_PATH)
}

fn candidate_source() -> Result<String, String> {
    std::fs::read_to_string(candidate_source_path())
        .map_err(|error| format!("read {}: {error}", candidate_source_path().display()))
}

fn kernel_parameter_count(source: &str, symbol: &str) -> Result<usize, String> {
    let marker = format!("void {symbol}(");
    let (_, tail) = source
        .split_once(&marker)
        .ok_or_else(|| format!("missing {symbol}"))?;
    let (parameters, _) = tail
        .split_once(") {")
        .ok_or_else(|| format!("missing {symbol} parameter terminator"))?;
    Ok(parameters
        .split(',')
        .filter(|parameter| !parameter.trim().is_empty())
        .count())
}

fn validate_candidate_source(source: &str) -> Result<(), String> {
    for helper in [
        "prism_issue_full_tile",
        "prism_issue_tail_tile",
        "prism_compute_tile",
        "prism_exact_body",
    ] {
        let arguments = kernel_parameter_count(source, helper)?;
        if arguments > 7 {
            return Err(format!(
                "{helper} exceeds the seven-argument function contract: {arguments}"
            ));
        }
    }
    for (spec, launch_bounds) in [
        (CANDIDATE1, "__launch_bounds__(128, 5)"),
        (CANDIDATE2, "__launch_bounds__(256, 2)"),
    ] {
        if source.matches(&format!("void {}(", spec.symbol)).count() != 1 {
            return Err(format!("{} must be defined exactly once", spec.symbol));
        }
        if kernel_parameter_count(source, spec.symbol)? != 5 {
            return Err(format!(
                "{} must keep the packed five-argument ABI",
                spec.symbol
            ));
        }
        let symbol_offset = source
            .find(&format!("void {}(", spec.symbol))
            .ok_or_else(|| format!("missing {}", spec.symbol))?;
        let prefix = &source[..symbol_offset];
        if !prefix.ends_with(&format!("{launch_bounds}\n")) {
            return Err(format!("{} launch bounds changed", spec.symbol));
        }
    }

    for required in [
        "PRISM_EXPERIMENT_M 4621",
        "PRISM_EXPERIMENT_N 384",
        "PRISM_EXPERIMENT_K 1928",
        "PRISM_EXPERIMENT_PADDED_K 1936",
        "PRISM_EXPERIMENT_K_TILES 121",
        "PRISM_EXPERIMENT_FULL_TILES 120",
        "int pid_m = blockIdx.x;",
        "int pid_n = blockIdx.y;",
        "prism_issue_tail_tile<BN, THREADS>(",
        "for (int tile = 0; tile < PRISM_EXPERIMENT_FULL_TILES; ++tile)",
        "for (int dot_index = 0; dot_index < BK; ++dot_index)",
        "__fmaf_rn(",
        "reinterpret_cast<float4*>(destination)[0] = output;",
        "TOTAL_SMEM_BYTES == 17408",
        "TOTAL_SMEM_BYTES == 25600",
    ] {
        if !source.contains(required) {
            return Err(format!(
                "candidate source lost required contract {required:?}"
            ));
        }
    }
    for forbidden in [
        "atomic",
        "mma.sync",
        "wgmma",
        "tcgen05",
        "split_k",
        "splitK",
        "params.alpha *",
        "params.beta *",
        "blockIdx.x /",
        "blockIdx.x %",
    ] {
        if source.contains(forbidden) {
            return Err(format!(
                "candidate source contains forbidden token {forbidden:?}"
            ));
        }
    }
    if source
        .chars()
        .any(|value| matches!(value, '\u{0400}'..='\u{04ff}'))
    {
        return Err("candidate source comments must remain English-only".into());
    }
    Ok(())
}

fn parse_windows(value: Result<String, std::env::VarError>) -> Result<usize, String> {
    let windows = match value {
        Ok(raw) => raw.parse::<usize>().map_err(|error| {
            format!("invalid MAMBA_RS_NT_PRISM_POST_TAG30_WINDOWS={raw:?}: {error}")
        })?,
        Err(std::env::VarError::NotPresent) => OFFICIAL_WINDOWS,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("MAMBA_RS_NT_PRISM_POST_TAG30_WINDOWS is not valid Unicode".into());
        }
    };
    if windows != MIN_SCREEN_WINDOWS && windows != OFFICIAL_WINDOWS {
        return Err(format!(
            "timing windows must be exactly {MIN_SCREEN_WINDOWS} or {OFFICIAL_WINDOWS}, found {windows}"
        ));
    }
    Ok(windows)
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("percentile requires samples".into());
    }
    if !fraction.is_finite() || !(0.0 < fraction && fraction <= 1.0) {
        return Err(format!("invalid percentile fraction {fraction}"));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("percentile samples must be finite and positive".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn paired_sample(
    baseline_first: f64,
    baseline_second: f64,
    candidate_first: f64,
    candidate_second: f64,
) -> Result<(f64, f64, f64), String> {
    if [
        baseline_first,
        baseline_second,
        candidate_first,
        candidate_second,
    ]
    .into_iter()
    .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("paired samples must be finite and positive".into());
    }
    let baseline = (baseline_first + baseline_second) * 0.5;
    let candidate = (candidate_first + candidate_second) * 0.5;
    let speedup = baseline / candidate;
    if [baseline, candidate, speedup]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("paired result must be finite and positive".into());
    }
    Ok((baseline, candidate, speedup))
}

fn validate_speedup_gate(
    order: &str,
    windows: usize,
    p05: f64,
    p50: f64,
    p95: f64,
) -> Result<(), String> {
    if !matches!(order, "ABBA" | "BAAB") {
        return Err(format!("unsupported timing order {order}"));
    }
    if windows != MIN_SCREEN_WINDOWS && windows != OFFICIAL_WINDOWS {
        return Err(format!("unsupported timing window count {windows}"));
    }
    if [p05, p50, p95]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("speedup percentiles must be finite and positive".into());
    }
    if p05 < MIN_P05_SPEEDUP || p50 < MIN_P50_SPEEDUP {
        return Err(format!(
            "candidate speedup failed: p05={p05:.9} p50={p50:.9}; required p05>={MIN_P05_SPEEDUP:.3} p50>={MIN_P50_SPEEDUP:.2}"
        ));
    }
    Ok(())
}

#[test]
fn prism_candidate_specs_are_exact_and_bounded() {
    assert_ne!(PRODUCTION_SYMBOL, CANDIDATE1.symbol);
    assert_ne!(PRODUCTION_SYMBOL, CANDIDATE2.symbol);
    assert_eq!(CANDIDATE1.grid, (73, 6, 1));
    assert_eq!(CANDIDATE2.grid, (73, 3, 1));
    assert_eq!(CANDIDATE1.block, 128);
    assert_eq!(CANDIDATE2.block, 256);
    assert_eq!(CANDIDATE1.shared, 17_408);
    assert_eq!(CANDIDATE2.shared, 25_600);
    assert_eq!(CANDIDATE1.minimum_blocks, 5);
    assert_eq!(CANDIDATE2.minimum_blocks, 2);
    assert_eq!(CANDIDATE1.sm89_register_cap, 102);
    assert_eq!(CANDIDATE1.sm120_register_cap, 102);
    assert_eq!(CANDIDATE2.sm89_register_cap, 128);
    assert_eq!(CANDIDATE2.sm120_register_cap, 128);
    assert_eq!(384 * 1_928, 740_352);
    assert!(740_352 * size_of::<f32>() <= 16 * 1024 * 1024);
}

#[test]
fn physical_identity_gate_rejects_every_structural_mutation() {
    for candidate in [CANDIDATE1, CANDIDATE2] {
        let valid = [
            NodeIdentity {
                symbol: TRANSPOSE_SYMBOL.into(),
                grid: (61, 12, 1),
                block: (32, 16, 1),
                shared: 0,
                arguments_digest: [1; 32],
            },
            NodeIdentity {
                symbol: candidate.symbol.into(),
                grid: candidate.grid,
                block: (candidate.block, 1, 1),
                shared: candidate.shared,
                arguments_digest: [2; 32],
            },
        ];
        assert!(validate_two_node_identity(&valid, candidate).is_ok());
        assert!(validate_two_node_identity(&valid[..1], candidate).is_err());
        let mut reversed = valid.clone();
        reversed.reverse();
        assert!(validate_two_node_identity(&reversed, candidate).is_err());
        let mutations: [fn(&mut [NodeIdentity; 2]); 10] = [
            |nodes| nodes[0].symbol.push_str("_wrong"),
            |nodes| nodes[0].grid.0 += 1,
            |nodes| nodes[0].block.1 -= 1,
            |nodes| nodes[0].shared += 4,
            |nodes| nodes[1].symbol.push_str("_wrong"),
            |nodes| nodes[1].grid.0 += 1,
            |nodes| nodes[1].grid.1 += 1,
            |nodes| nodes[1].block.0 -= 1,
            |nodes| nodes[1].shared -= 4,
            |nodes| nodes[1].arguments_digest = nodes[0].arguments_digest,
        ];
        for mutate in mutations {
            let mut changed = valid.clone();
            mutate(&mut changed);
            assert!(validate_two_node_identity(&changed, candidate).is_err());
        }
        let mut zero = valid;
        zero[0].arguments_digest = [0; 32];
        assert!(validate_two_node_identity(&zero, candidate).is_err());
    }
}

#[test]
fn prism_candidate_cuda_source_is_exact_and_human_scoped() {
    let source = candidate_source().expect("test-only candidate CUDA source");
    validate_candidate_source(&source).expect("test-only candidate source contract");
}

#[test]
fn prism_candidate_source_gate_rejects_mutations() {
    let source = candidate_source().expect("test-only candidate CUDA source");
    for (from, to) in [
        ("PRISM_EXPERIMENT_M 4621", "PRISM_EXPERIMENT_M 4620"),
        (
            "PRISM_EXPERIMENT_PADDED_K 1936",
            "PRISM_EXPERIMENT_PADDED_K 1928",
        ),
        ("__launch_bounds__(128, 5)", "__launch_bounds__(128, 4)"),
        ("__launch_bounds__(256, 2)", "__launch_bounds__(256, 1)"),
        ("__fmaf_rn(", "fmaf("),
        (
            "reinterpret_cast<float4*>(destination)[0] = output;",
            "destination[0] = output.x;",
        ),
        (
            "    int bk_index\n) {",
            "    int bk_index,\n    int extra_argument,\n    int second_extra_argument\n) {",
        ),
    ] {
        assert!(source.contains(from), "mutation source token {from:?}");
        let mutated = source.replacen(from, to, 1);
        assert!(
            validate_candidate_source(&mutated).is_err(),
            "mutation {from:?} -> {to:?} escaped"
        );
    }
}

#[test]
fn timing_helpers_are_fail_closed_at_exact_boundaries() {
    assert_eq!(parse_windows(Err(std::env::VarError::NotPresent)), Ok(101));
    assert_eq!(parse_windows(Ok("21".into())), Ok(21));
    assert_eq!(parse_windows(Ok("101".into())), Ok(101));
    for invalid in ["0", "20", "22", "100", "102", "bad"] {
        assert!(parse_windows(Ok(invalid.into())).is_err(), "{invalid}");
    }
    assert!(percentile(&[], 0.5).is_err());
    assert!(percentile(&[1.0, f64::NAN], 0.5).is_err());
    assert!(percentile(&[1.0, 0.0], 0.5).is_err());
    assert!(percentile(&[1.0], 0.0).is_err());
    assert!(paired_sample(1.0, 1.0, 1.0, 1.0).is_ok());
    assert!(paired_sample(1.0, 1.0, 0.0, 1.0).is_err());
    for order in ["ABBA", "BAAB"] {
        assert!(validate_speedup_gate(order, 21, 1.005, 1.01, 1.02).is_ok());
        assert!(validate_speedup_gate(order, 101, 1.005, 1.01, 1.02).is_ok());
        assert!(validate_speedup_gate(order, 101, 1.004_999, 1.01, 1.02).is_err());
        assert!(validate_speedup_gate(order, 101, 1.005, 1.009_999, 1.02).is_err());
    }
    assert!(validate_speedup_gate("AABB", 101, 1.1, 1.1, 1.1).is_err());
    assert!(validate_speedup_gate("ABBA", 100, 1.1, 1.1, 1.1).is_err());
    assert!(validate_speedup_gate("ABBA", 101, f64::NAN, 1.1, 1.1).is_err());
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use std::ffi::{CStr, c_int, c_void};
    use std::mem::size_of;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION,
    };
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;

    use super::{
        CANDIDATE1, CANDIDATE1_SYMBOL, CANDIDATE2, CANDIDATE2_SYMBOL, CandidateSpec,
        MIN_P05_SPEEDUP, MIN_P50_SPEEDUP, NodeIdentity, OFFICIAL_WINDOWS, PRODUCTION_SYMBOL,
        TRANSPOSE_SYMBOL, paired_sample, parse_windows, percentile, validate_speedup_gate,
        validate_two_node_identity,
    };

    const DIMS: (usize, usize, usize) = (4_621, 384, 1_928);
    const SCRATCH_ELEMENTS: usize = 384 * 1_928;
    const TRANSPOSE_SHARED: usize = 32 * 33 * size_of::<f32>();
    const GENERIC_NT_SYMBOL: &str = "gemm_bi_nt";
    const GENERIC_NT_SHARED: usize = 33_376;
    const GUARD: usize = 64;
    const INPUT_GUARD: u32 = 0x7fc0_5a31;
    const OUTPUT_GUARD: u32 = 0x7fc0_c531;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const PEDANTIC_MIN_P05: f64 = 0.90;
    const PEDANTIC_MIN_P50: f64 = 0.92;
    const PRODUCTION_SPEC: CandidateSpec = CandidateSpec {
        symbol: PRODUCTION_SYMBOL,
        block: 128,
        grid: (438, 1, 1),
        shared: 17_408,
        minimum_blocks: 4,
        sm89_register_cap: 123,
        sm120_register_cap: 103,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        Candidate1,
        Candidate2,
        GenericNt,
        CublasPedantic,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Production => "production_prism_tag30",
                Self::Candidate1 => "prism_m64n64_occ5",
                Self::Candidate2 => "prism_m64n128",
                Self::GenericNt => "generic_ascending_nt",
                Self::CublasPedantic => "cublas_pedantic",
            }
        }

        const fn candidate_spec(self) -> Option<CandidateSpec> {
            match self {
                Self::Production => Some(PRODUCTION_SPEC),
                Self::Candidate1 => Some(CANDIDATE1),
                Self::Candidate2 => Some(CANDIDATE2),
                Self::GenericNt | Self::CublasPedantic => None,
            }
        }
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
        transpose: Kernel,
        production: Kernel,
        candidate1: Kernel,
        candidate2: Kernel,
        generic_nt: Kernel,
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

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
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

        fn expected_bits(&self) -> Vec<u32> {
            self.expected[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.buffer.to_cpu(stream)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} or its red zones changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        scratch: GuardedBuffer,
        output: GuardedBuffer,
        generic_output: GuardedBuffer,
        cublas_output: GuardedBuffer,
        params: NnParams,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct CublasGeometry {
        m: c_int,
        n: c_int,
        k: c_int,
        lda: c_int,
        ldb: c_int,
        ldc: c_int,
        output_elements: usize,
    }

    fn cublas_geometry(logical: (usize, usize, usize)) -> Result<CublasGeometry, String> {
        let (m, k_out, n_reduction) = logical;
        let as_int = |value: usize, name: &str| {
            c_int::try_from(value).map_err(|_| format!("cuBLAS {name} exceeds c_int: {value}"))
        };
        Ok(CublasGeometry {
            m: as_int(k_out, "physical M")?,
            n: as_int(m, "physical N")?,
            k: as_int(n_reduction, "physical K")?,
            lda: as_int(n_reduction, "lda")?,
            ldb: as_int(n_reduction, "ldb")?,
            ldc: as_int(k_out, "ldc")?,
            output_elements: m
                .checked_mul(k_out)
                .ok_or_else(|| "cuBLAS output extent overflow".to_owned())?,
        })
    }

    fn compose_source() -> String {
        [
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/scalar.cu"),
            include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
            include_str!("../kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu"),
            include_str!("gemm_bi_scalar_nt_prism_experiment.cu"),
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
    ) -> Result<Kernel, String> {
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        if config.shared_mem_bytes > 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    config.shared_mem_bytes as i32,
                )
                .map_err(|error| format!("set {symbol} dynamic shared memory: {error:?}"))?;
        }
        Ok(Kernel {
            function,
            config,
            symbol,
        })
    }

    fn validate_environment(device: &GpuDevice, ctx: &GpuCtx) -> Result<(), String> {
        let identity = device.identity();
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
        if identity.compute_capability != (12, 0)
            || identity.multiprocessor_count != 170
            || device.nvrtc_target() != "compute_120"
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
                "prism experiment requires exact CC12.0/170SM compute_120 NVRTC13.2 TriadScalar identity: identity={identity:?} compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        validate_environment(&device, &ctx)?;
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
            .map_err(|error| format!("compile prism experiment: {error:?}"))?;
        if !ptx.to_src().contains("fma.rn.f32") {
            return Err("candidate PTX lost fma.rn.f32".into());
        }
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load prism experiment: {error:?}"))?;
        let transpose = load_kernel(
            &module,
            TRANSPOSE_SYMBOL,
            LaunchConfig {
                grid_dim: (61, 12, 1),
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
            },
        )?;
        let load_spec = |spec: CandidateSpec| {
            load_kernel(
                &module,
                spec.symbol,
                LaunchConfig {
                    grid_dim: spec.grid,
                    block_dim: (spec.block, 1, 1),
                    shared_mem_bytes: spec.shared,
                },
            )
        };
        let production = load_spec(PRODUCTION_SPEC)?;
        let candidate1 = load_spec(CANDIDATE1)?;
        let candidate2 = load_spec(CANDIDATE2)?;
        let generic_nt = load_kernel(
            &module,
            GENERIC_NT_SYMBOL,
            LaunchConfig {
                grid_dim: (111, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: GENERIC_NT_SHARED as u32,
            },
        )?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            transpose,
            production,
            candidate1,
            candidate2,
            generic_nt,
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

    fn fixture_with_values(
        runtime: &Runtime,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
    ) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        if a_values.len() != m * n || b_values.len() != k_out * n {
            return Err("prism experiment input dimensions changed".into());
        }
        let output = vec![0.0; m * k_out];
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a_values, INPUT_GUARD)?,
            b: GuardedBuffer::new(&runtime.ctx.stream, b_values, INPUT_GUARD)?,
            scratch: GuardedBuffer::new(
                &runtime.ctx.stream,
                vec![0.0; SCRATCH_ELEMENTS],
                OUTPUT_GUARD,
            )?,
            output: GuardedBuffer::new(&runtime.ctx.stream, output.clone(), OUTPUT_GUARD)?,
            generic_output: GuardedBuffer::new(&runtime.ctx.stream, output.clone(), OUTPUT_GUARD)?,
            cublas_output: GuardedBuffer::new(&runtime.ctx.stream, output, OUTPUT_GUARD)?,
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

    fn regular_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        fixture_with_values(
            runtime,
            values(DIMS.0 * DIMS.2, 0xa531_1001),
            values(DIMS.1 * DIMS.2, 0xb531_1002),
        )
    }

    fn exceptional_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let mut a = vec![0.0; m * n];
        for row in 0..m {
            a[row * n + n - 1] = 1.0;
        }
        let patterns = [
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
        ];
        let mut b = vec![0.0; k_out * n];
        for row in 0..k_out {
            b[row * n + n - 1] = f32::from_bits(patterns[row % patterns.len()]);
        }
        fixture_with_values(runtime, a, b)
    }

    fn candidate_kernel(runtime: &Runtime, arm: Arm) -> Result<&Kernel, String> {
        match arm {
            Arm::Production => Ok(&runtime.production),
            Arm::Candidate1 => Ok(&runtime.candidate1),
            Arm::Candidate2 => Ok(&runtime.candidate2),
            _ => Err(format!("{} has no NN kernel", arm.name())),
        }
    }

    fn launch_transpose_nn(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        let scratch = fixture.scratch.ptr(stream);
        let b = fixture.b.ptr(stream);
        let rows = DIMS.1 as i32;
        let cols = DIMS.2 as i32;
        let mut transpose = stream.launch_builder(&runtime.transpose.function);
        transpose.arg(&scratch);
        transpose.arg(&b);
        transpose.arg(&rows);
        transpose.arg(&cols);
        unsafe { transpose.launch(runtime.transpose.config) }
            .map_err(|error| format!("launch {TRANSPOSE_SYMBOL}: {error:?}"))?;

        let kernel = candidate_kernel(runtime, arm)?;
        let output = fixture.output.ptr(stream);
        let a = fixture.a.ptr(stream);
        let bias = 0_u64;
        let mut builder = stream.launch_builder(&kernel.function);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&scratch);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", kernel.symbol))
    }

    fn launch_generic(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        let output = fixture.generic_output.ptr(stream);
        let a = fixture.a.ptr(stream);
        let b = fixture.b.ptr(stream);
        let alpha = 1.0_f32;
        let m = DIMS.0 as i32;
        let n = DIMS.2 as i32;
        let k = DIMS.1 as i32;
        let mut builder = stream.launch_builder(&runtime.generic_nt.function);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&alpha);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&k);
        unsafe { builder.launch(runtime.generic_nt.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {GENERIC_NT_SYMBOL}: {error:?}"))
    }

    fn launch_cublas_nt(
        runtime: &Runtime,
        output: u64,
        a: u64,
        b: u64,
        logical: (usize, usize, usize),
    ) -> Result<(), String> {
        use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC;
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

        let geometry = cublas_geometry(logical)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *runtime.ctx.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                geometry.m,
                geometry.n,
                geometry.k,
                (&alpha as *const f32).cast::<c_void>(),
                b as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.lda,
                a as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldb,
                (&beta as *const f32).cast::<c_void>(),
                output as *mut c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldc,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|error| format!("cuBLAS pedantic launch: {error:?}"))?;
        }
        Ok(())
    }

    fn launch_cublas_pedantic(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        launch_cublas_nt(
            runtime,
            fixture.cublas_output.ptr(stream),
            fixture.a.ptr(stream),
            fixture.b.ptr(stream),
            DIMS,
        )
    }

    fn validate_cublas_orientation(runtime: &Runtime) -> Result<(), String> {
        const LOGICAL: (usize, usize, usize) = (3, 2, 4);
        let geometry = cublas_geometry(LOGICAL)?;
        let expected = [14.0_f32, 34.0, 33.0, 57.0, 65.0, 115.0];
        if geometry.output_elements != expected.len() {
            return Err(format!("cuBLAS orientation extent changed: {geometry:?}"));
        }
        let a = GuardedBuffer::new(
            &runtime.ctx.stream,
            vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            INPUT_GUARD,
        )?;
        let b = GuardedBuffer::new(
            &runtime.ctx.stream,
            vec![7.0, 11.0, 13.0, 0.0, 17.0, 19.0, 23.0, 0.0],
            INPUT_GUARD,
        )?;
        let output =
            GuardedBuffer::new(&runtime.ctx.stream, vec![0.0; expected.len()], OUTPUT_GUARD)?;
        for repeat in 0..3 {
            launch_cublas_nt(
                runtime,
                output.ptr(&runtime.ctx.stream),
                a.ptr(&runtime.ctx.stream),
                b.ptr(&runtime.ctx.stream),
                LOGICAL,
            )?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("cuBLAS orientation sync: {error:?}"))?;
            if output.active_bits(&runtime.ctx.stream, "cuBLAS orientation")?
                != expected.map(f32::to_bits)
            {
                return Err(format!("cuBLAS orientation repeat {repeat} changed"));
            }
            a.validate_unchanged(&runtime.ctx.stream, "cuBLAS orientation A")?;
            b.validate_unchanged(&runtime.ctx.stream, "cuBLAS orientation B")?;
        }
        Ok(())
    }

    fn launch_arm(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        match arm {
            Arm::Production | Arm::Candidate1 | Arm::Candidate2 => {
                launch_transpose_nn(runtime, fixture, arm)
            }
            Arm::GenericNt => launch_generic(runtime, fixture),
            Arm::CublasPedantic => launch_cublas_pedantic(runtime, fixture),
        }
    }

    fn capture_arm(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch_arm(runtime, fixture, arm)) }
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"nt-prism-post-tag30-arguments.v1");
        for argument in arguments {
            digest.update((argument.len() as u64).to_le_bytes());
            digest.update(argument);
        }
        digest.finalize().into()
    }

    fn eager_identity(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Vec<NodeIdentity> {
        let stream = &runtime.ctx.stream;
        let scratch = fixture.scratch.ptr(stream);
        let b = fixture.b.ptr(stream);
        let rows = DIMS.1 as i32;
        let cols = DIMS.2 as i32;
        let output = fixture.output.ptr(stream);
        let a = fixture.a.ptr(stream);
        let bias = 0_u64;
        let kernel = candidate_kernel(runtime, arm).expect("candidate arm");
        vec![
            NodeIdentity {
                symbol: TRANSPOSE_SYMBOL.into(),
                grid: runtime.transpose.config.grid_dim,
                block: runtime.transpose.config.block_dim,
                shared: runtime.transpose.config.shared_mem_bytes,
                arguments_digest: digest_arguments(&[
                    bytes_of(&scratch),
                    bytes_of(&b),
                    bytes_of(&rows),
                    bytes_of(&cols),
                ]),
            },
            NodeIdentity {
                symbol: kernel.symbol.into(),
                grid: kernel.config.grid_dim,
                block: kernel.config.block_dim,
                shared: kernel.config.shared_mem_bytes,
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

    unsafe fn graph_node_identity(node: sys::CUgraphNode) -> Result<NodeIdentity, String> {
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
            "graph kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "graph kernel name",
        )?;
        if name.is_null() {
            return Err("graph kernel name is null".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph kernel name is not UTF-8: {error}"))?
            .to_owned();
        let sizes: &[usize] = match symbol.as_str() {
            TRANSPOSE_SYMBOL => &[8, 8, 4, 4],
            PRODUCTION_SYMBOL | CANDIDATE1_SYMBOL | CANDIDATE2_SYMBOL => {
                &[8, 8, 8, 8, size_of::<NnParams>()]
            }
            _ => return Err(format!("unexpected graph symbol {symbol}")),
        };
        if params.kernelParams.is_null() {
            return Err(format!("{symbol} graph parameters are null"));
        }
        let mut arguments = Vec::with_capacity(sizes.len());
        for (index, size) in sizes.iter().copied().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{symbol} argument {index} is null"));
            }
            arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        Ok(NodeIdentity {
            symbol,
            grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block: (params.blockDimX, params.blockDimY, params.blockDimZ),
            shared: params.sharedMemBytes,
            arguments_digest: digest_arguments(&arguments),
        })
    }

    fn graph_identity(
        graph: &CudaGraph,
        candidate: CandidateSpec,
    ) -> Result<Vec<NodeIdentity>, String> {
        let raw = graph.cu_graph();
        let mut nodes = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut nodes) },
            "graph node count",
        )?;
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
            "graph edge count",
        )?;
        if nodes != 2 || edges != 1 {
            return Err(format!(
                "candidate graph must have two nodes and one edge: {nodes}/{edges}"
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
                    &mut edges,
                )
            },
            "graph edge",
        )?;
        if edge_data[0].from_port != 0
            || edge_data[0].to_port != 0
            || edge_data[0].type_
                != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
            || edge_data[0].reserved != [0; 5]
        {
            return Err("candidate graph edge descriptor changed".into());
        }
        let identity = vec![unsafe { graph_node_identity(from[0]) }?, unsafe {
            graph_node_identity(to[0])
        }?];
        validate_two_node_identity(&identity, candidate)?;
        Ok(identity)
    }

    fn validate_scratch(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let input = fixture.b.expected_bits();
        let scratch = fixture
            .scratch
            .active_bits(&runtime.ctx.stream, "transpose scratch")?;
        for row in 0..DIMS.1 {
            for column in 0..DIMS.2 {
                let expected = input[row * DIMS.2 + column];
                let actual = scratch[column * DIMS.1 + row];
                if actual != expected {
                    return Err(format!(
                        "transpose mismatch ({row},{column}): actual=0x{actual:08x} expected=0x{expected:08x}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn arm_bits(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<Vec<u32>, String> {
        match arm {
            Arm::GenericNt => fixture
                .generic_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            Arm::CublasPedantic => fixture
                .cublas_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            _ => fixture.output.active_bits(&runtime.ctx.stream, arm.name()),
        }
    }

    fn validate_exceptional_oracle(bits: &[u32]) -> Result<(), String> {
        let expected_len = DIMS.0 * DIMS.1;
        if bits.len() != expected_len {
            return Err(format!(
                "exceptional output has {} elements, expected {expected_len}",
                bits.len()
            ));
        }
        for row in [0, DIMS.0 - 1] {
            let values = &bits[row * DIMS.1..row * DIMS.1 + 7];
            if values[0] & 0x7fff_ffff != 0
                || values[1] != 0x0000_0001
                || values[2] != 0x007f_ffff
                || values[3] != 0x7f80_0000
                || values[4] != 0xff80_0000
                || values[5] & 0x7f80_0000 != 0x7f80_0000
                || values[5] & 0x007f_ffff == 0
                || values[6] & 0x7f80_0000 != 0x7f80_0000
                || values[6] & 0x007f_ffff == 0
            {
                return Err(format!(
                    "exceptional tail oracle changed at row {row}: {values:08x?}"
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
        minimum_blocks: u32,
    }

    fn check_resources(kernel: &Kernel, contract: ResourceContract) -> Result<(), String> {
        let threads = kernel.config.block_dim.0 * kernel.config.block_dim.1;
        let registers = kernel
            .function
            .num_regs()
            .map_err(|error| format!("{} registers: {error:?}", kernel.symbol))?;
        let local = kernel
            .function
            .local_size_bytes()
            .map_err(|error| format!("{} local memory: {error:?}", kernel.symbol))?;
        let static_shared = kernel
            .function
            .shared_size_bytes()
            .map_err(|error| format!("{} static shared: {error:?}", kernel.symbol))?;
        let blocks = kernel
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                kernel.config.shared_mem_bytes as usize,
                None,
            )
            .map_err(|error| format!("{} occupancy: {error:?}", kernel.symbol))?;
        eprintln!(
            "post_tag30_resource symbol={} threads={} regs={} local={} static_shared={} dynamic_shared={} blocks={}",
            kernel.symbol,
            threads,
            registers,
            local,
            static_shared,
            kernel.config.shared_mem_bytes,
            blocks
        );
        if threads != contract.threads
            || static_shared as usize != contract.static_shared
            || kernel.config.shared_mem_bytes as usize != contract.dynamic_shared
            || local != 0
            || registers > contract.register_cap
            || blocks < contract.minimum_blocks
        {
            return Err(format!(
                "{} resource contract failed: threads={threads} registers={registers} local={local} static={static_shared} dynamic={} blocks={blocks}; expected static={}",
                kernel.symbol, kernel.config.shared_mem_bytes, contract.static_shared
            ));
        }
        Ok(())
    }

    fn validate_production_identity(runtime: &Runtime) -> Result<(), String> {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let qualified = qualify_physical_launch(&runtime.ctx, request)?;
        let evidence = qualified.evidence();
        let nodes = evidence.nodes();
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 2
            || evidence.launch_digest() == [0; 32]
            || nodes.len() != 2
            || nodes.iter().any(|node| {
                node.module_kind != ModuleKind::TriadScalar
                    || node.logical_op != ResolvedGemmOp::Nt
                    || node.shape != DIMS
                    || node.strides != (1_928, 1_928, 384)
                    || node.launch.arguments_digest == [0; 32]
            })
            || nodes[0].symbol != TRANSPOSE_SYMBOL
            || nodes[0].tile != Some((32, 32))
            || (
                nodes[0].launch.grid_dim,
                nodes[0].launch.block_dim,
                nodes[0].launch.shared_mem_bytes,
            ) != ((61, 12, 1), (32, 16, 1), 0)
            || nodes[1].symbol != PRODUCTION_SYMBOL
            || nodes[1].tile != Some((64, 64))
            || (
                nodes[1].launch.grid_dim,
                nodes[1].launch.block_dim,
                nodes[1].launch.shared_mem_bytes,
            ) != ((438, 1, 1), (128, 1, 1), 17_408)
            || nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest
        {
            return Err(format!("production tag30 identity changed: {nodes:?}"));
        }
        qualified.validate_red_zones(&runtime.ctx)?;
        Ok(())
    }

    fn validate_fixture(
        runtime: &Runtime,
        fixture: &Fixture,
        exceptional: bool,
    ) -> Result<(), String> {
        launch_arm(runtime, fixture, Arm::GenericNt)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("generic reference sync: {error:?}"))?;
        let exact = arm_bits(runtime, fixture, Arm::GenericNt)?;
        if exceptional {
            validate_exceptional_oracle(&exact)?;
        }

        for arm in [Arm::Production, Arm::Candidate1, Arm::Candidate2] {
            let spec = arm.candidate_spec().expect("candidate spec");
            for repeat in 0..3 {
                launch_arm(runtime, fixture, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{} eager sync: {error:?}", arm.name()))?;
                if arm_bits(runtime, fixture, arm)? != exact {
                    return Err(format!(
                        "{} eager repeat {repeat} changed exact bits",
                        arm.name()
                    ));
                }
                validate_scratch(runtime, fixture)?;
            }
            let eager = eager_identity(runtime, fixture, arm);
            validate_two_node_identity(&eager, spec)?;
            let graph = capture_arm(runtime, fixture, arm)?;
            let captured = graph_identity(&graph, spec)?;
            if eager != captured {
                return Err(format!(
                    "{} eager/graph identity differs: eager={eager:?} graph={captured:?}",
                    arm.name()
                ));
            }
            for repeat in 0..3 {
                graph
                    .launch()
                    .map_err(|error| format!("{} graph launch: {error:?}", arm.name()))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{} graph sync: {error:?}", arm.name()))?;
                if arm_bits(runtime, fixture, arm)? != exact {
                    return Err(format!(
                        "{} graph repeat {repeat} changed exact bits",
                        arm.name()
                    ));
                }
                validate_scratch(runtime, fixture)?;
            }
        }
        fixture.a.validate_unchanged(&runtime.ctx.stream, "A")?;
        fixture.b.validate_unchanged(&runtime.ctx.stream, "B")?;
        Ok(())
    }

    fn parse_registers(block: &str) -> i32 {
        let (_, tail) = block
            .split_once("Used ")
            .unwrap_or_else(|| panic!("ptxas log lost resource line:\n{block}"));
        tail.split_whitespace()
            .next()
            .expect("ptxas register count")
            .parse()
            .expect("numeric ptxas register count")
    }

    #[test]
    #[ignore = "requires CUDA NVRTC and ptxas but launches no GPU work"]
    fn prism_candidates_compile_without_spills() {
        for (arch, sm) in [("compute_89", "sm_89"), ("compute_120", "sm_120")] {
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some(arch),
                options: vec![
                    "--fmad=true".into(),
                    "--extra-device-vectorization".into(),
                    "-DNDEBUG".into(),
                    "-DGEMM_BI_GROUP_M=16".into(),
                ],
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
                .unwrap_or_else(|error| panic!("{arch} compile failed: {error:?}"))
                .to_src();
            assert!(ptx.contains("fma.rn.f32"));
            let stem = format!("mamba-prism-post-tag30-{}-{sm}", std::process::id());
            let ptx_path = std::env::temp_dir().join(format!("{stem}.ptx"));
            let cubin_path = std::env::temp_dir().join(format!("{stem}.cubin"));
            std::fs::write(&ptx_path, &ptx).expect("write candidate PTX");
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
                .unwrap_or_else(|error| panic!("launch ptxas {sm}: {error}"));
            let log = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.status.success(), "ptxas {sm} failed:\n{log}");
            for spec in [CANDIDATE1, CANDIDATE2] {
                let marker = format!("Compiling entry function '{}'", spec.symbol);
                let block = log
                    .split_once(&marker)
                    .unwrap_or_else(|| panic!("{sm} resource log lost {}", spec.symbol))
                    .1
                    .split("Compiling entry function '")
                    .next()
                    .expect("ptxas resource block");
                assert!(
                    block
                        .contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"),
                    "{sm} {} local/spill regression:\n{block}",
                    spec.symbol
                );
                let registers = parse_registers(block);
                let cap = if sm == "sm_89" {
                    spec.sm89_register_cap
                } else {
                    spec.sm120_register_cap
                };
                assert!(
                    registers <= cap as i32,
                    "{sm} {} uses {registers} registers, cap {cap}",
                    spec.symbol
                );
                eprintln!(
                    "post_tag30_ptxas sm={sm} symbol={} registers={registers} stack=0 spills=0",
                    spec.symbol
                );
            }
            let marker = format!("Compiling entry function '{TRANSPOSE_SYMBOL}'");
            let block = log
                .split_once(&marker)
                .unwrap_or_else(|| panic!("{sm} resource log lost {TRANSPOSE_SYMBOL}"))
                .1
                .split("Compiling entry function '")
                .next()
                .expect("transpose ptxas resource block");
            assert!(
                block.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"),
                "{sm} transpose local/spill regression:\n{block}"
            );
            let registers = parse_registers(block);
            let cap = if sm == "sm_89" { 18 } else { 24 };
            assert!(
                registers <= cap,
                "{sm} transpose uses {registers} registers, cap {cap}"
            );
            assert!(
                block.contains(&format!("{TRANSPOSE_SHARED} bytes smem")),
                "{sm} transpose static shared changed:\n{block}"
            );
            let _ = std::fs::remove_file(ptx_path);
            let _ = std::fs::remove_file(cubin_path);
        }
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM GPU"]
    fn prism_candidates_are_exact_and_graph_stable() -> Result<(), String> {
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("prism-post-tag30-correctness-pre-context")?;
        let runtime = new_runtime()?;
        quiet.require_cohort("prism-post-tag30-correctness")?;
        validate_production_identity(&runtime)?;
        validate_cublas_orientation(&runtime)?;
        check_resources(
            &runtime.transpose,
            ResourceContract {
                threads: 512,
                static_shared: TRANSPOSE_SHARED,
                dynamic_shared: 0,
                register_cap: 28,
                minimum_blocks: 3,
            },
        )?;
        check_resources(
            &runtime.candidate1,
            ResourceContract {
                threads: 128,
                static_shared: 0,
                dynamic_shared: CANDIDATE1.shared as usize,
                register_cap: CANDIDATE1.sm120_register_cap as i32,
                minimum_blocks: CANDIDATE1.minimum_blocks,
            },
        )?;
        check_resources(
            &runtime.candidate2,
            ResourceContract {
                threads: 256,
                static_shared: 0,
                dynamic_shared: CANDIDATE2.shared as usize,
                register_cap: CANDIDATE2.sm120_register_cap as i32,
                minimum_blocks: CANDIDATE2.minimum_blocks,
            },
        )?;
        let regular = regular_fixture(&runtime)?;
        validate_fixture(&runtime, &regular, false)?;
        let exceptional = exceptional_fixture(&runtime)?;
        validate_fixture(&runtime, &exceptional, true)?;
        quiet.verify_post_cohort("prism-post-tag30-correctness")?;
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iteration count is zero".into());
        }
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_arm(runtime, fixture, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record end event: {error:?}"))?;
        let elapsed = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("elapsed event time: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err(format!("{} measured invalid time {elapsed}", arm.name()));
        }
        Ok(elapsed)
    }

    fn calibrated_iterations(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<usize, String> {
        let pilot = measure(runtime, fixture, arm, 3)?;
        Ok((TARGET_WINDOW_US / pilot).round().clamp(3.0, 500.0) as usize)
    }

    fn validate_pedantic_gate(
        order: &str,
        windows: usize,
        p05: f64,
        p50: f64,
        p95: f64,
    ) -> Result<(), String> {
        if !matches!(order, "ABBA" | "BAAB")
            || !matches!(windows, 21 | 101)
            || [p05, p50, p95]
                .into_iter()
                .any(|value| !value.is_finite() || value <= 0.0)
            || p05 < PEDANTIC_MIN_P05
            || p50 < PEDANTIC_MIN_P50
        {
            return Err(format!(
                "pedantic guard failed: order={order} windows={windows} p05={p05} p50={p50} p95={p95}"
            ));
        }
        Ok(())
    }

    fn paired(
        runtime: &Runtime,
        fixture: &Fixture,
        baseline: Arm,
        candidate: Arm,
        windows: usize,
        hard_candidate_gate: bool,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        let label = format!("post-tag30-{}-to-{}", baseline.name(), candidate.name());
        quiet.require_cohort(&label)?;
        let iterations = calibrated_iterations(runtime, fixture, baseline)?
            .max(calibrated_iterations(runtime, fixture, candidate)?);
        let mut failures = Vec::new();
        for (order, baseline_first) in [("ABBA", true), ("BAAB", false)] {
            let mut baseline_samples = Vec::with_capacity(windows);
            let mut candidate_samples = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (b0, b1, c0, c1) = if baseline_first {
                    let b0 = measure(runtime, fixture, baseline, iterations)?;
                    let c0 = measure(runtime, fixture, candidate, iterations)?;
                    let c1 = measure(runtime, fixture, candidate, iterations)?;
                    let b1 = measure(runtime, fixture, baseline, iterations)?;
                    (b0, b1, c0, c1)
                } else {
                    let c0 = measure(runtime, fixture, candidate, iterations)?;
                    let b0 = measure(runtime, fixture, baseline, iterations)?;
                    let b1 = measure(runtime, fixture, baseline, iterations)?;
                    let c1 = measure(runtime, fixture, candidate, iterations)?;
                    (b0, b1, c0, c1)
                };
                let (baseline_us, candidate_us, ratio) = paired_sample(b0, b1, c0, c1)?;
                baseline_samples.push(baseline_us);
                candidate_samples.push(candidate_us);
                ratios.push(ratio);
            }
            let baseline_p50 = percentile(&baseline_samples, 0.50)?;
            let candidate_p50 = percentile(&candidate_samples, 0.50)?;
            let p05 = percentile(&ratios, 0.05)?;
            let p50 = percentile(&ratios, 0.50)?;
            let p95 = percentile(&ratios, 0.95)?;
            eprintln!(
                "post_tag30 baseline={} candidate={} order={order} windows={windows} iterations={iterations} baseline_us_p50={baseline_p50:.6} candidate_us_p50={candidate_p50:.6} speedup_p05={p05:.9} speedup_p50={p50:.9} speedup_p95={p95:.9}",
                baseline.name(),
                candidate.name()
            );
            let gate = if hard_candidate_gate {
                validate_speedup_gate(order, windows, p05, p50, p95)
            } else {
                validate_pedantic_gate(order, windows, p05, p50, p95)
            };
            if let Err(error) = gate {
                failures.push(format!("{order}: {error}"));
            }
        }
        quiet.verify_post_cohort(&label)?;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    }

    fn warmup(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        for arm in [
            Arm::Production,
            Arm::Candidate1,
            Arm::Candidate2,
            Arm::GenericNt,
            Arm::CublasPedantic,
        ] {
            for _ in 0..5 {
                launch_arm(runtime, fixture, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("timing warmup: {error:?}"))
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM GPU and exactly 21 windows"]
    fn prism_screening_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("timing requires a release build".into());
        }
        let windows = parse_windows(std::env::var("MAMBA_RS_NT_PRISM_POST_TAG30_WINDOWS"))?;
        if windows != 21 {
            return Err(format!(
                "screening requires exactly 21 windows, found {windows}"
            ));
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("prism-post-tag30-screen-pre-context")?;
        let runtime = new_runtime()?;
        let fixture = regular_fixture(&runtime)?;
        warmup(&runtime, &fixture)?;
        let mut failures = Vec::new();
        for candidate in [Arm::Candidate1, Arm::Candidate2] {
            if let Err(error) = paired(
                &runtime,
                &fixture,
                Arm::Production,
                candidate,
                windows,
                true,
                &quiet,
            ) {
                failures.push(format!("production -> {}: {error}", candidate.name()));
            }
            if let Err(error) = paired(
                &runtime,
                &fixture,
                Arm::CublasPedantic,
                candidate,
                windows,
                false,
                &quiet,
            ) {
                failures.push(format!("pedantic -> {}: {error}", candidate.name()));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    }

    fn official_winner(value: Result<String, std::env::VarError>) -> Result<Arm, String> {
        match value {
            Ok(value) if value == "candidate1" => Ok(Arm::Candidate1),
            Ok(value) if value == "candidate2" => Ok(Arm::Candidate2),
            Ok(value) => Err(format!("unsupported official winner {value:?}")),
            Err(error) => Err(format!("official winner is required: {error}")),
        }
    }

    #[test]
    #[ignore = "requires parent-selected winner and an exclusive quiet CC12.0/170-SM GPU"]
    fn prism_official_101_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("timing requires a release build".into());
        }
        let windows = parse_windows(std::env::var("MAMBA_RS_NT_PRISM_POST_TAG30_WINDOWS"))?;
        if windows != OFFICIAL_WINDOWS {
            return Err(format!(
                "official timing requires exactly 101 windows, found {windows}"
            ));
        }
        let winner = official_winner(std::env::var("MAMBA_RS_NT_PRISM_POST_TAG30_WINNER"))?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("prism-post-tag30-official-pre-context")?;
        let runtime = new_runtime()?;
        let fixture = regular_fixture(&runtime)?;
        warmup(&runtime, &fixture)?;
        paired(
            &runtime,
            &fixture,
            Arm::Production,
            winner,
            windows,
            true,
            &quiet,
        )?;
        paired(
            &runtime,
            &fixture,
            Arm::CublasPedantic,
            winner,
            windows,
            false,
            &quiet,
        )
    }

    #[test]
    fn timing_boundary_helpers_cover_pedantic_and_winner_selection() {
        assert_eq!(
            cublas_geometry((3, 2, 4)),
            Ok(CublasGeometry {
                m: 2,
                n: 3,
                k: 4,
                lda: 4,
                ldb: 4,
                ldc: 2,
                output_elements: 6,
            })
        );
        assert!(validate_pedantic_gate("ABBA", 21, 0.90, 0.92, 0.95).is_ok());
        assert!(validate_pedantic_gate("BAAB", 101, 0.90, 0.92, 0.95).is_ok());
        assert!(validate_pedantic_gate("ABBA", 21, 0.899_999, 0.92, 0.95).is_err());
        assert!(validate_pedantic_gate("ABBA", 21, 0.90, 0.919_999, 0.95).is_err());
        assert!(validate_pedantic_gate("ABBA", 100, 1.0, 1.0, 1.0).is_err());
        assert_eq!(
            official_winner(Ok("candidate1".into())),
            Ok(Arm::Candidate1)
        );
        assert_eq!(
            official_winner(Ok("candidate2".into())),
            Ok(Arm::Candidate2)
        );
        assert!(official_winner(Ok("production".into())).is_err());
        assert!(official_winner(Err(std::env::VarError::NotPresent)).is_err());
        assert_eq!(MIN_P05_SPEEDUP, 1.005);
        assert_eq!(MIN_P50_SPEEDUP, 1.01);
    }

    #[test]
    fn exceptional_tail_oracle_rejects_each_value_class_mutation() {
        let pattern = [
            0x0000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
        ];
        let mut bits = vec![0; DIMS.0 * DIMS.1];
        bits[..pattern.len()].copy_from_slice(&pattern);
        let tail = (DIMS.0 - 1) * DIMS.1;
        bits[tail..tail + pattern.len()].copy_from_slice(&pattern);
        validate_exceptional_oracle(&bits).expect("valid isolated exceptional tail oracle");
        for column in 0..pattern.len() {
            let mut mutated = bits.clone();
            mutated[column] = 0x3f80_0000;
            assert!(
                validate_exceptional_oracle(&mutated).is_err(),
                "column {column} mutation escaped"
            );
        }
        assert!(validate_exceptional_oracle(&bits[..bits.len() - 1]).is_err());
    }
}
