use std::cmp::Ordering;

#[cfg(feature = "cuda")]
mod common;

const CANDIDATE_SYMBOL: &str = "gemm_bi_nn_splitk32_m32n64_exactshape_exp_v1";
const PRODUCTION_PARTIAL_SYMBOL: &str = "gemm_bi_nn_splitk32_partial";
const REDUCER_SYMBOL: &str = "gemm_bi_splitk_reduce";
const CANDIDATE_SOURCE: &str =
    include_str!("gemm_bi_scalar_nn_splitk_m32n64_exactshape_tournament.cu");
const HARNESS_SOURCE: &str =
    include_str!("gemm_bi_scalar_nn_splitk_m32n64_exactshape_tournament.rs");
const DIMS: (usize, usize, usize) = (128, 8_192, 128);
const K_CHUNKS: usize = DIMS.1 / 32;
const PARTIAL_ELEMENTS: usize = K_CHUNKS * DIMS.0 * DIMS.2;
const CANDIDATE_GRID: (u32, u32, u32) = (2_048, 1, 1);
const PRODUCTION_GRID: (u32, u32, u32) = (2_048, 1, 1);
const REDUCER_GRID: (u32, u32, u32) = (64, 1, 1);
const CANDIDATE_SHARED: u32 = 0;
const SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const MIN_P05_SPEEDUP: f64 = 1.002;
const MIN_P50_SPEEDUP: f64 = 1.002;

const _: () = assert!(K_CHUNKS == 256);
const _: () = assert!(PARTIAL_ELEMENTS == 4_194_304);
const _: () = assert!(PARTIAL_ELEMENTS * size_of::<f32>() == 16 * 1024 * 1024);

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeIdentity {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    arguments_digest: [u8; 32],
}

type ExpectedNodeConfig = (&'static str, (u32, u32, u32), (u32, u32, u32), u32);

fn expected_candidate_configs() -> [ExpectedNodeConfig; 2] {
    [
        (
            CANDIDATE_SYMBOL,
            CANDIDATE_GRID,
            (128, 1, 1),
            CANDIDATE_SHARED,
        ),
        (REDUCER_SYMBOL, REDUCER_GRID, (256, 1, 1), 0),
    ]
}

fn validate_candidate_identity(nodes: &[NodeIdentity]) -> Result<(), String> {
    let expected = expected_candidate_configs();
    if nodes.len() != expected.len() {
        return Err(format!(
            "candidate must contain two ordered nodes, found {}",
            nodes.len()
        ));
    }
    for (node, (symbol, grid, block, shared)) in nodes.iter().zip(expected) {
        if node.symbol != symbol
            || node.grid != grid
            || node.block != block
            || node.shared != shared
        {
            return Err(format!("candidate node order/config changed: {node:?}"));
        }
        if node.arguments_digest == [0; 32] {
            return Err(format!("{} has a zero argument digest", node.symbol));
        }
    }
    if nodes[0].arguments_digest == nodes[1].arguments_digest {
        return Err("candidate node argument digests collided".into());
    }
    Ok(())
}

fn positive_finite(value: f64, label: &str) -> Result<f64, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!(
            "{label} must be finite and positive, found {value}"
        ))
    }
}

fn percentile(samples: &[f64], quantile: f64) -> Result<f64, String> {
    if samples.is_empty() {
        return Err("cannot compute a percentile of empty samples".into());
    }
    if !(0.0..=1.0).contains(&quantile) || !quantile.is_finite() {
        return Err(format!("invalid percentile quantile {quantile}"));
    }
    let mut sorted = Vec::with_capacity(samples.len());
    for &sample in samples {
        sorted.push(positive_finite(sample, "timing sample")?);
    }
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    Ok(sorted[index])
}

fn qualification_windows(value: Option<&str>) -> Result<usize, String> {
    let windows = match value {
        None => SCREEN_WINDOWS,
        Some(value) => value
            .parse::<usize>()
            .map_err(|error| format!("invalid qualification windows {value:?}: {error}"))?,
    };
    if windows != SCREEN_WINDOWS && windows != OFFICIAL_WINDOWS {
        return Err(format!(
            "qualification requires exactly {SCREEN_WINDOWS} or {OFFICIAL_WINDOWS} windows, found {windows}"
        ));
    }
    Ok(windows)
}

fn required_qualification_windows(
    value: Option<&str>,
    expected: usize,
    label: &str,
) -> Result<usize, String> {
    let windows = match value {
        None => expected,
        Some(value) => qualification_windows(Some(value))?,
    };
    if windows != expected {
        return Err(format!(
            "{label} requires exactly {expected} windows, found {windows}"
        ));
    }
    Ok(windows)
}

fn validate_performance_gate(order: &str, p05: f64, p50: f64, p95: f64) -> Result<(), String> {
    positive_finite(p05, "speedup p05")?;
    positive_finite(p50, "speedup p50")?;
    positive_finite(p95, "speedup p95")?;
    if p05 < MIN_P05_SPEEDUP || p50 < MIN_P50_SPEEDUP {
        return Err(format!(
            "M32N64 exact-shape Split-K candidate loses the {order} gate: p05={p05:.9} p50={p50:.9}; required p05>={MIN_P05_SPEEDUP:.2} and p50>={MIN_P50_SPEEDUP:.2}"
        ));
    }
    Ok(())
}

#[test]
fn candidate_source_contract() {
    assert_eq!(
        CANDIDATE_SOURCE
            .matches(&format!("void {CANDIDATE_SYMBOL}("))
            .count(),
        1
    );
    let (_, signature) = CANDIDATE_SOURCE
        .split_once(&format!("void {CANDIDATE_SYMBOL}("))
        .expect("candidate signature");
    let (parameters, _) = signature
        .split_once(") {")
        .expect("candidate parameter list");
    assert_eq!(parameters.matches(',').count() + 1, 7);
    for required in [
        "#define NN_SPLITK_EXACT_M 128",
        "#define NN_SPLITK_EXACT_K 8192",
        "#define NN_SPLITK_EXACT_N 128",
        "#define NN_SPLITK_EXACT_BM 32",
        "#define NN_SPLITK_EXACT_BN 64",
        "#define NN_SPLITK_EXACT_BK 32",
        "#define NN_SPLITK_EXACT_WARP_SIZE 32",
        "#define NN_SPLITK_EXACT_A_PAD 4",
        "#define NN_SPLITK_EXACT_B_PAD 4",
        "TOTAL_BLOCKS == 2048",
        "sizeof(As) + sizeof(Bs) == 13312",
        "M != EXACT_M",
        "N != EXACT_N",
        "K_CHUNKS != EXACT_CHUNKS",
        "lda != EXACT_K",
        "gridDim.x != TOTAL_BLOCKS",
        "blockDim.x != NN_SPLITK_EXACT_THREADS",
        "for (int dot = 0; dot < NN_SPLITK_EXACT_BK; ++dot)",
        "results[index] = __fmaf_rn(",
        "+ (long long)pid_k * EXACT_M * EXACT_N",
        "(long long)(inner_row_a + offset) * EXACT_K",
        "(long long)(inner_row_b + offset) * EXACT_N",
    ] {
        assert!(
            CANDIDATE_SOURCE.contains(required),
            "candidate lost {required}"
        );
    }
    for forbidden in [
        "atomic",
        "mma.sync",
        "wmma",
        "global_row <",
        "global_column + 3 <",
        "(N & 3)",
        "GEMM_BI_SCALAR_SMEM_A_PAD",
        "GEMM_BI_SCALAR_SMEM_B_PAD",
        "GEMM_BI_SCALAR_WARP_SIZE",
    ] {
        assert!(
            !CANDIDATE_SOURCE.contains(forbidden),
            "candidate contains {forbidden}"
        );
    }
    let registry = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert!(!registry.contains(CANDIDATE_SYMBOL));
}

#[test]
fn harness_contract_is_exact_shape_two_nodes_and_full_partial_compare() {
    for required in [
        "const DIMS: (usize, usize, usize) = (128, 8_192, 128)",
        "compare_entire_partial_tensor",
        "exceptional_fixture",
        "capture_into_graph",
        "validate_unchanged",
        "(\"compute_89\", \"sm_89\")",
        "const SCREEN_WINDOWS: usize = 21",
        "const OFFICIAL_WINDOWS: usize = 101",
        "QuietGpu::for_cuda_ordinal(0)",
        "verify_post_cohort_even_on_error",
        "for (order, baseline_first) in [(\"ABBA\", true), (\"BAAB\", false)]",
        "candidate_vs_production_official_101_window_abba_baab",
        "nn-splitk-m32n64-exactshape-official-101",
    ] {
        assert!(HARNESS_SOURCE.contains(required), "harness lost {required}");
    }
    assert_eq!(expected_candidate_configs()[0].1, (2_048, 1, 1));
    assert_eq!(expected_candidate_configs()[1].1, (64, 1, 1));
    assert_eq!(PRODUCTION_GRID, (2_048, 1, 1));
    assert!(HARNESS_SOURCE.contains(PRODUCTION_PARTIAL_SYMBOL));
}

#[test]
fn candidate_identity_rejects_count_order_config_zero_and_collision_mutations() {
    let valid = vec![
        NodeIdentity {
            symbol: CANDIDATE_SYMBOL.into(),
            grid: CANDIDATE_GRID,
            block: (128, 1, 1),
            shared: CANDIDATE_SHARED,
            arguments_digest: [1; 32],
        },
        NodeIdentity {
            symbol: REDUCER_SYMBOL.into(),
            grid: REDUCER_GRID,
            block: (256, 1, 1),
            shared: 0,
            arguments_digest: [2; 32],
        },
    ];
    assert!(validate_candidate_identity(&valid).is_ok());
    assert!(validate_candidate_identity(&valid[..1]).is_err());
    let mut order = valid.clone();
    order.swap(0, 1);
    assert!(validate_candidate_identity(&order).is_err());
    for index in 0..2 {
        let mut grid = valid.clone();
        grid[index].grid.0 += 1;
        assert!(validate_candidate_identity(&grid).is_err());
        let mut block = valid.clone();
        block[index].block.0 -= 1;
        assert!(validate_candidate_identity(&block).is_err());
        let mut shared = valid.clone();
        shared[index].shared ^= 4;
        assert!(validate_candidate_identity(&shared).is_err());
        let mut zero = valid.clone();
        zero[index].arguments_digest = [0; 32];
        assert!(validate_candidate_identity(&zero).is_err());
    }
    let mut collision = valid;
    collision[1].arguments_digest = collision[0].arguments_digest;
    assert!(validate_candidate_identity(&collision).is_err());
}

#[test]
fn timing_gates_reject_invalid_samples_thresholds_and_tiny_windows() {
    assert_eq!(qualification_windows(None).unwrap(), SCREEN_WINDOWS);
    assert_eq!(qualification_windows(Some("21")).unwrap(), SCREEN_WINDOWS);
    assert_eq!(
        qualification_windows(Some("101")).unwrap(),
        OFFICIAL_WINDOWS
    );
    for value in ["0", "1", "20", "22", "100", "102", "bad"] {
        assert!(qualification_windows(Some(value)).is_err());
    }
    assert_eq!(
        required_qualification_windows(None, SCREEN_WINDOWS, "screen").unwrap(),
        SCREEN_WINDOWS
    );
    assert_eq!(
        required_qualification_windows(None, OFFICIAL_WINDOWS, "official").unwrap(),
        OFFICIAL_WINDOWS
    );
    assert!(required_qualification_windows(Some("101"), SCREEN_WINDOWS, "screen").is_err());
    assert!(required_qualification_windows(Some("21"), OFFICIAL_WINDOWS, "official").is_err());
    assert!(percentile(&[], 0.5).is_err());
    for value in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(percentile(&[value], 0.5).is_err());
        assert!(validate_performance_gate("mutation", value, 1.10, 1.20).is_err());
        assert!(validate_performance_gate("mutation", 1.10, value, 1.20).is_err());
        assert!(validate_performance_gate("mutation", 1.10, 1.10, value).is_err());
    }
    assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5).unwrap(), 2.0);
    assert!(validate_performance_gate("test", 1.002, 1.002, 1.10).is_ok());
    assert!(validate_performance_gate("test", 1.001, 1.01, 1.10).is_err());
    assert!(validate_performance_gate("test", 1.01, 1.001, 1.10).is_err());
}

#[cfg(feature = "cuda")]
fn verify_post_cohort_even_on_error<T, U>(
    body: Result<T, String>,
    postflight: Result<U, String>,
) -> Result<T, String> {
    match (body, postflight) {
        (Ok(value), Ok(_)) => Ok(value),
        (Err(body), Ok(_)) => Err(body),
        (Ok(_), Err(postflight)) => Err(postflight),
        (Err(body), Err(postflight)) => Err(format!("{body}; quiet-GPU postflight: {postflight}")),
    }
}

#[cfg(feature = "cuda")]
fn compose_cuda_source() -> String {
    [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/scalar.cu"),
        CANDIDATE_SOURCE,
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

#[cfg(feature = "cuda")]
fn parsed_registers(resource_block: &str) -> usize {
    let (_, tail) = resource_block
        .split_once("Used ")
        .expect("ptxas register line");
    tail.split_whitespace()
        .next()
        .expect("register count")
        .parse()
        .expect("numeric register count")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA 13.2 NVRTC/PTXAS but launches no GPU work"]
fn candidate_compiles_and_meets_sm89_sm120_ptxas_resources() {
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
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
            .unwrap_or_else(|error| panic!("compile tournament for {arch}: {error:?}"))
            .to_src();
        for symbol in [CANDIDATE_SYMBOL, PRODUCTION_PARTIAL_SYMBOL, REDUCER_SYMBOL] {
            assert!(ptx.contains(symbol), "{arch} PTX lost {symbol}");
        }
        assert!(ptx.contains("fma.rn.f32"));
        let stem = format!(
            "mamba-rs-nn-splitk-m32n64-exact-{}-{sm}",
            std::process::id()
        );
        let ptx_path = std::env::temp_dir().join(format!("{stem}.ptx"));
        let cubin_path = std::env::temp_dir().join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx).expect("write PTX");
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
        eprintln!("NN Split-K M32N64 exact-shape resources {sm}:\n{log}");
        assert!(output.status.success(), "ptxas {sm} failed:\n{log}");
        for (symbol, cap, smem) in [
            (CANDIDATE_SYMBOL, 64, 13_312),
            (PRODUCTION_PARTIAL_SYMBOL, 64, 13_312),
            (REDUCER_SYMBOL, 72, 0),
        ] {
            let marker = format!("Compiling entry function '{symbol}'");
            let block = log
                .split_once(&marker)
                .unwrap_or_else(|| panic!("{sm} lost {symbol}"))
                .1
                .split("Compiling entry function '")
                .next()
                .unwrap();
            assert!(
                block.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"),
                "{sm} {symbol}:\n{block}"
            );
            assert!(parsed_registers(block) <= cap, "{sm} {symbol}:\n{block}");
            if smem == 0 {
                assert!(!block.contains("bytes smem"), "{sm} {symbol}:\n{block}");
            } else {
                assert!(
                    block.contains(&format!("{smem} bytes smem")),
                    "{sm} {symbol}:\n{block}"
                );
            }
        }
        let _ = std::fs::remove_file(ptx_path);
        let _ = std::fs::remove_file(cubin_path);
    }
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use std::ffi::CStr;
    use std::mem::size_of;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    };
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;
    use super::*;

    const GUARD: usize = 64;
    const INPUT_GUARD: u32 = 0x7fc0_a128;
    const OUTPUT_GUARD: u32 = 0x7fc0_c128;
    const TARGET_WINDOW_US: f64 = 10_000.0;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        Candidate,
    }

    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Self::Production => "production_splitk32_m32n64",
                Self::Candidate => "candidate_splitk32_m32n64_exactshape",
            }
        }
    }

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
        symbol: &'static str,
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _module: Arc<CudaModule>,
        production_partial: Kernel,
        candidate_partial: Kernel,
        reducer: Kernel,
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        offset: usize,
        len: usize,
        guard: u32,
    }

    impl GuardedBuffer {
        fn new(stream: &Arc<CudaStream>, active: Vec<f32>, guard: u32) -> Result<Self, String> {
            let len = active.len();
            let offset = GUARD;
            let total = offset
                .checked_add(len)
                .and_then(|value| value.checked_add(GUARD))
                .ok_or_else(|| "guarded allocation size overflow".to_string())?;
            let mut expected = vec![f32::from_bits(guard); total];
            expected[offset..offset + len].copy_from_slice(&active);
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
            self.buffer.upload(stream, &self.expected)
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            for (index, value) in values[..self.offset]
                .iter()
                .chain(&values[self.offset + self.len..])
                .enumerate()
            {
                if value.to_bits() != self.guard {
                    return Err(format!("{label} red zone changed at guard element {index}"));
                }
            }
            Ok(values[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let values = self.buffer.to_cpu(stream)?;
            if values
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} input or red zone changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        production_partial: GuardedBuffer,
        candidate_partial: GuardedBuffer,
        production_output: GuardedBuffer,
        candidate_output: GuardedBuffer,
    }

    impl Fixture {
        fn partial(&self, arm: Arm) -> &GuardedBuffer {
            match arm {
                Arm::Production => &self.production_partial,
                Arm::Candidate => &self.candidate_partial,
            }
        }
        fn partial_mut(&mut self, arm: Arm) -> &mut GuardedBuffer {
            match arm {
                Arm::Production => &mut self.production_partial,
                Arm::Candidate => &mut self.candidate_partial,
            }
        }
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
        fn reset_arm(&mut self, runtime: &Runtime, arm: Arm) -> Result<(), String> {
            self.partial_mut(arm).reset(&runtime.ctx.stream)?;
            self.output_mut(arm).reset(&runtime.ctx.stream)
        }
        fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
            self.a.validate_unchanged(&runtime.ctx.stream, "A")?;
            self.b.validate_unchanged(&runtime.ctx.stream, "B")
        }
    }

    fn validate_qualified_environment(
        cc: (u32, u32),
        sms: u32,
        nvrtc_target: &str,
        compiler: CompilerIdentity,
        artifact: ArtifactIdentity,
    ) -> Result<(), String> {
        if cc != (12, 0)
            || sms != 170
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
                "tournament requires qualified compute_120 NVRTC13.2 TriadScalar: cc={cc:?} sms={sms} target={nvrtc_target} compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
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
                .map_err(|error| format!("set {symbol} shared: {error:?}"))?;
        }
        Ok(Kernel {
            function,
            config,
            symbol,
        })
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
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
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
            .map_err(|error| format!("compile tournament: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let digest: [u8; 32] = Sha256::digest(ptx_source.as_bytes()).into();
        if digest == [0; 32] {
            return Err("candidate PTX digest is zero".into());
        }
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load tournament: {error:?}"))?;
        let production_partial = load_kernel(
            &module,
            PRODUCTION_PARTIAL_SYMBOL,
            LaunchConfig {
                grid_dim: PRODUCTION_GRID,
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        let candidate_partial = load_kernel(
            &module,
            CANDIDATE_SYMBOL,
            LaunchConfig {
                grid_dim: CANDIDATE_GRID,
                block_dim: (128, 1, 1),
                shared_mem_bytes: CANDIDATE_SHARED,
            },
            CANDIDATE_SHARED as usize,
        )?;
        let reducer = load_kernel(
            &module,
            REDUCER_SYMBOL,
            LaunchConfig {
                grid_dim: REDUCER_GRID,
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            production_partial,
            candidate_partial,
            reducer,
        })
    }

    fn seeded_values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4093 == 0 {
                    -0.0
                } else {
                    let signed = ((state >> 32) as u32 % 2049) as i32 - 1024;
                    signed as f32 * (0.25 / 1024.0)
                }
            })
            .collect()
    }

    fn new_fixture_with_values(
        runtime: &Runtime,
        a: Vec<f32>,
        b: Vec<f32>,
    ) -> Result<Fixture, String> {
        if a.len() != DIMS.0 * DIMS.1 || b.len() != DIMS.1 * DIMS.2 {
            return Err("fixture dimension mismatch".into());
        }
        let partial = vec![0.0; PARTIAL_ELEMENTS];
        let output = vec![0.0; DIMS.0 * DIMS.2];
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a, INPUT_GUARD)?,
            b: GuardedBuffer::new(&runtime.ctx.stream, b, INPUT_GUARD)?,
            production_partial: GuardedBuffer::new(
                &runtime.ctx.stream,
                partial.clone(),
                OUTPUT_GUARD,
            )?,
            candidate_partial: GuardedBuffer::new(&runtime.ctx.stream, partial, OUTPUT_GUARD)?,
            production_output: GuardedBuffer::new(
                &runtime.ctx.stream,
                output.clone(),
                OUTPUT_GUARD,
            )?,
            candidate_output: GuardedBuffer::new(&runtime.ctx.stream, output, OUTPUT_GUARD)?,
        })
    }

    fn new_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        new_fixture_with_values(
            runtime,
            seeded_values(DIMS.0 * DIMS.1, 0xa128_0001),
            seeded_values(DIMS.1 * DIMS.2, 0xb128_0002),
        )
    }

    fn exceptional_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let mut a = vec![0.0; DIMS.0 * DIMS.1];
        for row in 0..DIMS.0 {
            a[row * DIMS.1] = 1.0;
        }
        let mut b = vec![0.0; DIMS.1 * DIMS.2];
        for (column, bits) in [
            0x7fc0_1234_u32,
            0x7f80_0000,
            0xff80_0000,
            0x8000_0000,
            0x0000_0001,
        ]
        .into_iter()
        .enumerate()
        {
            b[column] = f32::from_bits(bits);
        }
        new_fixture_with_values(runtime, a, b)
    }

    fn launch_partial(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let kernel = match arm {
            Arm::Production => &runtime.production_partial,
            Arm::Candidate => &runtime.candidate_partial,
        };
        let partial = fixture.partial(arm).ptr(&runtime.ctx.stream);
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
        let m = DIMS.0 as i32;
        let n = DIMS.2 as i32;
        let chunks = K_CHUNKS as i32;
        let lda = DIMS.1 as i32;
        let mut builder = runtime.ctx.stream.launch_builder(&kernel.function);
        builder.arg(&partial);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&chunks);
        builder.arg(&lda);
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", kernel.symbol))
    }

    fn launch_reducer(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let output = fixture.output(arm).ptr(&runtime.ctx.stream);
        let partial = fixture.partial(arm).ptr(&runtime.ctx.stream);
        let null = 0_u64;
        let alpha = 1.0_f32;
        let m = DIMS.0 as i32;
        let n = DIMS.2 as i32;
        let chunks = K_CHUNKS as i32;
        let zero = 0_i32;
        let mut builder = runtime.ctx.stream.launch_builder(&runtime.reducer.function);
        builder.arg(&output);
        builder.arg(&partial);
        builder.arg(&null);
        builder.arg(&null);
        builder.arg(&null);
        builder.arg(&alpha);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&chunks);
        builder.arg(&zero);
        builder.arg(&zero);
        builder.arg(&zero);
        unsafe { builder.launch(runtime.reducer.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {REDUCER_SYMBOL}: {error:?}"))
    }

    fn launch_arm(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        launch_partial(runtime, fixture, arm)?;
        launch_reducer(runtime, fixture, arm)
    }

    fn bytes_of<T: DeviceRepr>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for argument in arguments {
            hasher.update((argument.len() as u64).to_le_bytes());
            hasher.update(argument);
        }
        hasher.finalize().into()
    }

    fn eager_candidate_identity(runtime: &Runtime, fixture: &Fixture) -> Vec<NodeIdentity> {
        let partial = fixture.candidate_partial.ptr(&runtime.ctx.stream);
        let output = fixture.candidate_output.ptr(&runtime.ctx.stream);
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
        let null = 0_u64;
        let alpha = 1.0_f32;
        let m = DIMS.0 as i32;
        let n = DIMS.2 as i32;
        let chunks = K_CHUNKS as i32;
        let lda = DIMS.1 as i32;
        let zero = 0_i32;
        vec![
            NodeIdentity {
                symbol: CANDIDATE_SYMBOL.into(),
                grid: CANDIDATE_GRID,
                block: (128, 1, 1),
                shared: CANDIDATE_SHARED,
                arguments_digest: digest_arguments(&[
                    bytes_of(&partial),
                    bytes_of(&a),
                    bytes_of(&b),
                    bytes_of(&m),
                    bytes_of(&n),
                    bytes_of(&chunks),
                    bytes_of(&lda),
                ]),
            },
            NodeIdentity {
                symbol: REDUCER_SYMBOL.into(),
                grid: REDUCER_GRID,
                block: (256, 1, 1),
                shared: 0,
                arguments_digest: digest_arguments(&[
                    bytes_of(&output),
                    bytes_of(&partial),
                    bytes_of(&null),
                    bytes_of(&null),
                    bytes_of(&null),
                    bytes_of(&alpha),
                    bytes_of(&m),
                    bytes_of(&n),
                    bytes_of(&chunks),
                    bytes_of(&zero),
                    bytes_of(&zero),
                    bytes_of(&zero),
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
            return Err(format!("non-kernel graph node {kind:?}"));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "graph params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "graph symbol",
        )?;
        if name.is_null() {
            return Err("null graph symbol".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph symbol UTF-8: {error}"))?
            .to_owned();
        let sizes: &[usize] = match symbol.as_str() {
            CANDIDATE_SYMBOL => &[8, 8, 8, 4, 4, 4, 4],
            REDUCER_SYMBOL => &[8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4],
            _ => return Err(format!("unexpected graph symbol {symbol}")),
        };
        if params.kernelParams.is_null() {
            return Err(format!("{symbol} graph arguments are null"));
        }
        let mut arguments = Vec::with_capacity(sizes.len());
        for (index, size) in sizes.iter().copied().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{symbol} graph argument {index} is null"));
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

    fn graph_candidate_identity(graph: &CudaGraph) -> Result<Vec<NodeIdentity>, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "graph nodes",
        )?;
        if node_count != 2 {
            return Err(format!("candidate graph has {node_count} nodes"));
        }
        let mut edge_count = 0;
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
            "graph edges",
        )?;
        if edge_count != 1 {
            return Err(format!("candidate graph has {edge_count} edges"));
        }
        let mut from = [std::ptr::null_mut()];
        let mut to = [std::ptr::null_mut()];
        let mut edge_data: [sys::CUgraphEdgeData; 1] = [unsafe { std::mem::zeroed() }];
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
            "graph edge",
        )?;
        let edge = edge_data[0];
        if edge.from_port != 0
            || edge.to_port != 0
            || edge.type_ != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
            || edge.reserved != [0; 5]
        {
            return Err("candidate graph edge is not the CUDA default".into());
        }
        let nodes = vec![unsafe { graph_node_identity(from[0]) }?, unsafe {
            graph_node_identity(to[0])
        }?];
        validate_candidate_identity(&nodes)?;
        Ok(nodes)
    }

    fn capture(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch_arm(runtime, fixture, arm)) }
    }

    fn compare_entire_partial_tensor(
        runtime: &Runtime,
        fixture: &Fixture,
    ) -> Result<Vec<u32>, String> {
        let production = fixture
            .production_partial
            .active_bits(&runtime.ctx.stream, "production partial")?;
        let candidate = fixture
            .candidate_partial
            .active_bits(&runtime.ctx.stream, "candidate partial")?;
        if production.len() != PARTIAL_ELEMENTS || candidate.len() != PARTIAL_ELEMENTS {
            return Err("partial tensor length changed".into());
        }
        if let Some((index, (expected, actual))) = production
            .iter()
            .zip(&candidate)
            .enumerate()
            .find(|(_, (expected, actual))| expected != actual)
        {
            return Err(format!(
                "partial differs at {index}: production=0x{expected:08x} candidate=0x{actual:08x}"
            ));
        }
        Ok(production)
    }

    fn validate_partial_oracle_spots(fixture: &Fixture, partial: &[u32]) -> Result<(), String> {
        let a = &fixture.a.expected[fixture.a.offset..fixture.a.offset + fixture.a.len];
        let b = &fixture.b.expected[fixture.b.offset..fixture.b.offset + fixture.b.len];
        for chunk in [0, 1, 127, 255] {
            for row in [0, 63, 64, 127] {
                for column in [0, 31, 64, 127] {
                    let mut sum = 0.0_f32;
                    for local_k in 0..32 {
                        let k = chunk * 32 + local_k;
                        sum = a[row * DIMS.1 + k].mul_add(b[k * DIMS.2 + column], sum);
                    }
                    let index = chunk * DIMS.0 * DIMS.2 + row * DIMS.2 + column;
                    if partial[index] != sum.to_bits() {
                        return Err(format!(
                            "partial oracle differs chunk={chunk} row={row} column={column}: actual=0x{:08x} expected=0x{:08x}",
                            partial[index],
                            sum.to_bits()
                        ));
                    }
                }
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
        let registers = kernel
            .function
            .num_regs()
            .map_err(|error| format!("registers: {error:?}"))?;
        let local = kernel
            .function
            .local_size_bytes()
            .map_err(|error| format!("local: {error:?}"))?;
        let static_shared = kernel
            .function
            .shared_size_bytes()
            .map_err(|error| format!("shared: {error:?}"))?;
        let threads =
            kernel.config.block_dim.0 * kernel.config.block_dim.1 * kernel.config.block_dim.2;
        let occupancy = kernel
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                kernel.config.shared_mem_bytes as usize,
                None,
            )
            .map_err(|error| format!("occupancy: {error:?}"))?;
        eprintln!(
            "nn_splitk_m32n64_exactshape resource symbol={} threads={} registers={} local={} static_shared={} dynamic_shared={} occupancy={}",
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
            return Err(format!("{} resource contract failed", kernel.symbol));
        }
        Ok(())
    }

    fn validate_exceptional_output(bits: &[u32]) -> Result<(), String> {
        for row in [0, DIMS.0 - 1] {
            let base = row * DIMS.2;
            if !f32::from_bits(bits[base]).is_nan()
                || bits[base + 1] != 0x7f80_0000
                || bits[base + 2] != 0xff80_0000
                || bits[base + 3] != 0x0000_0000
                || bits[base + 4] != 0x0000_0001
            {
                return Err(format!(
                    "exceptional output changed at row {row}: {:08x?}",
                    &bits[base..base + 5]
                ));
            }
        }
        Ok(())
    }

    fn exactness_case(
        runtime: &Runtime,
        fixture: &mut Fixture,
        exceptional: bool,
    ) -> Result<(), String> {
        fixture.reset_arm(runtime, Arm::Production)?;
        fixture.reset_arm(runtime, Arm::Candidate)?;
        launch_partial(runtime, fixture, Arm::Production)?;
        launch_partial(runtime, fixture, Arm::Candidate)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("partial sync: {error:?}"))?;
        let partial = compare_entire_partial_tensor(runtime, fixture)?;
        if !exceptional {
            validate_partial_oracle_spots(fixture, &partial)?;
        }

        launch_reducer(runtime, fixture, Arm::Production)?;
        launch_reducer(runtime, fixture, Arm::Candidate)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("reducer sync: {error:?}"))?;
        let production_bits = fixture
            .production_output
            .active_bits(&runtime.ctx.stream, "production output")?;
        let candidate_bits = fixture
            .candidate_output
            .active_bits(&runtime.ctx.stream, "candidate output")?;
        if candidate_bits != production_bits {
            return Err("candidate final output differs from production bits".into());
        }
        if exceptional {
            validate_exceptional_output(&candidate_bits)?;
        }

        let eager_identity = eager_candidate_identity(runtime, fixture);
        validate_candidate_identity(&eager_identity)?;
        for _ in 0..3 {
            fixture.reset_arm(runtime, Arm::Candidate)?;
            launch_arm(runtime, fixture, Arm::Candidate)?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("eager repeat: {error:?}"))?;
            let repeat = fixture
                .candidate_output
                .active_bits(&runtime.ctx.stream, "candidate eager repeat")?;
            if repeat != production_bits {
                return Err("candidate eager repeat differs".into());
            }
        }

        let production_graph = capture(runtime, fixture, Arm::Production)?;
        let candidate_graph = capture(runtime, fixture, Arm::Candidate)?;
        let graph_identity = graph_candidate_identity(&candidate_graph)?;
        if graph_identity != eager_identity {
            return Err(format!(
                "candidate eager/graph identities differ: eager={eager_identity:?} graph={graph_identity:?}"
            ));
        }
        for (arm, graph) in [
            (Arm::Production, &production_graph),
            (Arm::Candidate, &candidate_graph),
        ] {
            for _ in 0..3 {
                fixture.reset_arm(runtime, arm)?;
                graph
                    .launch()
                    .map_err(|error| format!("{} graph: {error:?}", arm.name()))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{} graph sync: {error:?}", arm.name()))?;
                let bits = fixture
                    .output(arm)
                    .active_bits(&runtime.ctx.stream, arm.name())?;
                if bits != production_bits {
                    return Err(format!("{} graph repeat differs", arm.name()));
                }
                if arm == Arm::Candidate {
                    compare_entire_partial_tensor(runtime, fixture)?;
                }
            }
        }
        fixture.validate_inputs(runtime)
    }

    fn measure(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be nonzero".into());
        }
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("timing start: {error:?}"))?;
        for _ in 0..iterations {
            launch_arm(runtime, fixture, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("timing end: {error:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("timing elapsed: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        positive_finite(us, "per-call timing")
    }

    fn calibrated_iterations(runtime: &Runtime, fixture: &Fixture) -> Result<usize, String> {
        let production = measure(runtime, fixture, Arm::Production, 3)?;
        let candidate = measure(runtime, fixture, Arm::Candidate, 3)?;
        Ok((TARGET_WINDOW_US / production.min(candidate))
            .round()
            .clamp(3.0, 500.0) as usize)
    }

    struct PairedSummary {
        order: &'static str,
        p05: f64,
        p50: f64,
        p95: f64,
    }

    fn paired(
        runtime: &Runtime,
        fixture: &Fixture,
        windows: usize,
    ) -> Result<Vec<PairedSummary>, String> {
        let iterations = calibrated_iterations(runtime, fixture)?;
        let mut summaries = Vec::with_capacity(2);
        for (order, baseline_first) in [("ABBA", true), ("BAAB", false)] {
            let mut production_samples = Vec::with_capacity(windows);
            let mut candidate_samples = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (p0, p1, c0, c1) = if baseline_first {
                    let p0 = measure(runtime, fixture, Arm::Production, iterations)?;
                    let c0 = measure(runtime, fixture, Arm::Candidate, iterations)?;
                    let c1 = measure(runtime, fixture, Arm::Candidate, iterations)?;
                    let p1 = measure(runtime, fixture, Arm::Production, iterations)?;
                    (p0, p1, c0, c1)
                } else {
                    let c0 = measure(runtime, fixture, Arm::Candidate, iterations)?;
                    let p0 = measure(runtime, fixture, Arm::Production, iterations)?;
                    let p1 = measure(runtime, fixture, Arm::Production, iterations)?;
                    let c1 = measure(runtime, fixture, Arm::Candidate, iterations)?;
                    (p0, p1, c0, c1)
                };
                let production_us = positive_finite(0.5 * (p0 + p1), "paired production timing")?;
                let candidate_us = positive_finite(0.5 * (c0 + c1), "paired candidate timing")?;
                production_samples.push(production_us);
                candidate_samples.push(candidate_us);
                ratios.push(positive_finite(
                    production_us / candidate_us,
                    "paired speedup",
                )?);
            }
            let production_p50 = percentile(&production_samples, 0.50)?;
            let candidate_p50 = percentile(&candidate_samples, 0.50)?;
            let p05 = percentile(&ratios, 0.05)?;
            let p50 = percentile(&ratios, 0.50)?;
            let p95 = percentile(&ratios, 0.95)?;
            eprintln!(
                "nn_splitk_m32n64_exactshape baseline={} candidate={} order={} windows={} production_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9} equal_iterations={}",
                Arm::Production.name(),
                Arm::Candidate.name(),
                order,
                windows,
                production_p50,
                candidate_p50,
                p05,
                p50,
                p95,
                iterations
            );
            summaries.push(PairedSummary {
                order,
                p05,
                p50,
                p95,
            });
        }
        Ok(summaries)
    }

    #[test]
    fn qualified_environment_rejects_mutations() {
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
        let compiler_mutations: [fn(&mut CompilerIdentity); 12] = [
            |value| value.source_digest = [0; 32],
            |value| value.invocation_digest = [0; 32],
            |value| value.header_manifest_digest = [0; 32],
            |value| value.target = CudaTarget::new("compute_89").unwrap(),
            |value| value.nvrtc_version = (13, 1),
            |value| value.nvrtc_library_domain = [0; 32],
            |value| value.nvrtc_library_known = false,
            |value| value.output_kind = ArtifactKind::Cubin,
            |value| value.composer_revision ^= 1,
            |value| value.compiler_revision ^= 1,
            |value| value.numeric_abi_revision ^= 1,
            |value| value.schedule_revision ^= 1,
        ];
        for mutate in compiler_mutations {
            let mut bad = compiler;
            mutate(&mut bad);
            assert!(
                validate_qualified_environment((12, 0), 170, "compute_120", bad, artifact).is_err()
            );
        }
        for bad_artifact in [
            ArtifactIdentity {
                module_kind: ModuleKind::TriadSm80,
                ..artifact
            },
            ArtifactIdentity {
                artifact_kind: ArtifactKind::Cubin,
                ..artifact
            },
            ArtifactIdentity {
                compile_key: [0; 32],
                ..artifact
            },
            ArtifactIdentity {
                compile_key: [9; 32],
                ..artifact
            },
            ArtifactIdentity {
                artifact_digest: [0; 32],
                ..artifact
            },
        ] {
            assert!(
                validate_qualified_environment((12, 0), 170, "compute_120", compiler, bad_artifact)
                    .is_err()
            );
        }
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM GPU"]
    fn candidate_partial_and_final_bits_are_exact_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        check_resources(
            &runtime.production_partial,
            ResourceContract {
                threads: 128,
                static_shared: 13_312,
                dynamic_shared: 0,
                register_cap: 64,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.candidate_partial,
            ResourceContract {
                threads: 128,
                static_shared: 13_312,
                dynamic_shared: 0,
                register_cap: 64,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.reducer,
            ResourceContract {
                threads: 256,
                static_shared: 0,
                dynamic_shared: 0,
                register_cap: 72,
                minimum_occupancy: 3,
            },
        )?;
        let mut finite = new_fixture(&runtime)?;
        exactness_case(&runtime, &mut finite, false)?;
        let mut exceptional = exceptional_fixture(&runtime)?;
        exactness_case(&runtime, &mut exceptional, true)
    }

    fn configured_windows(expected: usize, label: &str) -> Result<usize, String> {
        let value = match std::env::var("MAMBA_RS_NN_SPLITK_M32N64_EXACTSHAPE_WINDOWS") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("MAMBA_RS_NN_SPLITK_M32N64_EXACTSHAPE_WINDOWS is not Unicode".into());
            }
        };
        required_qualification_windows(value.as_deref(), expected, label)
    }

    fn run_timing(windows: usize, label: &str) -> Result<(), String> {
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context(label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            let fixture = new_fixture(&runtime)?;
            for _ in 0..10 {
                launch_arm(&runtime, &fixture, Arm::Production)?;
                launch_arm(&runtime, &fixture, Arm::Candidate)?;
            }
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("warmup: {error:?}"))?;
            quiet.require_cohort(label)?;
            for summary in paired(&runtime, &fixture, windows)? {
                validate_performance_gate(summary.order, summary.p05, summary.p50, summary.p95)?;
            }
            fixture.validate_inputs(&runtime)?;
            for arm in [Arm::Production, Arm::Candidate] {
                fixture
                    .partial(arm)
                    .active_bits(&runtime.ctx.stream, "timing partial guard")?;
                fixture
                    .output(arm)
                    .active_bits(&runtime.ctx.stream, "timing output guard")?;
            }
            Ok(())
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(label).map(drop))
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM GPU"]
    fn candidate_vs_production_21_window_abba_baab() -> Result<(), String> {
        let label = "nn-splitk-m32n64-exactshape-screen-21";
        let windows = configured_windows(SCREEN_WINDOWS, label)?;
        run_timing(windows, label)
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM GPU"]
    fn candidate_vs_production_official_101_window_abba_baab() -> Result<(), String> {
        let label = "nn-splitk-m32n64-exactshape-official-101";
        let windows = configured_windows(OFFICIAL_WINDOWS, label)?;
        run_timing(windows, label)
    }
}
