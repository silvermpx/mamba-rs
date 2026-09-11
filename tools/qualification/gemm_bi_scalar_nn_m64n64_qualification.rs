const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";
const FIXED_COPYPLAN_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
const GENERIC_SYMBOL: &str = "gemm_bi_nn";
const PRODUCTION_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/scalar_nn_m64n64.cu");
const TEST_SOURCE: &str = include_str!("gemm_bi_scalar_nn_m64n64_qualification.rs");
#[cfg(feature = "cuda")]
const SCALAR_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/scalar.cu");
#[cfg(feature = "cuda")]
#[path = "../../tests/support/fixed_full_mantissa.rs"]
mod fixed_full_mantissa;
#[cfg(feature = "cuda")]
#[path = "../../tests/common/gpu_quiet.rs"]
mod gpu_quiet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaShortScreenCell {
    id: &'static str,
    dims: (usize, usize, usize),
    symbol: &'static str,
}

fn ada_short_screen_cells() -> Vec<AdaShortScreenCell> {
    vec![
        AdaShortScreenCell {
            id: "d768_in_proj",
            dims: (2_048, 768, 3_072),
            symbol: PRODUCTION_SYMBOL,
        },
        AdaShortScreenCell {
            id: "d768_out_proj",
            dims: (2_048, 1_536, 768),
            symbol: PRODUCTION_SYMBOL,
        },
        AdaShortScreenCell {
            id: "prism_in_proj",
            dims: (4_621, 384, 1_928),
            symbol: PRODUCTION_SYMBOL,
        },
    ]
}

fn ada_short_screen_retains(strata: &[[f64; 2]]) -> bool {
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

fn ada_copyplan_screen_cells() -> Vec<AdaShortScreenCell> {
    ada_short_screen_cells()
        .into_iter()
        .map(|cell| AdaShortScreenCell {
            symbol: FIXED_COPYPLAN_SYMBOL,
            ..cell
        })
        .collect()
}

/// The deep cell the Triad family also hands to the copy plan: a 4096-row
/// product with a 3072-long reduction, screened on its own after the three
/// retained cells.
fn ada_copyplan_deep_cells() -> Vec<AdaShortScreenCell> {
    vec![AdaShortScreenCell {
        id: "large_deep",
        dims: (4_096, 3_072, 1_536),
        symbol: FIXED_COPYPLAN_SYMBOL,
    }]
}

fn ada_candidate_shared(symbol: &str) -> Result<(usize, usize), String> {
    match symbol {
        PRODUCTION_SYMBOL => Ok((0, 17_408)),
        FIXED_COPYPLAN_SYMBOL => Ok((32_768, 0)),
        _ => Err(format!("unknown Ada exact NN candidate {symbol}")),
    }
}

fn copyplan_epilogue_supported(alpha: f32, beta: f32, has_bias: bool) -> bool {
    alpha.to_bits() == 1.0_f32.to_bits() && beta.to_bits() == 0 && !has_bias
}

fn ada_live_toolkit_supported(cc: (u32, u32), sms: u32, nvrtc: (i32, i32)) -> bool {
    cc == (8, 9) && sms == 142 && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveCopyPlanStage {
    Discovery,
    Integrated,
}

impl LiveCopyPlanStage {
    fn expected_auto_symbol(self) -> &'static str {
        match self {
            Self::Discovery => GENERIC_SYMBOL,
            Self::Integrated => FIXED_COPYPLAN_SYMBOL,
        }
    }

    fn windows(self) -> usize {
        match self {
            Self::Discovery => 7,
            Self::Integrated => 21,
        }
    }
}

#[test]
fn integrated_copyplan_must_use_actual_fixed_auto_and_confirmation_windows() {
    assert_eq!(
        LiveCopyPlanStage::Discovery.expected_auto_symbol(),
        GENERIC_SYMBOL
    );
    assert_eq!(LiveCopyPlanStage::Discovery.windows(), 7);
    assert_eq!(
        LiveCopyPlanStage::Integrated.expected_auto_symbol(),
        FIXED_COPYPLAN_SYMBOL
    );
    assert_eq!(LiveCopyPlanStage::Integrated.windows(), 21);
}

#[test]
fn live_comparison_releases_the_qualified_lease_before_other_cuda_work() {
    let (_, body) = TEST_SOURCE
        .split_once("\n    fn run_live_copyplan_comparison(")
        .unwrap();
    let (_, lease_body) = body
        .split_once("let mut auto = qualify_physical_launch(")
        .unwrap();
    let (isolated_preflight, released) = lease_body
        .split_once("drop(auto);")
        .expect("the qualification lease must end before comparative timing");
    for unrelated in [
        "live_fast_launch(",
        "reset_ada_observation(",
        "check_ada_pair(",
    ] {
        assert!(
            !isolated_preflight.contains(unrelated),
            "unrelated CUDA work inside lease: {unrelated}"
        );
    }
    assert!(released.contains("PublicAutoFixture::new("));
    assert!(released.contains("validate_public_auto_graph("));
    assert!(!released.contains("auto.measure_eager_window_ms("));
    assert!(!released.contains("auto.measure_graph_window_ms("));
}

#[test]
fn live_copyplan_comparison_only_accepts_the_three_ada_toolkit_lanes() {
    for version in [(12, 8), (13, 0), (13, 2)] {
        assert!(ada_live_toolkit_supported((8, 9), 142, version));
        assert!(!ada_live_toolkit_supported((12, 0), 170, version));
        assert!(!ada_live_toolkit_supported((8, 9), 128, version));
    }
    for version in [(12, 7), (12, 9), (13, 1), (13, 3)] {
        assert!(!ada_live_toolkit_supported((8, 9), 142, version));
    }
}

#[test]
fn copyplan_reuse_declines_every_unqualified_epilogue() {
    assert!(copyplan_epilogue_supported(1.0, 0.0, false));
    for (alpha, beta, bias) in [
        (-0.75, 0.5, false),
        (1.0, 1.0, false),
        (1.0, 0.0, true),
        (1.0, -0.0, false),
        (f32::NAN, 0.0, false),
        (1.0, f32::NAN, false),
    ] {
        assert!(!copyplan_epilogue_supported(alpha, beta, bias));
    }
}

#[test]
fn copyplan_discovery_uses_its_static_shared_abi_for_all_three_cells() {
    let cells = ada_copyplan_screen_cells();
    assert_eq!(cells.len(), 3);
    for (cell, dims) in cells.iter().zip([
        (2_048, 768, 3_072),
        (2_048, 1_536, 768),
        (4_621, 384, 1_928),
    ]) {
        assert_eq!(cell.dims, dims);
        assert_eq!(cell.symbol, FIXED_COPYPLAN_SYMBOL);
        assert_eq!(ada_candidate_shared(cell.symbol).unwrap(), (32_768, 0));
    }
    assert_eq!(
        ada_candidate_shared(PRODUCTION_SYMBOL).unwrap(),
        (0, 17_408)
    );
    assert!(ada_candidate_shared("unknown").is_err());
}

#[test]
fn ada_short_screen_cells_and_direction_are_exact() {
    assert_eq!(
        ada_short_screen_cells(),
        [
            AdaShortScreenCell {
                id: "d768_in_proj",
                dims: (2_048, 768, 3_072),
                symbol: PRODUCTION_SYMBOL,
            },
            AdaShortScreenCell {
                id: "d768_out_proj",
                dims: (2_048, 1_536, 768),
                symbol: PRODUCTION_SYMBOL,
            },
            AdaShortScreenCell {
                id: "prism_in_proj",
                dims: (4_621, 384, 1_928),
                symbol: PRODUCTION_SYMBOL,
            },
        ]
    );
    assert!(ada_short_screen_retains(&[[0.98, 0.989]; 4]));
    assert!(!ada_short_screen_retains(&[]));
    assert!(!ada_short_screen_retains(&[[0.98, 0.989]; 3]));
    assert!(!ada_short_screen_retains(&[[0.98, 0.989]; 5]));
    assert!(!ada_short_screen_retains(&[[-0.01, 0.98]; 4]));
    assert!(!ada_short_screen_retains(&[[0.0, 0.98]; 4]));
    assert!(!ada_short_screen_retains(&[[0.98, 0.99]; 4]));
    assert!(!ada_short_screen_retains(&[[0.98, f64::NAN]; 4]));
}

#[test]
fn production_source_is_an_isolated_one_owner_exact_kernel() {
    let source = PRODUCTION_SOURCE;
    assert_eq!(
        source
            .matches(&format!("void {PRODUCTION_SYMBOL}("))
            .count(),
        1
    );
    for required in [
        "#define EXACT_NN_M64N64_BM 64",
        "#define EXACT_NN_M64N64_BN 64",
        "#define EXACT_NN_M64N64_BK 16",
        "#define EXACT_NN_M64N64_THREADS 128",
        "threadResults[idx] = __fmaf_rn(",
        "cp.async.ca.shared.global",
        "int tile_id = blockIdx.x;",
    ] {
        assert!(source.contains(required), "production lost {required}");
    }
    for forbidden in ["atomic", "split_k", "mma.sync"] {
        assert!(
            !source.contains(forbidden),
            "production contains forbidden token {forbidden}"
        );
    }
    let (_, signature_tail) = source
        .split_once(&format!("void {PRODUCTION_SYMBOL}("))
        .expect("production signature");
    let (parameters, _) = signature_tail
        .split_once(") {")
        .expect("production parameter list");
    assert!(parameters.matches(',').count() < 7);
    // The fragment now also owns the prism twin of this schedule; nothing
    // else may join it without being named here.
    assert_eq!(source.matches("threadResults[idx] = __fmaf_rn(").count(), 2);
    assert_eq!(source.matches("extern \"C\" __global__").count(), 2);
    assert!(source.contains("void gemm_bi_nn_prism_m64n64_bk16_s2_v1("));
    let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut ascending = normalized.as_str();
    for token in [
        "for (int tile = 0; tile < num_k_tiles; ++tile)",
        "for (int dot_index = 0; dot_index < EXACT_NN_M64N64_BK; ++dot_index)",
        "for (int result_row = 0; result_row < EXACT_NN_M64N64_TM; ++result_row)",
        "for (int result_column = 0; result_column < EXACT_NN_M64N64_TN; ++result_column)",
        "threadResults[idx] = __fmaf_rn(",
    ] {
        let (_, tail) = ascending
            .split_once(token)
            .unwrap_or_else(|| panic!("production arithmetic order changed at {token}"));
        ascending = tail;
    }
    let registry = include_str!("../../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert!(!registry.contains("gemm_bi_nn_m64n64_bk16_s2_exp_v1"));
    assert!(!PRODUCTION_SOURCE.contains("Experiment"));
    assert!(!PRODUCTION_SOURCE.contains("EXPERIMENT"));
    assert!(!PRODUCTION_SOURCE.contains("_exp_"));
    assert!(!std::path::Path::new("tests/gemm_bi_scalar_nn_m64n64_experiment.cu").exists());
}

#[test]
fn qualification_cells_match_the_exact_nn_performance_matrix() {
    let source = TEST_SOURCE.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in [
        "id: \"large\", dims: (2_048, 3_072, 768)",
        "id: \"large_deep\", dims: (4_096, 3_072, 1_536)",
        "id: \"rect_wide\", dims: (512, 3_072, 768)",
        "id: \"rect_tall\", dims: (4_096, 512, 768)",
        "id: \"d768_in_proj\", dims: (2_048, 768, 3_072)",
        "id: \"d768_out_proj\", dims: (2_048, 1_536, 768)",
        "id: \"prism_in_proj\", dims: (4_621, 384, 1_928)",
    ] {
        assert!(source.contains(required), "qualification lost {required}");
    }
    let (_, cases) = TEST_SOURCE
        .rsplit_once("const CASES:")
        .expect("qualification cases");
    let (cases, _) = cases
        .split_once("enum Arm")
        .expect("qualification case terminator");
    assert_eq!(cases.matches("cpu_oracle: false").count(), 7);
}

#[test]
fn production_scalar_module_owns_the_m64n64_source_and_required_handle() {
    let modules = include_str!("../../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    for required in [
        "kernels/gemm_bi_triad/scalar_nn_m64n64.cu",
        "include_str!(\"../../../../kernels/gemm_bi_triad/scalar_nn_m64n64.cu\")",
        "\"gemm_bi_nn_m64n64_bk16_s2_v1\"",
        "pub gemm_bi_nn_m64n64_bk16_s2_v1: CudaFunction",
        "let gemm_bi_nn_m64n64_bk16_s2_v1 = load(\"gemm_bi_nn_m64n64_bk16_s2_v1\")?",
        "gemm_bi_nn_m64n64_bk16_s2_v1,",
    ] {
        assert!(
            modules.contains(required),
            "production wiring lost {required}"
        );
    }
}

#[cfg(feature = "cuda")]
fn compose_cuda_source() -> String {
    [
        include_str!("../../kernels/_typed_prelude.cuh"),
        include_str!("../../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../../kernels/gemm_bi_triad/epilogue.cuh"),
        SCALAR_SOURCE,
        PRODUCTION_SOURCE,
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
#[test]
#[ignore = "requires CUDA NVRTC but does not launch GPU work"]
fn production_compiles_for_the_portable_compute80_floor() {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_80"),
        options: vec![
            "--fmad=true".to_owned(),
            "--extra-device-vectorization".to_owned(),
            "-DNDEBUG".to_owned(),
            "-DGEMM_BI_GROUP_M=8".to_owned(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
        .expect("M64N64 exact-F32 production must compile for compute_80");
    let ptx = image.to_src();
    assert!(ptx.contains(PRODUCTION_SYMBOL));
    assert!(ptx.contains("fma.rn.f32"));
}

#[cfg(feature = "cuda")]
mod cuda_qualification {
    use mamba_rs::mamba_ssm::gpu::GemmMode;
    use std::sync::Arc;

    use super::gpu_quiet::QuietGpu;
    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

    use super::{
        AdaShortScreenCell, FIXED_COPYPLAN_SYMBOL, GENERIC_SYMBOL, LiveCopyPlanStage,
        PRODUCTION_SYMBOL, ada_candidate_shared, ada_copyplan_deep_cells,
        ada_copyplan_screen_cells, ada_live_toolkit_supported, ada_short_screen_cells,
        ada_short_screen_retains, compose_cuda_source, copyplan_epilogue_supported,
        fixed_full_mantissa,
    };

    const GUARD_ELEMENTS: usize = 32;
    const INPUT_GUARD_BITS: u32 = 0x7fc0_a651;
    const OUTPUT_GUARD_BITS: u32 = 0x7fc0_c651;
    const PRODUCTION_SHARED_BYTES: usize = 17_408;
    const GENERIC_SHARED_BYTES: usize = 34 * 1_024;
    const EXPECTED_REGISTERS: usize = 103;
    const ADA_WINDOWS: usize = 7;
    const ADA_WARMUPS: usize = 8;

    #[derive(Clone, Copy)]
    struct Case {
        id: &'static str,
        dims: (usize, usize, usize),
        alpha: f32,
        beta: f32,
        bias: bool,
        cpu_oracle: bool,
    }

    const CASES: [Case; 8] = [
        Case {
            id: "tail_bias_beta",
            dims: (67, 19, 137),
            alpha: 1.0,
            beta: -0.25,
            bias: true,
            cpu_oracle: true,
        },
        Case {
            id: "large",
            dims: (2_048, 3_072, 768),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "large_deep",
            dims: (4_096, 3_072, 1_536),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "rect_wide",
            dims: (512, 3_072, 768),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "rect_tall",
            dims: (4_096, 512, 768),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "d768_in_proj",
            dims: (2_048, 768, 3_072),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "d768_out_proj",
            dims: (2_048, 1_536, 768),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
        Case {
            id: "prism_in_proj",
            dims: (4_621, 384, 1_928),
            alpha: 1.0,
            beta: 0.0,
            bias: false,
            cpu_oracle: false,
        },
    ];

    #[derive(Clone, Copy, Debug)]
    enum Arm {
        Production,
        Generic,
    }

    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Self::Production => "production_m64n64",
                Self::Generic => "production_generic",
            }
        }
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
        _device: GpuDevice,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        live_copyplan: Option<CudaFunction>,
    }

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
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
            let active_len = active.len();
            let active_offset = GUARD_ELEMENTS;
            let total = active_offset
                .checked_add(active_len)
                .and_then(|value| value.checked_add(GUARD_ELEMENTS))
                .ok_or_else(|| "guarded allocation size overflows usize".to_string())?;
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

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.buffer.to_cpu(stream)?;
            for (index, value) in actual[..self.active_offset]
                .iter()
                .chain(&actual[self.active_offset + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != self.guard_bits {
                    return Err(format!(
                        "{label} red zone changed at guard element {index}: 0x{:08x}",
                        value.to_bits()
                    ));
                }
            }
            Ok(
                actual[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.buffer.to_cpu(stream)?;
            if actual
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
        bias: Option<GuardedBuffer>,
        production_output: GuardedBuffer,
        generic_output: GuardedBuffer,
        oracle: Option<Vec<u32>>,
        params: KernelParams,
    }

    impl Fixture {
        fn bias_ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.bias.as_ref().map_or(0, |bias| bias.ptr(stream))
        }

        fn output(&self, arm: Arm) -> &GuardedBuffer {
            match arm {
                Arm::Production => &self.production_output,
                Arm::Generic => &self.generic_output,
            }
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedBuffer {
            match arm {
                Arm::Production => &mut self.production_output,
                Arm::Generic => &mut self.generic_output,
            }
        }

        fn validate_inputs(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.a.validate_unchanged(stream, "A")?;
            self.b.validate_unchanged(stream, "B")?;
            if let Some(bias) = &self.bias {
                bias.validate_unchanged(stream, "bias")?;
            }
            Ok(())
        }
    }

    fn compile_ptx(arch: &'static str) -> Result<String, String> {
        compile_ptx_source(arch, compose_cuda_source())
    }

    fn compile_ptx_source(arch: &'static str, source: String) -> Result<String, String> {
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
        cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile M64N64 qualification: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "M64N64 qualification requires CC 12.0 with 170 SMs, found CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let ptx = compile_ptx(device.nvrtc_target())?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load M64N64 qualification module: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            stream,
            module,
            live_copyplan: None,
        })
    }

    fn new_ada_runtime(copyplan: bool) -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
            return Err(format!(
                "Ada M64N64 discovery requires CC 8.9 with 142 SMs, found CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let ptx = if copyplan {
            // Reuse the production inference kernel without changing its arithmetic.
            // Its sole external helper is the identical 16-byte alignment predicate.
            let source = format!(
                "{}\n#define gbf_aligned16 gemm_bi_is_aligned_16\n{}\n#undef gbf_aligned16\n",
                compose_cuda_source(),
                include_str!("../../kernels/gemm_bi_inference/sm89_f32_n64_copyplan.cu"),
            );
            compile_ptx_source(device.nvrtc_target(), source)?
        } else {
            compile_ptx(device.nvrtc_target())?
        };
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load Ada M64N64 discovery module: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            stream,
            module,
            live_copyplan: None,
        })
    }

    fn seeded_values(len: usize, salt: u64, scale: f32) -> Vec<f32> {
        let mut state = salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 257 == 0 {
                    -0.0
                } else {
                    let signed = ((state >> 32) as u32 % 2049) as i32 - 1024;
                    signed as f32 * (scale / 1024.0)
                }
            })
            .collect()
    }

    fn nn_oracle(case: Case, a: &[f32], b: &[f32], bias: &[f32], seed: &[f32]) -> Vec<u32> {
        let (m, k, n) = case.dims;
        let mut output = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut accumulator = if case.bias { bias[column] } else { 0.0 };
                for reduction in 0..k {
                    accumulator =
                        a[row * k + reduction].mul_add(b[reduction * n + column], accumulator);
                }
                let mut value = case.alpha * accumulator;
                if case.beta != 0.0 {
                    value += case.beta * seed[row * n + column];
                }
                output.push(value.to_bits());
            }
        }
        output
    }

    fn new_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        let (m, k, n) = case.dims;
        let a_values = seeded_values(m * k, 0xa651_0001, 0.25);
        let b_values = seeded_values(k * n, 0xb651_0002, 0.125);
        let bias_values = seeded_values(n, 0xd651_0003, 0.03125);
        let output_seed = seeded_values(m * n, 0xc651_0004, 0.0625);
        let oracle = case
            .cpu_oracle
            .then(|| nn_oracle(case, &a_values, &b_values, &bias_values, &output_seed));
        let bias = case
            .bias
            .then(|| GuardedBuffer::new(&runtime.stream, bias_values, INPUT_GUARD_BITS))
            .transpose()?;
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.stream, a_values, INPUT_GUARD_BITS)?,
            b: GuardedBuffer::new(&runtime.stream, b_values, INPUT_GUARD_BITS)?,
            bias,
            production_output: GuardedBuffer::new(
                &runtime.stream,
                output_seed.clone(),
                OUTPUT_GUARD_BITS,
            )?,
            generic_output: GuardedBuffer::new(&runtime.stream, output_seed, OUTPUT_GUARD_BITS)?,
            oracle,
            params: KernelParams {
                alpha: case.alpha,
                beta: case.beta,
                m: i32::try_from(m).map_err(|_| "M exceeds i32")?,
                n: i32::try_from(n).map_err(|_| "N exceeds i32")?,
                k: i32::try_from(k).map_err(|_| "K exceeds i32")?,
                lda: i32::try_from(k).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(n).map_err(|_| "ldc exceeds i32")?,
            },
        })
    }

    fn new_ada_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        let (m, k, n) = case.dims;
        let a_values = fixed_full_mantissa::finite_full_mantissa_values(m * k, 0xa89a_0001);
        let b_values = fixed_full_mantissa::finite_full_mantissa_values(k * n, 0xb89a_0002);
        let bias_values = fixed_full_mantissa::finite_full_mantissa_values(n, 0xd89a_0003);
        let output_seed = fixed_full_mantissa::finite_full_mantissa_values(m * n, 0xc89a_0004);
        let oracle = case
            .cpu_oracle
            .then(|| nn_oracle(case, &a_values, &b_values, &bias_values, &output_seed));
        let bias = case
            .bias
            .then(|| GuardedBuffer::new(&runtime.stream, bias_values, INPUT_GUARD_BITS))
            .transpose()?;
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.stream, a_values, INPUT_GUARD_BITS)?,
            b: GuardedBuffer::new(&runtime.stream, b_values, INPUT_GUARD_BITS)?,
            bias,
            production_output: GuardedBuffer::new(
                &runtime.stream,
                output_seed.clone(),
                OUTPUT_GUARD_BITS,
            )?,
            generic_output: GuardedBuffer::new(&runtime.stream, output_seed, OUTPUT_GUARD_BITS)?,
            oracle,
            params: KernelParams {
                alpha: case.alpha,
                beta: case.beta,
                m: i32::try_from(m).map_err(|_| "M exceeds i32")?,
                n: i32::try_from(n).map_err(|_| "N exceeds i32")?,
                k: i32::try_from(k).map_err(|_| "K exceeds i32")?,
                lda: i32::try_from(k).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(n).map_err(|_| "ldc exceeds i32")?,
            },
        })
    }

    fn load_kernel(runtime: &Runtime, arm: Arm, case: Case) -> Result<Kernel, String> {
        let (symbol, tile_m, tile_n, threads, shared_bytes) = match arm {
            Arm::Production => (PRODUCTION_SYMBOL, 64, 64, 128, PRODUCTION_SHARED_BYTES),
            Arm::Generic => (GENERIC_SYMBOL, 128, 128, 256, GENERIC_SHARED_BYTES),
        };
        let function = runtime
            .module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared memory: {error:?}"))?;
        let (m, _, n) = case.dims;
        let blocks = m.div_ceil(tile_m) * n.div_ceil(tile_n);
        Ok(Kernel {
            function,
            config: LaunchConfig {
                grid_dim: (u32::try_from(blocks).map_err(|_| "grid exceeds u32")?, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: shared_bytes as u32,
            },
        })
    }

    fn load_ada_candidate(
        runtime: &Runtime,
        case: Case,
        symbol: &'static str,
    ) -> Result<Kernel, String> {
        let (_, dynamic_shared) = ada_candidate_shared(symbol)?;
        let function = if symbol == FIXED_COPYPLAN_SYMBOL && runtime.live_copyplan.is_some() {
            runtime.live_copyplan.as_ref().unwrap().clone()
        } else {
            runtime
                .module
                .load_function(symbol)
                .map_err(|error| format!("load {symbol}: {error:?}"))?
        };
        if dynamic_shared > 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared as i32,
                )
                .map_err(|error| format!("set {symbol} dynamic shared memory: {error:?}"))?;
        }
        let (m, _, n) = case.dims;
        let blocks = m.div_ceil(64) * n.div_ceil(64);
        Ok(Kernel {
            function,
            config: LaunchConfig {
                grid_dim: (u32::try_from(blocks).map_err(|_| "grid exceeds u32")?, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: dynamic_shared as u32,
            },
        })
    }

    fn launch(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        let output = fixture.output(arm).ptr(&runtime.stream);
        let a = if fixture.params.k == 0 {
            0
        } else {
            fixture.a.ptr(&runtime.stream)
        };
        let b = if fixture.params.k == 0 {
            0
        } else {
            fixture.b.ptr(&runtime.stream)
        };
        let bias = fixture.bias_ptr(&runtime.stream);
        let mut builder = runtime.stream.launch_builder(&kernel.function);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&bias);
        match arm {
            Arm::Production => {
                builder.arg(&fixture.params);
            }
            Arm::Generic => {
                builder.arg(&fixture.params.alpha);
                builder.arg(&fixture.params.beta);
                builder.arg(&fixture.params.m);
                builder.arg(&fixture.params.n);
                builder.arg(&fixture.params.k);
                builder.arg(&fixture.params.lda);
                builder.arg(&fixture.params.ldb);
                builder.arg(&fixture.params.ldc);
            }
        }
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", arm.name()))
    }

    fn capture(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch(runtime, kernel, fixture, arm)) }
    }

    fn resource_gate(function: &CudaFunction) -> Result<(), String> {
        let local = usize::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query production local bytes: {error:?}"))?,
        )
        .map_err(|_| "production reports negative local bytes".to_string())?;
        let registers = usize::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query production registers: {error:?}"))?,
        )
        .map_err(|_| "production reports negative registers".to_string())?;
        let static_shared = usize::try_from(
            function
                .shared_size_bytes()
                .map_err(|error| format!("query production static shared bytes: {error:?}"))?,
        )
        .map_err(|_| "production reports negative static shared bytes".to_string())?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(128, PRODUCTION_SHARED_BYTES, None)
            .map_err(|error| format!("query production occupancy: {error:?}"))?;
        if local != 0 {
            return Err(format!("production uses {local} local/spill bytes"));
        }
        if registers != EXPECTED_REGISTERS {
            return Err(format!(
                "production uses {registers} registers, expected {EXPECTED_REGISTERS}"
            ));
        }
        if static_shared != 0 {
            return Err(format!(
                "production uses {static_shared} static shared bytes, expected dynamic-only shared"
            ));
        }
        if occupancy < 4 {
            return Err(format!(
                "production occupancy {occupancy} is below four CTAs/SM"
            ));
        }
        eprintln!(
            "scalar_nn_m64n64 production registers={registers} local_bytes={local} dynamic_shared_bytes={PRODUCTION_SHARED_BYTES} active_blocks={occupancy}"
        );
        Ok(())
    }

    fn ada_resource_gate(function: &CudaFunction, symbol: &str) -> Result<(), String> {
        let (expected_static, dynamic_shared) = ada_candidate_shared(symbol)?;
        let expected_static = i32::try_from(expected_static)
            .map_err(|_| format!("{symbol} expected static shared exceeds i32"))?;
        let local = function
            .local_size_bytes()
            .map_err(|error| format!("query {symbol} local bytes: {error:?}"))?;
        let registers = function
            .num_regs()
            .map_err(|error| format!("query {symbol} registers: {error:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|error| format!("query {symbol} static shared bytes: {error:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("query {symbol} max threads: {error:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|error| format!("query {symbol} max dynamic shared: {error:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(128, dynamic_shared, None)
            .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
        println!(
            "{{\"schema\":\"MambaBiScalarNnM64AdaDiscoveryResourceV1\",\"symbol\":\"{symbol}\",\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{dynamic_shared},\"max_dynamic_shared_bytes\":{max_dynamic},\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":1}}"
        );
        if registers <= 0
            || local != 0
            || static_shared != expected_static
            || max_threads < 128
            || max_dynamic < dynamic_shared as i32
            || occupancy < 1
        {
            return Err(format!(
                "Ada {symbol} resource gate failed: registers={registers} local={local} static_shared={static_shared} max_threads={max_threads} max_dynamic={max_dynamic} occupancy={occupancy}"
            ));
        }
        Ok(())
    }

    fn check_case(runtime: &Runtime, case: Case) -> Result<(), String> {
        let production = load_kernel(runtime, Arm::Production, case)?;
        let generic = load_kernel(runtime, Arm::Generic, case)?;
        resource_gate(&production.function)?;
        let mut fixture = new_fixture(runtime, case)?;
        launch(runtime, &generic, &fixture, Arm::Generic)?;
        launch(runtime, &production, &fixture, Arm::Production)?;
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize {} eager: {error:?}", case.id))?;
        let production_bits = fixture
            .production_output
            .active_bits(&runtime.stream, "production output")?;
        let generic_bits = fixture
            .generic_output
            .active_bits(&runtime.stream, "generic output")?;
        if production_bits != generic_bits {
            return Err(format!("{} production bits differ from generic", case.id));
        }
        if fixture
            .oracle
            .as_ref()
            .is_some_and(|oracle| oracle != &production_bits)
        {
            return Err(format!(
                "{} production bits differ from CPU oracle",
                case.id
            ));
        }

        let production_graph = capture(runtime, &production, &fixture, Arm::Production)?;
        let generic_graph = capture(runtime, &generic, &fixture, Arm::Generic)?;
        for (arm, graph) in [
            (Arm::Production, &production_graph),
            (Arm::Generic, &generic_graph),
        ] {
            fixture.output_mut(arm).reset(&runtime.stream)?;
            graph
                .launch()
                .map_err(|error| format!("launch {} {} graph: {error:?}", case.id, arm.name()))?;
            runtime.stream.synchronize().map_err(|error| {
                format!("synchronize {} {} graph: {error:?}", case.id, arm.name())
            })?;
            let graph_bits = fixture
                .output(arm)
                .active_bits(&runtime.stream, arm.name())?;
            if graph_bits != production_bits {
                return Err(format!(
                    "{} {} graph bits differ from eager",
                    case.id,
                    arm.name()
                ));
            }
        }
        fixture.validate_inputs(&runtime.stream)?;
        drop(production_graph);
        drop(generic_graph);
        Ok(())
    }

    fn measure_window(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record timing start: {error:?}"))?;
        for _ in 0..iterations {
            launch(runtime, kernel, fixture, arm)?;
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record timing end: {error:?}"))?;
        let per_call_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure timing window: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !per_call_us.is_finite() || per_call_us <= 0.0 {
            return Err(format!("invalid timing sample {per_call_us}"));
        }
        Ok(per_call_us)
    }

    fn percentile(samples: &[f64], percentile: f64) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
        sorted[index]
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

    #[derive(Clone, Copy)]
    enum AdaOrder {
        Abba,
        Baab,
    }

    impl AdaOrder {
        const fn name(self) -> &'static str {
            match self {
                Self::Abba => "ABBA",
                Self::Baab => "BAAB",
            }
        }
    }

    fn reset_ada_observation(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        fixture.output_mut(arm).reset(&runtime.stream)?;
        fixture.a.reset(&runtime.stream)?;
        fixture.b.reset(&runtime.stream)?;
        if let Some(bias) = &mut fixture.bias {
            bias.reset(&runtime.stream)?;
        }
        Ok(())
    }

    fn launch_ada_path(
        runtime: &Runtime,
        kernel: &Kernel,
        graph: &CudaGraph,
        fixture: &Fixture,
        arm: Arm,
        path: AdaPath,
    ) -> Result<(), String> {
        match path {
            AdaPath::Eager => launch(runtime, kernel, fixture, arm),
            AdaPath::Graph => graph
                .launch()
                .map_err(|error| format!("launch {} graph: {error:?}", arm.name())),
        }
    }

    fn check_ada_pair(
        runtime: &Runtime,
        case: Case,
        candidate_symbol: &'static str,
    ) -> Result<(Kernel, Kernel, CudaGraph, CudaGraph, Fixture, Vec<u32>), String> {
        if candidate_symbol == FIXED_COPYPLAN_SYMBOL
            && !copyplan_epilogue_supported(case.alpha, case.beta, case.bias)
        {
            return Err("Fixed CopyPlan reuse is limited to alpha1/beta+0/no-bias; other Triad epilogues retain their prior route".into());
        }
        let candidate = load_ada_candidate(runtime, case, candidate_symbol)?;
        let generic = load_kernel(runtime, Arm::Generic, case)?;
        ada_resource_gate(&candidate.function, candidate_symbol)?;
        let mut fixture = new_ada_fixture(runtime, case)?;
        let candidate_graph = capture(runtime, &candidate, &fixture, Arm::Production)?;
        let generic_graph = capture(runtime, &generic, &fixture, Arm::Generic)?;
        let mut golden = None;
        for path in [AdaPath::Eager, AdaPath::Graph] {
            for repeat in 0..2 {
                reset_ada_observation(runtime, &mut fixture, Arm::Generic)?;
                launch_ada_path(
                    runtime,
                    &generic,
                    &generic_graph,
                    &fixture,
                    Arm::Generic,
                    path,
                )?;
                runtime.stream.synchronize().map_err(|error| {
                    format!("synchronize {} generic correctness: {error:?}", case.id)
                })?;
                let generic_bits = fixture
                    .generic_output
                    .active_bits(&runtime.stream, "Ada generic exact output")?;
                fixture.validate_inputs(&runtime.stream)?;
                if golden
                    .as_ref()
                    .is_some_and(|expected| expected != &generic_bits)
                {
                    return Err(format!(
                        "{} generic exact bits changed on {} repeat {repeat}",
                        case.id,
                        path.name()
                    ));
                }
                golden.get_or_insert(generic_bits);

                reset_ada_observation(runtime, &mut fixture, Arm::Production)?;
                launch_ada_path(
                    runtime,
                    &candidate,
                    &candidate_graph,
                    &fixture,
                    Arm::Production,
                    path,
                )?;
                runtime.stream.synchronize().map_err(|error| {
                    format!("synchronize {} candidate correctness: {error:?}", case.id)
                })?;
                let candidate_bits = fixture
                    .production_output
                    .active_bits(&runtime.stream, "Ada M64N64 candidate output")?;
                if golden.as_ref() != Some(&candidate_bits) {
                    let mismatch = golden
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(&candidate_bits)
                        .position(|(left, right)| left != right)
                        .unwrap_or(candidate_bits.len());
                    return Err(format!(
                        "{} {candidate_symbol} differs from generic exact at {mismatch} on {} repeat {repeat}: expected={:08x?} actual={:08x?}",
                        case.id,
                        path.name(),
                        golden.as_ref().unwrap().get(mismatch),
                        candidate_bits.get(mismatch)
                    ));
                }
                fixture.validate_inputs(&runtime.stream)?;
            }
        }
        Ok((
            candidate,
            generic,
            candidate_graph,
            generic_graph,
            fixture,
            golden.unwrap(),
        ))
    }

    fn measure_ada_observation(
        runtime: &Runtime,
        kernel: &Kernel,
        graph: &CudaGraph,
        fixture: &mut Fixture,
        arm: Arm,
        path: AdaPath,
        expected: &[u32],
    ) -> Result<f64, String> {
        reset_ada_observation(runtime, fixture, arm)?;
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record Ada {} start: {error:?}", arm.name()))?;
        launch_ada_path(runtime, kernel, graph, fixture, arm, path)?;
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record Ada {} end: {error:?}", arm.name()))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure Ada {}: {error:?}", arm.name()))?,
        ) * 1_000.0;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("invalid Ada {} timing sample {us}", arm.name()));
        }
        let actual = fixture
            .output(arm)
            .active_bits(&runtime.stream, arm.name())?;
        if actual != expected {
            return Err(format!(
                "Ada {} output changed after {} timing observation",
                arm.name(),
                path.name()
            ));
        }
        fixture.validate_inputs(&runtime.stream)?;
        Ok(us)
    }

    fn json_f64s(values: &[f64]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn screen_ada_stratum(
        runtime: &Runtime,
        cell: AdaShortScreenCell,
        candidate: &Kernel,
        generic: &Kernel,
        candidate_graph: &CudaGraph,
        generic_graph: &CudaGraph,
        fixture: &mut Fixture,
        expected: &[u32],
        path: AdaPath,
        order: AdaOrder,
    ) -> Result<[f64; 2], String> {
        for _ in 0..ADA_WARMUPS {
            for arm in [Arm::Production, Arm::Generic] {
                let (kernel, graph) = match arm {
                    Arm::Production => (candidate, candidate_graph),
                    Arm::Generic => (generic, generic_graph),
                };
                measure_ada_observation(runtime, kernel, graph, fixture, arm, path, expected)?;
            }
        }
        let arms = match order {
            AdaOrder::Abba => [Arm::Production, Arm::Generic, Arm::Generic, Arm::Production],
            AdaOrder::Baab => [Arm::Generic, Arm::Production, Arm::Production, Arm::Generic],
        };
        let mut candidate_samples = Vec::with_capacity(ADA_WINDOWS);
        let mut generic_samples = Vec::with_capacity(ADA_WINDOWS);
        let mut ratios = Vec::with_capacity(ADA_WINDOWS);
        let mut observations = Vec::with_capacity(ADA_WINDOWS);
        for _ in 0..ADA_WINDOWS {
            let mut raw = [0.0; 4];
            for (index, arm) in arms.into_iter().enumerate() {
                let (kernel, graph) = match arm {
                    Arm::Production => (candidate, candidate_graph),
                    Arm::Generic => (generic, generic_graph),
                };
                raw[index] =
                    measure_ada_observation(runtime, kernel, graph, fixture, arm, path, expected)?;
            }
            let (candidate_us, generic_us) = match order {
                AdaOrder::Abba => ((raw[0] + raw[3]) * 0.5, (raw[1] + raw[2]) * 0.5),
                AdaOrder::Baab => ((raw[1] + raw[2]) * 0.5, (raw[0] + raw[3]) * 0.5),
            };
            candidate_samples.push(candidate_us);
            generic_samples.push(generic_us);
            ratios.push(candidate_us / generic_us);
            observations.push(raw);
        }
        let p50 = percentile(&ratios, 0.50);
        let p95 = percentile(&ratios, 0.95);
        let observations = format!(
            "[{}]",
            observations
                .iter()
                .map(|raw| format!("[{:.9},{:.9},{:.9},{:.9}]", raw[0], raw[1], raw[2], raw[3]))
                .collect::<Vec<_>>()
                .join(",")
        );
        let (m, k, n) = cell.dims;
        println!(
            "{{\"schema\":\"MambaBiScalarNnM64AdaDiscoveryScreenV1\",\"op\":\"NN\",\"cell\":\"{}\",\"shape\":[{m},{k},{n}],\"candidate_symbol\":\"{}\",\"comparator\":\"generic_exact\",\"comparator_symbol\":\"{GENERIC_SYMBOL}\",\"path\":\"{}\",\"order\":\"{}\",\"windows\":{ADA_WINDOWS},\"warmups_per_arm\":{ADA_WARMUPS},\"logical_gemms_per_observation\":1,\"reseed_scope\":\"C+A+B\",\"reseed_position\":\"before_start_event\",\"post_download_before_next_reset\":true,\"observation_arms\":[\"{}\",\"{}\",\"{}\",\"{}\"],\"observations_us\":{observations},\"ratio_direction\":\"candidate_over_generic_exact\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9},\"candidate_samples_us\":{},\"generic_samples_us\":{},\"ratios\":{}}}",
            cell.id,
            cell.symbol,
            path.name(),
            order.name(),
            arms[0].name(),
            arms[1].name(),
            arms[2].name(),
            arms[3].name(),
            json_f64s(&candidate_samples),
            json_f64s(&generic_samples),
            json_f64s(&ratios),
        );
        Ok([p50, p95])
    }

    fn timed_pair(runtime: &Runtime, case: Case, comparator_arm: Arm) -> Result<(), String> {
        let production = load_kernel(runtime, Arm::Production, case)?;
        let comparator = load_kernel(runtime, comparator_arm, case)?;
        let fixture = new_fixture(runtime, case)?;
        for _ in 0..20 {
            launch(runtime, &production, &fixture, Arm::Production)?;
            launch(runtime, &comparator, &fixture, comparator_arm)?;
        }
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize {} warmup: {error:?}", case.id))?;
        let production_pilot = measure_window(runtime, &production, &fixture, Arm::Production, 5)?;
        let comparator_pilot = measure_window(runtime, &comparator, &fixture, comparator_arm, 5)?;
        let production_iterations =
            (15_000.0 / production_pilot).round().clamp(5.0, 500.0) as usize;
        let comparator_iterations =
            (15_000.0 / comparator_pilot).round().clamp(5.0, 500.0) as usize;

        for (order, production_first) in [("ABBA", true), ("BAAB", false)] {
            let mut production_samples = Vec::with_capacity(21);
            let mut comparator_samples = Vec::with_capacity(21);
            let mut ratios = Vec::with_capacity(21);
            for _ in 0..21 {
                let (
                    production_first_sample,
                    production_second_sample,
                    comparator_first_sample,
                    comparator_second_sample,
                ) = if production_first {
                    let production_first_sample = measure_window(
                        runtime,
                        &production,
                        &fixture,
                        Arm::Production,
                        production_iterations,
                    )?;
                    let comparator_first_sample = measure_window(
                        runtime,
                        &comparator,
                        &fixture,
                        comparator_arm,
                        comparator_iterations,
                    )?;
                    let comparator_second_sample = measure_window(
                        runtime,
                        &comparator,
                        &fixture,
                        comparator_arm,
                        comparator_iterations,
                    )?;
                    let production_second_sample = measure_window(
                        runtime,
                        &production,
                        &fixture,
                        Arm::Production,
                        production_iterations,
                    )?;
                    (
                        production_first_sample,
                        production_second_sample,
                        comparator_first_sample,
                        comparator_second_sample,
                    )
                } else {
                    let comparator_first_sample = measure_window(
                        runtime,
                        &comparator,
                        &fixture,
                        comparator_arm,
                        comparator_iterations,
                    )?;
                    let production_first_sample = measure_window(
                        runtime,
                        &production,
                        &fixture,
                        Arm::Production,
                        production_iterations,
                    )?;
                    let production_second_sample = measure_window(
                        runtime,
                        &production,
                        &fixture,
                        Arm::Production,
                        production_iterations,
                    )?;
                    let comparator_second_sample = measure_window(
                        runtime,
                        &comparator,
                        &fixture,
                        comparator_arm,
                        comparator_iterations,
                    )?;
                    (
                        production_first_sample,
                        production_second_sample,
                        comparator_first_sample,
                        comparator_second_sample,
                    )
                };
                let production_us = 0.5 * (production_first_sample + production_second_sample);
                let comparator_us = 0.5 * (comparator_first_sample + comparator_second_sample);
                production_samples.push(production_us);
                comparator_samples.push(comparator_us);
                ratios.push(production_us / comparator_us);
            }
            eprintln!(
                concat!(
                    "scalar_nn_m64n64 case={} comparator={} order={} production_us_p50={:.6} ",
                    "comparator_us_p50={:.6} production_over_comparator_p05={:.9} ",
                    "production_over_comparator_p50={:.9} production_over_comparator_p95={:.9} ",
                    "production_iterations={} comparator_iterations={}"
                ),
                case.id,
                comparator_arm.name(),
                order,
                percentile(&production_samples, 0.50),
                percentile(&comparator_samples, 0.50),
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
                production_iterations,
                comparator_iterations,
            );
        }
        Ok(())
    }

    fn timed_case(runtime: &Runtime, case: Case) -> Result<(), String> {
        timed_pair(runtime, case, Arm::Generic)
    }

    #[test]
    #[ignore = "requires an exclusive CC 12.0 170-SM GPU"]
    fn production_resource_and_exactness_qualification() -> Result<(), String> {
        let runtime = new_runtime()?;
        for case in CASES {
            check_case(&runtime, case)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC 12.0 170-SM GPU"]
    fn production_vs_generic_abba_baab() -> Result<(), String> {
        let runtime = new_runtime()?;
        for &case in &CASES[1..] {
            timed_case(&runtime, case)?;
        }
        Ok(())
    }

    fn run_ada_discovery(cells: &[AdaShortScreenCell], cohort: &str) -> Result<(), String> {
        if cells.is_empty() {
            return Err("Ada NN discovery cell filter is empty".into());
        }
        assert!(!cfg!(debug_assertions), "Ada discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let pre = quiet.require_pre_context(&format!("{cohort}/pre-context"))?;
        let copyplan = cells
            .iter()
            .any(|cell| cell.symbol == FIXED_COPYPLAN_SYMBOL);
        let runtime = new_ada_runtime(copyplan)?;

        for case in [
            Case {
                id: if copyplan {
                    "ada_copyplan_tail_overwrite"
                } else {
                    "ada_tail_alpha_beta"
                },
                dims: (67, 19, 137),
                alpha: if copyplan { 1.0 } else { -0.75 },
                beta: if copyplan { 0.0 } else { 0.5 },
                bias: false,
                cpu_oracle: false,
            },
            Case {
                id: "ada_k0",
                dims: (65, 0, 67),
                alpha: 1.0,
                beta: 0.0,
                bias: false,
                cpu_oracle: false,
            },
        ] {
            for symbol in cells
                .iter()
                .map(|cell| cell.symbol)
                .collect::<std::collections::BTreeSet<_>>()
            {
                let _ = check_ada_pair(&runtime, case, symbol)?;
            }
        }

        let timed_pre = quiet.require_cohort(&format!("{cohort}/timed"))?;
        let mut decisions = Vec::new();
        for &cell in cells {
            let case = Case {
                id: cell.id,
                dims: cell.dims,
                alpha: 1.0,
                beta: 0.0,
                bias: false,
                cpu_oracle: false,
            };
            let (candidate, generic, candidate_graph, generic_graph, mut fixture, golden) =
                check_ada_pair(&runtime, case, cell.symbol)?;
            let mut strata = Vec::new();
            for path in [AdaPath::Eager, AdaPath::Graph] {
                for order in [AdaOrder::Abba, AdaOrder::Baab] {
                    strata.push(screen_ada_stratum(
                        &runtime,
                        cell,
                        &candidate,
                        &generic,
                        &candidate_graph,
                        &generic_graph,
                        &mut fixture,
                        &golden,
                        path,
                        order,
                    )?);
                }
            }
            let retain = ada_short_screen_retains(&strata);
            let strata_json = format!(
                "[{}]",
                strata
                    .iter()
                    .map(|[p50, p95]| format!("[{p50:.9},{p95:.9}]"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let (m, k, n) = cell.dims;
            println!(
                "{{\"schema\":\"MambaBiScalarNnM64AdaDiscoveryDecisionV1\",\"op\":\"NN\",\"cell\":\"{}\",\"shape\":[{m},{k},{n}],\"candidate_symbol\":\"{}\",\"comparator\":\"generic_exact\",\"comparator_symbol\":\"{GENERIC_SYMBOL}\",\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"strata_fields\":[\"ratio_p50\",\"ratio_p95\"],\"strata\":{strata_json},\"ratio_direction\":\"candidate_over_generic_exact\",\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
                cell.id,
                cell.symbol,
                if retain {
                    "advance_to_full_qualification"
                } else {
                    "stop_no_retry"
                },
            );
            decisions.push(retain);
        }
        let post = quiet.verify_post_cohort(&format!("{cohort}/post"))?;
        let retain = decisions.iter().all(|value| *value);
        let cells_json = format!(
            "[{}]",
            cells
                .iter()
                .map(|cell| format!("\"{}\"", cell.id))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "{{\"schema\":\"MambaBiScalarNnM64AdaDiscoveryBatchV1\",\"cells\":{cells_json},\"pre\":{pre:?},\"timed_pre\":{timed_pre:?},\"post\":{post:?},\"all_cells_retain\":{retain},\"promotion\":false}}"
        );
        Ok(())
    }

    fn live_fast_launch(runtime: &Runtime, ctx: &GpuCtx, fixture: &Fixture) -> Result<(), String> {
        use cudarc::cublas::{result, sys as blas};
        use std::ffi::c_void;
        let p = fixture.params;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let dtype = mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::F32.cuda_data_type();
        // Row-major C=A*B is column-major C^T=B^T*A^T. Explicit FAST_TF32,
        // never PEDANTIC. Comparative timing starts after the exclusive
        // qualification holder has been dropped.
        unsafe {
            result::gemm_ex(
                *ctx.blas.handle(),
                blas::cublasOperation_t::CUBLAS_OP_N,
                blas::cublasOperation_t::CUBLAS_OP_N,
                p.n,
                p.m,
                p.k,
                &alpha as *const f32 as *const c_void,
                fixture.b.ptr(&runtime.stream) as *const c_void,
                dtype,
                p.ldb,
                fixture.a.ptr(&runtime.stream) as *const c_void,
                dtype,
                p.lda,
                &beta as *const f32 as *const c_void,
                fixture.generic_output.ptr(&runtime.stream) as *mut c_void,
                dtype,
                p.ldc,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|error| format!("explicit FAST_TF32 NN: {error:?}"))
    }

    // Public F32 entry points take owning buffers, not interior pointer views.
    // Use active origin zero and trailing guards, matching the qualifier's
    // allocation model. The direct arms keep their existing two-sided guards.
    struct PublicAutoFixture {
        output: GuardedBuffer,
        a: GuardedBuffer,
        b: GuardedBuffer,
        case: Case,
    }

    impl PublicAutoFixture {
        fn new(ctx: &GpuCtx, case: Case, c: &[u32], a: &[u32], b: &[u32]) -> Result<Self, String> {
            let make = |words: &[u32], guard_bits| {
                let mut expected: Vec<f32> = words.iter().copied().map(f32::from_bits).collect();
                expected.resize(words.len() + GUARD_ELEMENTS, f32::from_bits(guard_bits));
                Ok::<_, String>(GuardedBuffer {
                    buffer: GpuBuffer::from_cpu(&ctx.stream, &expected)?,
                    expected,
                    active_offset: 0,
                    active_len: words.len(),
                    guard_bits,
                })
            };
            Ok(Self {
                output: make(c, OUTPUT_GUARD_BITS)?,
                a: make(a, INPUT_GUARD_BITS)?,
                b: make(b, INPUT_GUARD_BITS)?,
                case,
            })
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.reset(&ctx.stream)?;
            self.a.reset(&ctx.stream)?;
            self.b.reset(&ctx.stream)
        }

        fn launch(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_forward_raw(
                ctx,
                &mut self.output.buffer,
                &self.a.buffer,
                self.b.ptr(&ctx.stream),
                None,
                self.case.dims,
            )
        }

        fn validate(&self, ctx: &GpuCtx, golden: &[u32]) -> Result<(), String> {
            if self.output.active_bits(&ctx.stream, "public AUTO")? != golden {
                return Err("public AUTO output differs from the exact reference".into());
            }
            self.a.validate_unchanged(&ctx.stream, "public AUTO A")?;
            self.b.validate_unchanged(&ctx.stream, "public AUTO B")
        }

        fn measure(
            &mut self,
            ctx: &GpuCtx,
            graph: &CudaGraph,
            path: AdaPath,
            golden: &[u32],
        ) -> Result<f64, String> {
            self.reset(ctx)?;
            let start = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| format!("public AUTO start: {e:?}"))?;
            match path {
                AdaPath::Eager => self.launch(ctx)?,
                AdaPath::Graph => graph
                    .launch()
                    .map_err(|e| format!("public AUTO graph: {e:?}"))?,
            }
            let end = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| format!("public AUTO end: {e:?}"))?;
            let us = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|e| format!("public AUTO elapsed: {e:?}"))?,
            ) * 1_000.0;
            self.validate(ctx, golden)?;
            if !us.is_finite() || us <= 0.0 {
                return Err("invalid public AUTO timing".into());
            }
            Ok(us)
        }
    }

    fn validate_public_auto_graph(
        ctx: &GpuCtx,
        graph: &CudaGraph,
        fixture: &PublicAutoFixture,
        stage: LiveCopyPlanStage,
        expected: mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedKernelLaunch,
    ) -> Result<(), String> {
        use std::ffi::CStr;
        let mut count = 0;
        let mut node = std::ptr::null_mut();
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe {
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count != 1
            {
                return Err(format!("public AUTO graph node count {count}"));
            }
            if sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || sys::cuGraphKernelNodeGetParams_v2(node, &mut params)
                    != sys::CUresult::CUDA_SUCCESS
            {
                return Err("public AUTO graph node query failed".into());
            }
            let mut name = std::ptr::null();
            if sys::cuFuncGetName(&mut name, params.func) != sys::CUresult::CUDA_SUCCESS
                || name.is_null()
            {
                return Err("public AUTO graph symbol query failed".into());
            }
            if CStr::from_ptr(name).to_bytes() != stage.expected_auto_symbol().as_bytes()
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != expected.grid_dim
                || (params.blockDimX, params.blockDimY, params.blockDimZ) != expected.block_dim
                || params.sharedMemBytes != expected.shared_mem_bytes
                || params.kernelParams.is_null()
            {
                return Err(
                    "public AUTO graph differs from the independently qualified plan".into(),
                );
            }
            for (index, pointer) in [
                fixture.output.ptr(&ctx.stream),
                fixture.a.ptr(&ctx.stream),
                fixture.b.ptr(&ctx.stream),
                0,
            ]
            .into_iter()
            .enumerate()
            {
                let mut offset = 0;
                let mut size = 0;
                if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                    != sys::CUresult::CUDA_SUCCESS
                    || offset != index * 8
                    || size != 8
                {
                    return Err(format!("public AUTO pointer ABI {index} changed"));
                }
                let value = *params.kernelParams.add(index);
                if value.is_null() || std::ptr::read_unaligned(value.cast::<u64>()) != pointer {
                    return Err(format!("public AUTO pointer binding {index} changed"));
                }
            }
            let (m, k, n) = fixture.case.dims;
            let words = [
                1.0_f32.to_bits(),
                0,
                m as u32,
                n as u32,
                k as u32,
                k as u32,
                n as u32,
                n as u32,
            ];
            let bundled = stage == LiveCopyPlanStage::Integrated;
            for (word_index, expected_word) in words.into_iter().enumerate() {
                let index = if bundled { 4 } else { 4 + word_index };
                let mut offset = 0;
                let mut size = 0;
                if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                    != sys::CUresult::CUDA_SUCCESS
                    || (offset, size)
                        != if bundled {
                            (32, 32)
                        } else {
                            (32 + word_index * 4, 4)
                        }
                {
                    return Err(format!("public AUTO scalar ABI {index} changed"));
                }
                let value = *params.kernelParams.add(index);
                if value.is_null()
                    || std::ptr::read_unaligned(value.cast::<u32>().add(if bundled {
                        word_index
                    } else {
                        0
                    })) != expected_word
                {
                    return Err(format!("public AUTO scalar binding {word_index} changed"));
                }
            }
            let mut offset = 0;
            let mut size = 0;
            if sys::cuFuncGetParamInfo(
                params.func,
                if bundled { 5 } else { 12 },
                &mut offset,
                &mut size,
            ) != sys::CUresult::CUDA_ERROR_INVALID_VALUE
            {
                return Err("public AUTO graph accepted an unexpected terminal argument".into());
            }
        }
        Ok(())
    }

    fn live_fast_observation(
        runtime: &Runtime,
        ctx: &GpuCtx,
        fixture: &mut Fixture,
        graph: &CudaGraph,
        path: AdaPath,
        expected: &[u32],
    ) -> Result<f64, String> {
        reset_ada_observation(runtime, fixture, Arm::Generic)?;
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("Fast start: {error:?}"))?;
        match path {
            AdaPath::Eager => live_fast_launch(runtime, ctx, fixture)?,
            AdaPath::Graph => graph
                .launch()
                .map_err(|error| format!("Fast graph: {error:?}"))?,
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("Fast end: {error:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("Fast elapsed: {error:?}"))?,
        ) * 1_000.0;
        if !us.is_finite()
            || us <= 0.0
            || fixture
                .generic_output
                .active_bits(&runtime.stream, "Fast output")?
                != expected
        {
            return Err("Fast timing/output changed within the frozen comparator".into());
        }
        fixture.validate_inputs(&runtime.stream)?;
        Ok(us)
    }

    #[test]
    #[ignore = "requires quiet Ada and one of CUDA12.8/13.0/13.2; live Fixed/AUTO/Fast comparison"]
    fn ada_live_fixed_copyplan_vs_auto_and_fast_three_cell_once7() -> Result<(), String> {
        run_live_copyplan_comparison(LiveCopyPlanStage::Discovery, &ada_copyplan_screen_cells())
    }

    #[test]
    #[ignore = "requires quiet Ada and one of CUDA12.8/13.0/13.2; live Fixed/AUTO/Fast comparison on the deep cell"]
    fn ada_live_fixed_copyplan_vs_auto_and_fast_large_deep_once7() -> Result<(), String> {
        run_live_copyplan_comparison(LiveCopyPlanStage::Discovery, &ada_copyplan_deep_cells())
    }

    #[test]
    #[ignore = "requires integrated Ada CopyPlan AUTO on CUDA12.8/13.0/13.2; actual AUTO/generic/Fast once21"]
    fn ada_integrated_copyplan_auto_three_cell_once21() -> Result<(), String> {
        run_live_copyplan_comparison(LiveCopyPlanStage::Integrated, &ada_copyplan_screen_cells())
    }

    #[test]
    #[ignore = "requires integrated Ada CopyPlan AUTO on CUDA12.8/13.0/13.2; actual AUTO/generic/Fast once21 on the deep cell"]
    fn ada_integrated_copyplan_auto_large_deep_once21() -> Result<(), String> {
        run_live_copyplan_comparison(LiveCopyPlanStage::Integrated, &ada_copyplan_deep_cells())
    }

    fn run_live_copyplan_comparison(
        stage: LiveCopyPlanStage,
        cells: &[AdaShortScreenCell],
    ) -> Result<(), String> {
        use cudarc::cublas::sys as blas;
        use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
            PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
        };
        use mamba_rs::mamba_ssm::gpu::kernel_identity::{
            ModuleKind, ResolvedGemmOp, ResolvedNumericContract,
        };
        use serde_json::json;

        let integrated = stage == LiveCopyPlanStage::Integrated;
        let windows = stage.windows();
        let mut integrated_generic_pass = true;
        let screen_schema = if integrated {
            "MambaBiIntegratedCopyPlanScreenV1"
        } else {
            "MambaBiLiveCopyPlanScreenV1"
        };
        let decision_schema = if integrated {
            "MambaBiIntegratedCopyPlanDecisionV1"
        } else {
            "MambaBiLiveCopyPlanDecisionV1"
        };
        assert!(!cfg!(debug_assertions), "live comparison requires release");
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("FAST_TF32 is disabled by NVIDIA_TF32_OVERRIDE".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("live-copyplan-auto-fast/pre")?;
        let mut runtime = new_ada_runtime(false)?;
        let ctx = GpuCtx::new(&runtime._device)?;
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_bi_tensor_cores(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        runtime.stream = ctx.stream.clone();
        runtime.live_copyplan = Some(
            ctx.kernels
                .fixed_sm89_f32_n64_copyplan
                .as_ref()
                .ok_or_else(|| {
                    format!(
                        "live Fixed CopyPlan unavailable: {:?}",
                        ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                    )
                })?
                .clone(),
        );
        let compiler = ctx.kernels.compiler_identity();
        let artifact = ctx.kernels.artifact_set_identity().fixed;
        let identity = runtime._device.identity();
        if !ada_live_toolkit_supported(
            identity.compute_capability,
            identity.multiprocessor_count,
            compiler.nvrtc_version,
        ) || compiler.target.as_str() != "sm_89"
            || !compiler.nvrtc_library_known
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.nvrtc_library_domain == [0; 32]
            || artifact.module_kind != ModuleKind::Fixed
            || artifact.artifact_digest == [0; 32]
            || artifact.compile_key != compiler.invocation_digest
            || artifact.artifact_kind != compiler.output_kind
        {
            return Err(format!(
                "live Fixed artifact mismatch: {compiler:?} {artifact:?}"
            ));
        }
        // Deterministic mode leaves the context handle in pedantic math for
        // the product's own protection; the deterministic route never crosses
        // cuBLAS, so the handle mode is free for the vendor comparator, which
        // is a Fast arm and must run in default math.
        let mut math = blas::cublasMath_t::CUBLAS_DEFAULT_MATH;
        let mut pointer = blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
        unsafe {
            if blas::cublasSetMathMode(*ctx.blas.handle(), math)
                != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || blas::cublasGetMathMode(*ctx.blas.handle(), &mut math)
                    != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || blas::cublasGetPointerMode_v2(*ctx.blas.handle(), &mut pointer)
                    != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            {
                return Err("set/query live Fast handle modes failed".into());
            }
        }
        if math != blas::cublasMath_t::CUBLAS_DEFAULT_MATH
            || pointer != blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST
        {
            return Err(format!(
                "unsupported Fast handle modes: {math:?} {pointer:?}"
            ));
        }
        println!(
            "{}",
            json!({"schema":"MambaBiLiveCopyPlanBindingV1", "symbol":FIXED_COPYPLAN_SYMBOL,
            "module":"Fixed", "stage":format!("{stage:?}"), "compiler":format!("{compiler:?}"), "artifact":format!("{artifact:?}"),
            "fast_compute":"CUBLAS_COMPUTE_32F_FAST_TF32", "fast_math":format!("{math:?}"), "fast_algorithm":"CUBLAS_GEMM_DEFAULT"})
        );
        // A different compiled module is a new bit domain: repeat bounded tail
        // and null-K0 checks before any target timing, using the actual holder.
        for dims in [(67, 19, 137), (65, 0, 67)] {
            check_ada_pair(
                &runtime,
                Case {
                    id: "live_copyplan_preflight",
                    dims,
                    alpha: 1.0,
                    beta: 0.0,
                    bias: false,
                    cpu_oracle: false,
                },
                FIXED_COPYPLAN_SYMBOL,
            )?;
        }
        quiet.require_cohort("live-copyplan-auto-fast/timed")?;
        for cell in cells.iter().copied() {
            let case = Case {
                id: cell.id,
                dims: cell.dims,
                alpha: 1.0,
                beta: 0.0,
                bias: false,
                cpu_oracle: false,
            };
            let (candidate, generic, candidate_graph, generic_graph, mut fixture, golden) =
                check_ada_pair(&runtime, case, FIXED_COPYPLAN_SYMBOL)?;
            let words = |buffer: &GuardedBuffer| {
                buffer.expected[buffer.active_offset..buffer.active_offset + buffer.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            let a_words = words(&fixture.a);
            let b_words = words(&fixture.b);
            let c_words = words(&fixture.production_output);
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                cell.dims,
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            );
            let mut auto = qualify_physical_launch(&ctx, request)?;
            let evidence = auto.evidence();
            if !evidence.eager_graph_equal()
                || evidence.launch_count() != 1
                || evidence.single_launch_symbol() != Some(stage.expected_auto_symbol())
                || evidence.uniform_module_kind()
                    != Some(if integrated {
                        ModuleKind::Fixed
                    } else {
                        ModuleKind::TriadScalar
                    })
                || evidence.launch_digest() == [0; 32]
            {
                return Err(format!("live NN AUTO identity changed: {evidence:?}"));
            }
            if integrated {
                let node = &evidence.nodes()[0];
                let expected_grid = (cell.dims.0.div_ceil(64) * cell.dims.2.div_ceil(64)) as u32;
                if node.shape != cell.dims
                    || node.tile != Some((64, 64))
                    || node.numeric_contract != Some(ResolvedNumericContract::ScalarFmaV1)
                    || node.launch.block_dim != (128, 1, 1)
                    || node.launch.grid_dim != (expected_grid, 1, 1)
                    || node.launch.shared_mem_bytes != 0
                    || node.launch.arguments_digest == [0; 32]
                {
                    return Err(format!(
                        "integrated CopyPlan AUTO physical node mismatch: {node:?}"
                    ));
                }
            }
            println!(
                "{}",
                json!({"schema":"MambaBiLiveCopyPlanAutoIdentityV1", "cell":cell.id,"stage":format!("{stage:?}"),
                "shape":cell.dims, "evidence":format!("{evidence:?}")})
            );
            let qualified_launch = evidence.nodes()[0].launch;
            // The holder owns an exclusive policy lease: only its own CUDA
            // operations may run until it is dropped. Check exact inputs here
            // in isolation, then independently capture the real public API.
            for path in [AdaPath::Eager, AdaPath::Graph] {
                for _ in 0..2 {
                    auto.upload_exact_unbiased_f32_words(&ctx, &c_words, &a_words, &b_words)?;
                    match path {
                        AdaPath::Eager => auto.measure_eager_window_ms(&ctx, 1)?,
                        AdaPath::Graph => auto.measure_graph_window_ms(&ctx, 1)?,
                    };
                    if auto.f32_output_bits(&ctx)? != golden
                        || auto.f32_operand_bits(&ctx)? != (a_words.clone(), b_words.clone())
                    {
                        return Err(
                            "isolated qualified AUTO differs from the exact reference".into()
                        );
                    }
                    auto.validate_red_zones(&ctx)?;
                }
            }
            drop(auto);
            let mut auto = PublicAutoFixture::new(&ctx, case, &c_words, &a_words, &b_words)?;
            auto.reset(&ctx)?;
            auto.launch(&ctx)?;
            auto.validate(&ctx, &golden)?;
            auto.reset(&ctx)?;
            let auto_graph = unsafe { capture_into_graph(&ctx.stream, || auto.launch(&ctx)) }?;
            validate_public_auto_graph(&ctx, &auto_graph, &auto, stage, qualified_launch)?;
            println!(
                "{}",
                json!({"schema":"MambaBiPublicAutoIsolationV1", "cell":cell.id,
                "qualified_holder_dropped":true,"public_graph_matches_qualified_launch":true,
                "public_entrypoint":"gpu_gemm_bi_forward_raw","public_buffer_origin":0,
                "public_guards":"trailing","direct_guards":"leading+trailing"})
            );
            reset_ada_observation(&runtime, &mut fixture, Arm::Generic)?;
            live_fast_launch(&runtime, &ctx, &fixture)?;
            runtime
                .stream
                .synchronize()
                .map_err(|error| format!("Fast warmup: {error:?}"))?;
            let fast_bits = fixture
                .generic_output
                .active_bits(&runtime.stream, "Fast initial")?;
            if fast_bits
                .iter()
                .any(|bits| !f32::from_bits(*bits).is_finite())
                || fast_bits.iter().all(|bits| *bits == 0)
            {
                return Err("Fast reference is nonfinite or identically zero".into());
            }
            let fast_graph = unsafe {
                capture_into_graph(&runtime.stream, || {
                    live_fast_launch(&runtime, &ctx, &fixture)
                })
            }?;
            for path in [AdaPath::Eager, AdaPath::Graph] {
                for _ in 0..2 {
                    live_fast_observation(
                        &runtime,
                        &ctx,
                        &mut fixture,
                        &fast_graph,
                        path,
                        &fast_bits,
                    )?;
                    auto.measure(&ctx, &auto_graph, path, &golden)?;
                }
            }
            for fast in [false, true] {
                let comparator = if fast {
                    "cublas_fast_tf32"
                } else if integrated {
                    "test_composed_generic_exact"
                } else {
                    "actual_auto"
                };
                let mut strata = Vec::new();
                for path in [AdaPath::Eager, AdaPath::Graph] {
                    for order in [AdaOrder::Abba, AdaOrder::Baab] {
                        let arms = match order {
                            AdaOrder::Abba => [true, false, false, true],
                            AdaOrder::Baab => [false, true, true, false],
                        };
                        let mut raw = Vec::new();
                        let mut ratios = Vec::new();
                        for bracket in 0..(ADA_WARMUPS + windows) {
                            let mut observation = [0.0; 4];
                            for (index, is_candidate) in arms.into_iter().enumerate() {
                                let measure_auto = if integrated {
                                    is_candidate
                                } else {
                                    !is_candidate && !fast
                                };
                                observation[index] = if measure_auto {
                                    auto.measure(&ctx, &auto_graph, path, &golden)?
                                } else if is_candidate {
                                    measure_ada_observation(
                                        &runtime,
                                        &candidate,
                                        &candidate_graph,
                                        &mut fixture,
                                        Arm::Production,
                                        path,
                                        &golden,
                                    )?
                                } else if fast {
                                    live_fast_observation(
                                        &runtime,
                                        &ctx,
                                        &mut fixture,
                                        &fast_graph,
                                        path,
                                        &fast_bits,
                                    )?
                                } else {
                                    measure_ada_observation(
                                        &runtime,
                                        &generic,
                                        &generic_graph,
                                        &mut fixture,
                                        Arm::Generic,
                                        path,
                                        &golden,
                                    )?
                                };
                            }
                            if observation.iter().any(|us| !us.is_finite() || *us <= 0.0) {
                                return Err("invalid live comparison timing".into());
                            }
                            if bracket >= ADA_WARMUPS {
                                let candidate_sum: f64 = observation
                                    .iter()
                                    .zip(arms)
                                    .filter(|(_, arm)| *arm)
                                    .map(|(us, _)| *us)
                                    .sum();
                                let reference_sum: f64 = observation
                                    .iter()
                                    .zip(arms)
                                    .filter(|(_, arm)| !*arm)
                                    .map(|(us, _)| *us)
                                    .sum();
                                ratios.push(candidate_sum / reference_sum);
                                raw.push(observation);
                            }
                        }
                        ratios.sort_by(f64::total_cmp);
                        let pair = [percentile(&ratios, 0.50), percentile(&ratios, 0.95)];
                        strata.push(pair);
                        println!(
                            "{}",
                            json!({"schema":screen_schema, "cell":cell.id,"shape":cell.dims,"stage":format!("{stage:?}"),
                            "candidate_path":if integrated { "public_auto_with_separate_qualified_preflight" } else { "direct_fixed_holder" },
                            "candidate_symbol":FIXED_COPYPLAN_SYMBOL,"candidate_module":"Fixed","comparator":comparator,
                            "path":path.name(),"order":order.name(),"candidate_first":arms[0],"windows":windows,
                            "warmup_brackets":ADA_WARMUPS,"logical_gemms_per_observation":1,"reseed_scope":"C+A+B",
                            "raw_observations_us":raw,"ratio_direction":"candidate_over_reference","ratio_p50":pair[0],"ratio_p95":pair[1]})
                        );
                    }
                }
                let retain = ada_short_screen_retains(&strata);
                if integrated && !fast {
                    integrated_generic_pass &= retain;
                }
                println!(
                    "{}",
                    json!({"schema":decision_schema,"cell":cell.id,"comparator":comparator,
                    "strata":strata,"retain":retain,"promotion":false,"actual_auto_integrated":integrated})
                );
            }
        }
        drop(ctx);
        drop(runtime);
        quiet.verify_post_cohort("live-copyplan-auto-fast/post")?;
        if !integrated_generic_pass {
            return Err(
                "integrated CopyPlan AUTO lost one or more generic-reference confirmation strata"
                    .into(),
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC 8.9 142-SM GPU; discovery only"]
    fn ada_exact_nn_m64n64_three_cell_discovery_once7() -> Result<(), String> {
        run_ada_discovery(&ada_short_screen_cells(), "scalar-nn-m64n64-ada")
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC 8.9 142-SM GPU; Prism repair only"]
    fn ada_exact_nn_m64n64_prism_repair_discovery_once7() -> Result<(), String> {
        let cells = ada_short_screen_cells();
        run_ada_discovery(&cells[2..], "scalar-nn-m64n64-ada-prism-repair")
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC 8.9 142-SM GPU; inference reuse discovery only"]
    fn ada_exact_nn_fixed_copyplan_overwrite_three_cell_discovery_once7() -> Result<(), String> {
        run_ada_discovery(&ada_copyplan_screen_cells(), "scalar-nn-fixed-copyplan-ada")
    }
}
