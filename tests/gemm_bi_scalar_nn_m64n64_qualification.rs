const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";
#[cfg(feature = "cuda")]
const GENERIC_SYMBOL: &str = "gemm_bi_nn";
const PRODUCTION_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu");
const TEST_SOURCE: &str = include_str!("gemm_bi_scalar_nn_m64n64_qualification.rs");
#[cfg(feature = "cuda")]
const SCALAR_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar.cu");

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
    assert!(parameters.matches(',').count() + 1 <= 7);
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
    let registry = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
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
    let modules = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
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
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
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
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

    use super::{GENERIC_SYMBOL, PRODUCTION_SYMBOL, compose_cuda_source};

    const GUARD_ELEMENTS: usize = 32;
    const INPUT_GUARD_BITS: u32 = 0x7fc0_a651;
    const OUTPUT_GUARD_BITS: u32 = 0x7fc0_c651;
    const PRODUCTION_SHARED_BYTES: usize = 17_408;
    const GENERIC_SHARED_BYTES: usize = 34 * 1_024;
    const EXPECTED_REGISTERS: usize = 103;

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
        cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
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

    fn launch(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        let output = fixture.output(arm).ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
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
}
