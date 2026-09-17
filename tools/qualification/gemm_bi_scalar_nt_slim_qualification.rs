const PRODUCTION_SYMBOL: &str = "nt_slim";
const SCALAR_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/scalar.cu");
const TEST_SOURCE: &str = include_str!("gemm_bi_scalar_nt_slim_qualification.rs");

fn production_body() -> &'static str {
    let (_, kernel_and_tail) = SCALAR_SOURCE
        .split_once("void nt_slim(")
        .expect("production NT Slim kernel must exist");
    kernel_and_tail
        .split_once("// Slim geometry ends here")
        .map(|(body, _)| body)
        .expect("production NT Slim kernel must precede the slim geometry cleanup")
}

#[test]
fn production_symbol_is_unique_registered_and_the_experiment_is_removed() {
    assert_eq!(
        SCALAR_SOURCE
            .matches(&format!("void {PRODUCTION_SYMBOL}("))
            .count(),
        1,
        "production NT Slim must have exactly one implementation"
    );
    assert!(!SCALAR_SOURCE.contains("GEMM_BI_SCALAR_NT_SLIM_BRAW_EXPERIMENT"));
    assert!(!SCALAR_SOURCE.contains("nt_slim_braw_exp"));
    assert!(
        include_str!("../../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs")
            .contains(PRODUCTION_SYMBOL)
    );
    assert!(
        include_str!("../../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs").contains(PRODUCTION_SYMBOL)
    );
}

#[test]
fn production_keeps_the_seven_argument_abi() {
    let (_, signature_tail) = SCALAR_SOURCE
        .split_once(&format!("void {PRODUCTION_SYMBOL}("))
        .expect("production NT Slim signature");
    let (parameters, _) = signature_tail
        .split_once(") {")
        .expect("production NT Slim parameter list");
    assert_eq!(
        parameters.split(',').count(),
        7,
        "production ABI must remain <=7 args"
    );
    let normalized = parameters.split_whitespace().collect::<String>();
    assert_eq!(
        normalized,
        concat!(
            "float*__restrict__C,",
            "constfloat*__restrict__A,",
            "constfloat*__restrict__B,",
            "floatalpha,intM,intN,intK_out"
        )
    );
}

#[test]
fn production_uses_the_qualified_braw_static_shared_layout() {
    let body = production_body();
    for required in [
        "constexpr int A_STAGE = 32 * 132;",
        "constexpr int B_RAW_STAGE = 64 * 32;",
        "constexpr int B_COMPUTE_STAGE = 32 * 68;",
        "static_assert(TOTAL_SMEM_BYTES == 33792",
        "__shared__ __align__(16) float smem[TOTAL_SMEM_FLOATS]",
        "float* Braw = As + A_STAGE;",
        "float* Bcompute = Braw + B_RAW_STAGE;",
        "reinterpret_cast<const uint4*>(Braw",
        "reinterpret_cast<unsigned int*>(Bcompute)",
        "cp.async.ca.shared.global [%0], [%1], 16;",
        "cp.async.ca.shared.global [%0], [%1], 4, %2;",
    ] {
        assert!(
            body.contains(required),
            "production NT Slim lost layout contract: {required}"
        );
    }
}

#[test]
fn production_keeps_one_ascending_dot_loop_and_one_ffma_update_site() {
    let body = production_body();
    assert_eq!(
        body.matches("for (int dotIdx = 0; dotIdx < GEMM_BI_SCALAR_BK; ++dotIdx)")
            .count(),
        1
    );
    assert_eq!(body.matches("threadResults[idx] = __fmaf_rn(").count(), 1);
    assert!(!body.contains("mma.sync"));
    assert!(!body.contains("atomic"));
    let ordered_tokens = [
        "for (int dotIdx = 0; dotIdx < GEMM_BI_SCALAR_BK; ++dotIdx)",
        "for (int wSubRowIdx = 0; wSubRowIdx < GEMM_BI_SCALAR_WMITER; ++wSubRowIdx)",
        "for (int wSubColIdx = 0; wSubColIdx < GEMM_BI_SCALAR_WNITER; ++wSubColIdx)",
        "for (int resIdxM = 0; resIdxM < GEMM_BI_SCALAR_TM; ++resIdxM)",
        "for (int resIdxN = 0; resIdxN < GEMM_BI_SCALAR_TN; ++resIdxN)",
        "threadResults[idx] = __fmaf_rn(",
        "regM[wSubRowIdx * GEMM_BI_SCALAR_TM + resIdxM]",
        "regN[wSubColIdx * GEMM_BI_SCALAR_TN + resIdxN]",
        "threadResults[idx]);",
    ];
    let mut tail = body;
    for token in ordered_tokens {
        let (_, next) = tail
            .split_once(token)
            .unwrap_or_else(|| panic!("production NT Slim changed FFMA order at: {token}"));
        tail = next;
    }
}

#[test]
fn qualification_source_freezes_runtime_correctness_and_resource_gates() {
    let unsupported_nvrtc_seed = ["--frandom", "-seed"].concat();
    assert!(
        !TEST_SOURCE.contains(&unsupported_nvrtc_seed),
        "NVRTC 12.8 qualification options must stay portable"
    );
    let runtime = TEST_SOURCE
        .rsplit_once("mod cuda_qualification {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA qualification module");
    for required in [
        "#[ignore = \"requires an exclusive SM80+ GPU\"]",
        "capture_into_graph",
        "local_size_bytes",
        "num_regs",
        "shared_size_bytes",
        "occupancy_max_active_blocks_per_multiprocessor",
        "occupancy_max_active_blocks_per_multiprocessor(128, 0, None)",
        "CPU oracle bits differ from production",
        "production graph bits differ from eager",
        "validate_red_zones",
        "EXPECTED_STATIC_SHARED_BYTES: usize = 33_792",
        "MAX_REGISTERS: usize = 254",
    ] {
        assert!(
            runtime.contains(required),
            "runtime gate is missing: {required}"
        );
    }
    for forbidden in [
        "measurement_only",
        "MAMBA_RS_SCALAR_NT_BRAW_",
        "Abba",
        "Baab",
        "candidate",
        "incumbent",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "production qualification retained tournament token: {forbidden}"
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
        .expect("production scalar NT Slim must compile for compute_80");
    let ptx = image.to_src();
    assert!(ptx.contains(PRODUCTION_SYMBOL));
    assert!(ptx.contains("fma.rn.f32"));
}

#[cfg(feature = "cuda")]
mod cuda_qualification {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

    use super::{PRODUCTION_SYMBOL, compose_cuda_source};

    const GUARD_ELEMENTS: usize = 32;
    const A_GUARD_BITS: u32 = 0x7fc0_a117;
    const B_GUARD_BITS: u32 = 0x7fc0_b117;
    const C_GUARD_BITS: u32 = 0x7fc0_c117;
    const CORRECTNESS_REPEATS: usize = 3;
    const EXPECTED_STATIC_SHARED_BYTES: usize = 33_792;
    const MAX_REGISTERS: usize = 254;

    #[derive(Clone, Copy, Debug)]
    struct Case {
        id: &'static str,
        dims: (usize, usize, usize),
        offsets: (usize, usize, usize),
        cpu_oracle: bool,
    }

    const CASES: [Case; 3] = [
        Case {
            id: "prism_aligned",
            dims: (4_621, 384, 1_928),
            offsets: (0, 0, 0),
            cpu_oracle: false,
        },
        Case {
            id: "all_dimensional_tails",
            dims: (131, 97, 95),
            offsets: (0, 0, 0),
            cpu_oracle: true,
        },
        Case {
            id: "unaligned_bases_and_tails",
            dims: (129, 67, 69),
            offsets: (1, 1, 1),
            cpu_oracle: true,
        },
    ];

    #[derive(Clone, Copy, Debug)]
    enum Path {
        Eager,
        Graph,
    }

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
        fn input(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            pointer_offset: usize,
            guard_bits: u32,
        ) -> Result<Self, String> {
            Self::new(stream, active, pointer_offset, guard_bits)
        }

        fn output(
            stream: &Arc<CudaStream>,
            active_len: usize,
            pointer_offset: usize,
        ) -> Result<Self, String> {
            Self::new(
                stream,
                vec![f32::from_bits(0x3f00_0000); active_len],
                pointer_offset,
                C_GUARD_BITS,
            )
        }

        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            pointer_offset: usize,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_len = active.len();
            let active_offset = GUARD_ELEMENTS
                .checked_add(pointer_offset)
                .ok_or_else(|| "guard offset overflows usize".to_string())?;
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

        fn validate_red_zones(
            &self,
            stream: &Arc<CudaStream>,
            label: &str,
        ) -> Result<Vec<f32>, String> {
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
            Ok(actual)
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.validate_red_zones(stream, label)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} read-only input changed"));
            }
            Ok(())
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.validate_red_zones(stream, label)?;
            Ok(
                actual[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        output: GuardedBuffer,
        oracle: Option<Vec<u32>>,
        dims_i32: (i32, i32, i32),
        alpha: f32,
    }

    fn compile_ptx(arch: &'static str, group_m: usize) -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                format!("-DGEMM_BI_GROUP_M={group_m}"),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile production scalar NT Slim: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability.0 < 8 {
            return Err(format!(
                "production scalar NT Slim qualification requires SM80+, found {:?}",
                device.compute_capability
            ));
        }
        let arch = device.nvrtc_target();
        let group_m = if device.compute_capability < (8, 9) {
            8
        } else {
            16
        };
        let ptx = compile_ptx(arch, group_m)?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load production scalar NT Slim: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            stream,
            module,
        })
    }

    fn seeded_values(len: usize, salt: u64) -> Vec<f32> {
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
                    signed as f32 / 4096.0
                }
            })
            .collect()
    }

    fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, String> {
        left.checked_mul(right)
            .ok_or_else(|| format!("{label} extent overflows usize"))
    }

    fn nt_oracle(case: Case, a: &[f32], b: &[f32]) -> Vec<u32> {
        let (m, k, n) = case.dims;
        let mut output = Vec::with_capacity(m * k);
        for row in 0..m {
            for column in 0..k {
                let mut accumulator = 0.0f32;
                for inner in 0..n {
                    accumulator = a[row * n + inner].mul_add(b[column * n + inner], accumulator);
                }
                output.push(accumulator.to_bits());
            }
        }
        output
    }

    fn new_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        let (m, k, n) = case.dims;
        let a_len = checked_product(m, n, "A")?;
        let b_len = checked_product(k, n, "B")?;
        let output_len = checked_product(m, k, "output")?;
        let a_values = seeded_values(a_len, 0xa117_0001);
        let b_values = seeded_values(b_len, 0xb117_0002);
        let oracle = case
            .cpu_oracle
            .then(|| nt_oracle(case, &a_values, &b_values));
        Ok(Fixture {
            a: GuardedBuffer::input(&runtime.stream, a_values, case.offsets.0, A_GUARD_BITS)?,
            b: GuardedBuffer::input(&runtime.stream, b_values, case.offsets.1, B_GUARD_BITS)?,
            output: GuardedBuffer::output(&runtime.stream, output_len, case.offsets.2)?,
            oracle,
            dims_i32: (
                i32::try_from(m).map_err(|_| "M exceeds i32")?,
                i32::try_from(n).map_err(|_| "N exceeds i32")?,
                i32::try_from(k).map_err(|_| "K exceeds i32")?,
            ),
            alpha: 1.0,
        })
    }

    fn load_kernel(runtime: &Runtime, case: Case) -> Result<Kernel, String> {
        let function = runtime
            .module
            .load_function(PRODUCTION_SYMBOL)
            .map_err(|error| format!("load {PRODUCTION_SYMBOL}: {error:?}"))?;
        let (m, k, _) = case.dims;
        let blocks = m
            .div_ceil(128)
            .checked_mul(k.div_ceil(64))
            .ok_or_else(|| "NT Slim grid overflows usize".to_string())?;
        Ok(Kernel {
            function,
            config: LaunchConfig {
                grid_dim: (
                    u32::try_from(blocks).map_err(|_| "NT Slim grid exceeds u32")?,
                    1,
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    fn launch(runtime: &Runtime, kernel: &Kernel, fixture: &Fixture) -> Result<(), String> {
        let c = fixture.output.ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
        let (m, n, k) = fixture.dims_i32;
        let mut builder = runtime.stream.launch_builder(&kernel.function);
        builder.arg(&c);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&fixture.alpha);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&k);
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {PRODUCTION_SYMBOL}: {error:?}"))
    }

    fn capture(runtime: &Runtime, kernel: &Kernel, fixture: &Fixture) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch(runtime, kernel, fixture)) }
    }

    fn execute_and_read(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &mut Fixture,
        graph: &CudaGraph,
        path: Path,
    ) -> Result<Vec<u32>, String> {
        fixture.output.reset(&runtime.stream)?;
        match path {
            Path::Eager => launch(runtime, kernel, fixture)?,
            Path::Graph => graph
                .launch()
                .map_err(|error| format!("launch production graph: {error:?}"))?,
        }
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize production {path:?}: {error:?}"))?;
        fixture
            .output
            .active_bits(&runtime.stream, "production output")
    }

    fn correctness_gate(
        runtime: &Runtime,
        kernel: &Kernel,
        fixture: &mut Fixture,
        graph: &CudaGraph,
        case: Case,
    ) -> Result<(), String> {
        let mut repeated = None;
        for repeat in 0..CORRECTNESS_REPEATS {
            let eager = execute_and_read(runtime, kernel, fixture, graph, Path::Eager)?;
            if fixture
                .oracle
                .as_ref()
                .is_some_and(|oracle| oracle != &eager)
            {
                return Err(format!(
                    "{} CPU oracle bits differ from production",
                    case.id
                ));
            }
            let graph_bits = execute_and_read(runtime, kernel, fixture, graph, Path::Graph)?;
            if graph_bits != eager {
                return Err(format!(
                    "{} production graph bits differ from eager at repeat {repeat}",
                    case.id
                ));
            }
            if repeated.as_ref().is_some_and(|expected| expected != &eager) {
                return Err(format!("{} repeated production bits changed", case.id));
            }
            repeated.get_or_insert(eager);
            fixture.a.validate_unchanged(&runtime.stream, "A")?;
            fixture.b.validate_unchanged(&runtime.stream, "B")?;
        }
        Ok(())
    }

    fn resource_gate(function: &CudaFunction) -> Result<(), String> {
        let local = function
            .local_size_bytes()
            .map_err(|error| format!("query production local bytes: {error:?}"))?;
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
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .map_err(|error| format!("query production occupancy: {error:?}"))?;
        if local != 0 {
            return Err(format!("production uses {local} local/spill bytes"));
        }
        if static_shared != EXPECTED_STATIC_SHARED_BYTES {
            return Err(format!(
                "production uses {static_shared} static shared bytes, expected {EXPECTED_STATIC_SHARED_BYTES}"
            ));
        }
        if registers > MAX_REGISTERS {
            return Err(format!(
                "production uses {registers} registers, above {MAX_REGISTERS}"
            ));
        }
        if occupancy < 2 {
            return Err(format!(
                "production occupancy {occupancy} is below two CTAs/SM"
            ));
        }
        eprintln!(
            "scalar_nt_slim production registers={registers} local_bytes={local} active_blocks={occupancy} static_shared_bytes={static_shared}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive SM80+ GPU"]
    fn production_nt_slim_qualification() -> Result<(), String> {
        let runtime = new_runtime()?;
        for case in CASES {
            let kernel = load_kernel(&runtime, case)?;
            resource_gate(&kernel.function)?;
            let mut fixture = new_fixture(&runtime, case)?;
            let graph = capture(&runtime, &kernel, &fixture)?;
            correctness_gate(&runtime, &kernel, &mut fixture, &graph, case)?;
            runtime
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize before graph drop: {error:?}"))?;
            drop(graph);
        }
        Ok(())
    }
}
