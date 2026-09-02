#[cfg(feature = "cuda")]
mod common;

const PARTIAL_SYMBOL: &str = "gemm_bi_tn_splitm_partial";
const ALIGNED_PARTIAL_SYMBOL: &str = "gemm_bi_tn_splitm_partial_aligned";
const REDUCER_SYMBOL: &str = "gemm_bi_splitm_reduce";
const SCALAR_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar.cu");
const TEST_SOURCE: &str = include_str!("gemm_bi_tn_splitm_cp_async_qualification.rs");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Case {
    id: &'static str,
    dims: (usize, usize, usize),
    m_chunk: usize,
    chunks: usize,
    pointer_offset: usize,
    aligned: bool,
}

const CASES: [Case; 2] = [
    Case {
        id: "aligned_partition_and_dimensional_tails",
        dims: (4_111, 129, 257),
        m_chunk: 80,
        chunks: 52,
        pointer_offset: 0,
        aligned: true,
    },
    Case {
        id: "unaligned_bases_and_tails",
        dims: (131, 97, 95),
        m_chunk: 48,
        chunks: 3,
        pointer_offset: 1,
        aligned: false,
    },
];

fn splitm_impl_body() -> &'static str {
    let (_, body) = SCALAR_SOURCE
        .split_once("void gemm_bi_tn_splitm_partial_impl(")
        .expect("production TN split-M implementation");
    body.split_once("void gemm_bi_tn_splitm_partial(")
        .map(|(body, _)| body)
        .expect("production TN split-M exports follow the implementation")
}

fn production_parameters(symbol: &str) -> &'static str {
    let (_, signature) = SCALAR_SOURCE
        .split_once(&format!("void {symbol}("))
        .unwrap_or_else(|| panic!("missing production symbol {symbol}"));
    signature
        .split_once(") {")
        .map(|(parameters, _)| parameters)
        .unwrap_or_else(|| panic!("missing production parameter list for {symbol}"))
}

#[test]
fn production_symbols_are_unique_and_keep_the_seven_argument_abi() {
    let expected = concat!(
        "float*__restrict__partial,",
        "constfloat*__restrict__A,",
        "constfloat*__restrict__B,",
        "intM_red,intK_out,intN,",
        "intM_CHUNK"
    );
    for symbol in [PARTIAL_SYMBOL, ALIGNED_PARTIAL_SYMBOL] {
        assert_eq!(SCALAR_SOURCE.matches(&format!("void {symbol}(")).count(), 1);
        let parameters = production_parameters(symbol);
        assert_eq!(parameters.matches(',').count() + 1, 7, "{symbol} ABI");
        assert_eq!(parameters.split_whitespace().collect::<String>(), expected);
    }
    assert_eq!(
        SCALAR_SOURCE
            .matches(&format!("void {REDUCER_SYMBOL}("))
            .count(),
        1
    );
}

#[test]
fn production_uses_only_the_qualified_two_stage_body() {
    let body = splitm_impl_body();
    for required in [
        "constexpr int K_PIPE = 2",
        "static_assert(TOTAL_SMEM_BYTES == 33792",
        "ISSUE_TN_SPLITM_TILE(0, m_begin)",
        "cp.async.ca.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_group",
        "int next_tile = tile + 1",
        "int read_stage = 0",
        "int write_stage = 1",
        "BASES_ALIGNED || gemm_bi_is_aligned_16(A)",
        "BASES_ALIGNED || gemm_bi_is_aligned_16(B)",
        "!BASES_ALIGNED && !gemm_bi_is_aligned_16(destination)",
    ] {
        assert!(body.contains(required), "production body lost {required}");
    }
    for forbidden in ["cp.async.wait_all", "for (int mIdx", "atomic"] {
        assert!(!body.contains(forbidden), "production retained {forbidden}");
    }
    for specialization in [
        "gemm_bi_tn_splitm_partial_impl<false>(",
        "gemm_bi_tn_splitm_partial_impl<true>(",
    ] {
        assert!(
            SCALAR_SOURCE.contains(specialization),
            "production lost alignment specialization {specialization}"
        );
    }
}

#[test]
fn production_keeps_one_ascending_ffma_update_site() {
    let body = splitm_impl_body();
    assert_eq!(
        body.matches("for (int dotIdx = 0; dotIdx < GEMM_BI_SCALAR_BK; ++dotIdx)")
            .count(),
        1
    );
    assert_eq!(body.matches("threadResults[idx] = __fmaf_rn(").count(), 1);
    let ordered = [
        "for (int tile = 0; tile < num_tiles; ++tile)",
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
    for token in ordered {
        let (_, next) = tail
            .split_once(token)
            .unwrap_or_else(|| panic!("production changed exact order at {token}"));
        tail = next;
    }
}

#[test]
fn qualification_cases_cover_aligned_and_unaligned_tails() {
    assert_eq!(
        CASES,
        [
            Case {
                id: "aligned_partition_and_dimensional_tails",
                dims: (4_111, 129, 257),
                m_chunk: 80,
                chunks: 52,
                pointer_offset: 0,
                aligned: true,
            },
            Case {
                id: "unaligned_bases_and_tails",
                dims: (131, 97, 95),
                m_chunk: 48,
                chunks: 3,
                pointer_offset: 1,
                aligned: false,
            },
        ]
    );
    for case in CASES {
        assert_eq!(case.chunks, case.dims.0.div_ceil(case.m_chunk));
    }
}

#[test]
fn experimental_export_is_absent_from_production() {
    for forbidden in [
        "GEMM_BI_SCALAR_TN_SPLITM_CP_ASYNC_EXPERIMENT",
        "gemm_bi_tn_splitm_partial_cp_async_exp_v1",
    ] {
        assert!(!SCALAR_SOURCE.contains(forbidden));
        for source in [
            include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs"),
            include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs"),
            include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs"),
            include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs"),
        ] {
            assert!(!source.contains(forbidden));
        }
    }
}

#[test]
fn qualification_source_freezes_runtime_and_resource_gates() {
    let runtime = TEST_SOURCE
        .rsplit_once("mod cuda_qualification {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA qualification module");
    for required in [
        "#[ignore = \"requires an exclusive SM80+ GPU\"]",
        "capture_into_graph",
        "Path::Eager",
        "Path::Graph",
        "validate_red_zones",
        "validate_unchanged",
        "CPU output oracle bits differ from production",
        "CPU partial oracle bits differ from production",
        "production graph bits differ from eager",
        "local_size_bytes",
        "num_regs",
        "shared_size_bytes",
        "occupancy_max_active_blocks_per_multiprocessor(256, 0, None)",
        "EXPECTED_STATIC_SHARED_BYTES: usize = 33_792",
        "MAX_REGISTERS: usize = 128",
    ] {
        assert!(
            runtime.contains(required),
            "runtime gate is missing {required}"
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
        .expect("production exact-F32 TN split-M must compile for compute_80");
    let ptx = image.to_src();
    assert!(ptx.contains(PARTIAL_SYMBOL));
    assert!(ptx.contains(ALIGNED_PARTIAL_SYMBOL));
    assert!(ptx.contains(REDUCER_SYMBOL));
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

    use super::{
        ALIGNED_PARTIAL_SYMBOL, CASES, Case, PARTIAL_SYMBOL, REDUCER_SYMBOL, compose_cuda_source,
    };

    const GUARD_ELEMENTS: usize = 32;
    const A_GUARD_BITS: u32 = 0x7fc0_a317;
    const B_GUARD_BITS: u32 = 0x7fc0_b317;
    const OUTPUT_GUARD_BITS: u32 = 0x7fc0_c317;
    const SCRATCH_GUARD_BITS: u32 = 0x7fc0_d317;
    const CORRECTNESS_REPEATS: usize = 3;
    const EXPECTED_STATIC_SHARED_BYTES: usize = 33_792;
    const MAX_REGISTERS: usize = 128;

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

    struct Kernels {
        partial: CudaFunction,
        aligned_partial: CudaFunction,
        reducer: CudaFunction,
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        initial: Vec<f32>,
        active_offset: usize,
        active_len: usize,
        guard_bits: u32,
    }

    impl GuardedBuffer {
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
            let mut initial = vec![f32::from_bits(guard_bits); total];
            initial[active_offset..active_offset + active_len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &initial)?,
                initial,
                active_offset,
                active_len,
                guard_bits,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.active_offset)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.initial)
        }

        fn validate_red_zones(
            &self,
            stream: &Arc<CudaStream>,
            label: &str,
        ) -> Result<Vec<f32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            for (index, value) in values[..self.active_offset]
                .iter()
                .chain(&values[self.active_offset + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != self.guard_bits {
                    return Err(format!(
                        "{label} red zone changed at guard element {index}: 0x{:08x}",
                        value.to_bits()
                    ));
                }
            }
            Ok(values)
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let values = self.validate_red_zones(stream, label)?;
            Ok(
                values[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let values = self.validate_red_zones(stream, label)?;
            if values
                .iter()
                .zip(&self.initial)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} read-only input changed"));
            }
            Ok(())
        }
    }

    struct Oracle {
        output: Vec<u32>,
        partial: Vec<u32>,
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        output: GuardedBuffer,
        scratch: GuardedBuffer,
        oracle: Oracle,
        dims_i32: (i32, i32, i32),
        m_chunk_i32: i32,
        chunks_i32: i32,
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
            .map_err(|error| format!("compile production exact-F32 TN split-M: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability.0 < 8 {
            return Err(format!(
                "production exact-F32 TN split-M qualification requires SM80+, found {:?}",
                device.compute_capability
            ));
        }
        let group_m = if device.compute_capability < (8, 9) {
            8
        } else {
            16
        };
        let ptx = compile_ptx(device.nvrtc_target(), group_m)?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load production exact-F32 TN split-M: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            stream,
            module,
        })
    }

    fn load_kernels(runtime: &Runtime) -> Result<Kernels, String> {
        let load = |symbol| {
            runtime
                .module
                .load_function(symbol)
                .map_err(|error| format!("load {symbol}: {error:?}"))
        };
        Ok(Kernels {
            partial: load(PARTIAL_SYMBOL)?,
            aligned_partial: load(ALIGNED_PARTIAL_SYMBOL)?,
            reducer: load(REDUCER_SYMBOL)?,
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

    fn checked_product(values: &[usize], label: &str) -> Result<usize, String> {
        values.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or_else(|| format!("{label} extent overflows usize"))
        })
    }

    fn split_oracle(case: Case, a: &[f32], b: &[f32], seed: &[f32]) -> Oracle {
        let (m, k, n) = case.dims;
        let kn = k * n;
        let mut partial = vec![0.0f32; case.chunks * kn];
        for partition in 0..case.chunks {
            let begin = partition * case.m_chunk;
            let end = (begin + case.m_chunk).min(m);
            for row in 0..k {
                for column in 0..n {
                    let mut accumulator = 0.0f32;
                    for reduction in begin..end {
                        accumulator =
                            a[reduction * k + row].mul_add(b[reduction * n + column], accumulator);
                    }
                    partial[partition * kn + row * n + column] = accumulator;
                }
            }
        }
        let output = (0..kn)
            .map(|index| {
                let mut sum = f64::from(partial[index]);
                for partition in 1..case.chunks {
                    sum += f64::from(partial[partition * kn + index]);
                }
                (seed[index] + sum as f32).to_bits()
            })
            .collect();
        Oracle {
            output,
            partial: partial.iter().map(|value| value.to_bits()).collect(),
        }
    }

    fn new_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        let (m, k, n) = case.dims;
        let a_values = seeded_values(checked_product(&[m, k], "A")?, 0xa317_0001, 0.25);
        let b_values = seeded_values(checked_product(&[m, n], "B")?, 0xb317_0002, 0.125);
        let output_len = checked_product(&[k, n], "output")?;
        let scratch_len = checked_product(&[case.chunks, k, n], "scratch")?;
        let seed = seeded_values(output_len, 0xc317_0003, 0.03125);
        let oracle = split_oracle(case, &a_values, &b_values, &seed);
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.stream, a_values, case.pointer_offset, A_GUARD_BITS)?,
            b: GuardedBuffer::new(&runtime.stream, b_values, case.pointer_offset, B_GUARD_BITS)?,
            output: GuardedBuffer::new(
                &runtime.stream,
                seed,
                case.pointer_offset,
                OUTPUT_GUARD_BITS,
            )?,
            scratch: GuardedBuffer::new(
                &runtime.stream,
                vec![0.0; scratch_len],
                case.pointer_offset,
                SCRATCH_GUARD_BITS,
            )?,
            oracle,
            dims_i32: (
                i32::try_from(m).map_err(|_| "M exceeds i32")?,
                i32::try_from(k).map_err(|_| "K exceeds i32")?,
                i32::try_from(n).map_err(|_| "N exceeds i32")?,
            ),
            m_chunk_i32: i32::try_from(case.m_chunk).map_err(|_| "M chunk exceeds i32")?,
            chunks_i32: i32::try_from(case.chunks).map_err(|_| "chunks exceeds i32")?,
            alpha: 1.0,
        })
    }

    fn partial_function<'a>(kernels: &'a Kernels, case: Case) -> &'a CudaFunction {
        if case.aligned {
            &kernels.aligned_partial
        } else {
            &kernels.partial
        }
    }

    fn launch_route(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        case: Case,
    ) -> Result<(), String> {
        let (_, k, n) = case.dims;
        let grid_x = k
            .div_ceil(128)
            .checked_mul(n.div_ceil(128))
            .ok_or_else(|| "partial grid overflows usize".to_string())?;
        let partial_config = LaunchConfig {
            grid_dim: (
                u32::try_from(grid_x).map_err(|_| "partial grid exceeds u32")?,
                1,
                u32::try_from(case.chunks).map_err(|_| "chunks exceed u32")?,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let scratch = fixture.scratch.ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
        let (m_i32, k_i32, n_i32) = fixture.dims_i32;
        let mut partial = runtime
            .stream
            .launch_builder(partial_function(kernels, case));
        partial.arg(&scratch);
        partial.arg(&a);
        partial.arg(&b);
        partial.arg(&m_i32);
        partial.arg(&k_i32);
        partial.arg(&n_i32);
        partial.arg(&fixture.m_chunk_i32);
        unsafe { partial.launch(partial_config) }
            .map_err(|error| format!("launch {} partial: {error:?}", case.id))?;

        let total = u32::try_from(
            k.checked_mul(n)
                .ok_or_else(|| "reducer size overflows usize".to_string())?,
        )
        .map_err(|_| "reducer size exceeds u32")?;
        let output = fixture.output.ptr(&runtime.stream);
        let mut reducer = runtime.stream.launch_builder(&kernels.reducer);
        reducer.arg(&output);
        reducer.arg(&scratch);
        reducer.arg(&fixture.alpha);
        reducer.arg(&k_i32);
        reducer.arg(&n_i32);
        reducer.arg(&fixture.chunks_i32);
        unsafe {
            reducer.launch(LaunchConfig {
                grid_dim: (total.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| format!("launch {} reducer: {error:?}", case.id))
    }

    fn capture(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        case: Case,
    ) -> Result<CudaGraph, String> {
        unsafe {
            capture_into_graph(&runtime.stream, || {
                launch_route(runtime, kernels, fixture, case)
            })
        }
    }

    fn execute_and_read(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        graph: &CudaGraph,
        case: Case,
        path: Path,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        fixture.output.reset(&runtime.stream)?;
        fixture.scratch.reset(&runtime.stream)?;
        match path {
            Path::Eager => launch_route(runtime, kernels, fixture, case)?,
            Path::Graph => graph
                .launch()
                .map_err(|error| format!("launch production graph: {error:?}"))?,
        }
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize production {path:?}: {error:?}"))?;
        Ok((
            fixture
                .output
                .active_bits(&runtime.stream, "production output")?,
            fixture
                .scratch
                .active_bits(&runtime.stream, "production scratch")?,
        ))
    }

    fn correctness_gate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        graph: &CudaGraph,
        case: Case,
    ) -> Result<(), String> {
        let mut repeated = None;
        for repeat in 0..CORRECTNESS_REPEATS {
            let eager = execute_and_read(runtime, kernels, fixture, graph, case, Path::Eager)?;
            if eager.0 != fixture.oracle.output {
                return Err(format!(
                    "{} CPU output oracle bits differ from production",
                    case.id
                ));
            }
            if eager.1 != fixture.oracle.partial {
                return Err(format!(
                    "{} CPU partial oracle bits differ from production",
                    case.id
                ));
            }
            let graph_bits = execute_and_read(runtime, kernels, fixture, graph, case, Path::Graph)?;
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

    #[derive(Clone, Copy)]
    struct ResourceSnapshot {
        local: usize,
        registers: usize,
        shared: usize,
        occupancy: u32,
    }

    fn loaded_nvrtc_version() -> Result<(i32, i32), String> {
        let mut major = 0;
        let mut minor = 0;
        let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
        if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("query NVRTC version: {result:?}"));
        }
        Ok((major, minor))
    }

    fn resource_snapshot(
        symbol: &str,
        function: &CudaFunction,
    ) -> Result<ResourceSnapshot, String> {
        let local = usize::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query {symbol} local bytes: {error:?}"))?,
        )
        .map_err(|_| format!("{symbol} reports negative local bytes"))?;
        let registers = usize::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query {symbol} registers: {error:?}"))?,
        )
        .map_err(|_| format!("{symbol} reports negative registers"))?;
        let shared = usize::try_from(
            function
                .shared_size_bytes()
                .map_err(|error| format!("query {symbol} shared bytes: {error:?}"))?,
        )
        .map_err(|_| format!("{symbol} reports negative shared bytes"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
            .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
        let snapshot = ResourceSnapshot {
            local,
            registers,
            shared,
            occupancy,
        };
        eprintln!(
            "tn_splitm production symbol={symbol} registers={} local_bytes={} shared_bytes={} active_blocks={}",
            snapshot.registers, snapshot.local, snapshot.shared, snapshot.occupancy
        );
        Ok(snapshot)
    }

    fn resource_gate(
        symbol: &str,
        snapshot: ResourceSnapshot,
        nvrtc_version: (i32, i32),
    ) -> Result<(), String> {
        let max_local = match (nvrtc_version, symbol) {
            ((12, 8), PARTIAL_SYMBOL) => 16,
            _ => 0,
        };
        if snapshot.local > max_local {
            return Err(format!(
                "{symbol} uses {} local/spill bytes with NVRTC {}.{}, above the qualified {max_local}-byte ceiling",
                snapshot.local, nvrtc_version.0, nvrtc_version.1
            ));
        }
        if snapshot.registers > MAX_REGISTERS {
            return Err(format!(
                "{symbol} uses {} registers, above {MAX_REGISTERS}",
                snapshot.registers
            ));
        }
        if snapshot.shared != EXPECTED_STATIC_SHARED_BYTES {
            return Err(format!(
                "{symbol} uses {} shared bytes, expected {EXPECTED_STATIC_SHARED_BYTES}",
                snapshot.shared
            ));
        }
        if snapshot.occupancy < 2 {
            return Err(format!(
                "{symbol} occupancy {} is below two CTAs/SM",
                snapshot.occupancy
            ));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive SM80+ GPU"]
    fn production_tn_splitm_cp_async_qualification() -> Result<(), String> {
        let runtime = new_runtime()?;
        let kernels = load_kernels(&runtime)?;
        let nvrtc_version = loaded_nvrtc_version()?;
        let resources = [
            (
                PARTIAL_SYMBOL,
                resource_snapshot(PARTIAL_SYMBOL, &kernels.partial)?,
            ),
            (
                ALIGNED_PARTIAL_SYMBOL,
                resource_snapshot(ALIGNED_PARTIAL_SYMBOL, &kernels.aligned_partial)?,
            ),
        ];
        for (symbol, snapshot) in resources {
            resource_gate(symbol, snapshot, nvrtc_version)?;
        }
        for case in CASES {
            let mut fixture = new_fixture(&runtime, case)?;
            let graph = capture(&runtime, &kernels, &fixture, case)?;
            correctness_gate(&runtime, &kernels, &mut fixture, &graph, case)?;
            runtime
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize before graph drop: {error:?}"))?;
            drop(graph);
        }
        Ok(())
    }
}
