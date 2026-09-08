//! Test-only Ada TF32 NN N96 direct-epilogue no-barrier discovery.
//! Production kernels and dispatch remain untouched.

#[path = "support/triad_tf32_nn_n96_direct_epilogue_nobarrier_source.rs"]
mod candidate_source;
#[path = "support/triad_tf32_nn_n96_direct_epilogue_source.rs"]
#[allow(dead_code)]
mod direct_source;
#[path = "support/triad_nn_n96_source.rs"]
#[allow(dead_code)]
mod retained_source;

const FIXED_N96_SOURCE: &str = include_str!("../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

fn strict_once7_win(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .flatten()
            .all(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < 0.99)
}

#[test]
fn direct_float2_mapping_covers_full_n96_tile_once() {
    for (warp, m_atom, n_atom, half, lane, expected) in [
        (0, 0, 0, 0, 0, ((0, 0), (0, 1))),
        (0, 0, 0, 1, 31, ((15, 6), (15, 7))),
        (3, 2, 1, 0, 5, ((33, 82), (33, 83))),
        (7, 3, 2, 1, 30, ((127, 92), (127, 93))),
    ] {
        assert_eq!(
            direct_source::direct_pair_coordinates(warp, m_atom, n_atom, half, lane),
            expected,
        );
    }

    let mut owners = vec![0_u8; 128 * 96];
    for warp in 0..8 {
        for m_atom in 0..4 {
            for n_atom in 0..3 {
                for half in 0..2 {
                    for lane in 0..32 {
                        let ((row, column0), (row1, column1)) =
                            direct_source::direct_pair_coordinates(
                                warp, m_atom, n_atom, half, lane,
                            );
                        assert_eq!(row, row1);
                        assert_eq!(column1, column0 + 1);
                        for column in [column0, column1] {
                            assert!(row < 128 && column < 96);
                            owners[row * 96 + column] += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(owners.iter().all(|count| *count == 1));
}

#[test]
fn source_adapter_changes_only_epilogue_and_export() {
    let triad = retained_source::compose_triad_nn_n96_source(FIXED_N96_SOURCE).unwrap();
    let direct_retained = direct_source::compose_candidate_source(&triad).unwrap();
    let candidate = candidate_source::compose_candidate_source(&direct_retained).unwrap();
    assert!(candidate.contains("bool direct_pair_epilogue ="));
    assert!(candidate.contains("return;\n    }\n    __syncthreads();\n    float* tile_output"));
    assert_eq!(
        candidate_source::restore_direct_retained_source(&candidate).unwrap(),
        direct_retained
    );
}

#[test]
fn logical_nn_targets_have_the_frozen_n96_grids() {
    assert_eq!(direct_source::n96_grid(4_621, 1_928), 777);
}

#[test]
fn fast_screen_requires_four_strict_retained_strata() {
    assert!(strict_once7_win(&[[0.98, 0.989]; 4]));
    assert!(!strict_once7_win(&[[0.98, 0.99]; 4]));
    assert!(!strict_once7_win(&[[0.98, 0.989]; 3]));
    assert!(!strict_once7_win(&[
        [0.98, 0.989],
        [0.98, 0.989],
        [f64::NAN, 0.989],
        [0.98, 0.989],
    ]));
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

    const ENV: &str = "MAMBA_TRIAD_NN_N96_DIRECT_EPILOGUE_NOBARRIER_DISCOVERY";
    const SHARED: usize = 86_016;
    const OPS: usize = 20;
    const WINDOWS: usize = 7;
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
            let blocks = direct_source::n96_grid(self.m, self.n);
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
    }

    #[derive(Clone, Copy)]
    struct Case {
        label: &'static str,
        shape: Shape,
    }

    const PRISM: Case = Case {
        label: "prism_in_proj",
        shape: Shape {
            m: 4_621,
            k: 384,
            n: 1_928,
        },
    };
    const FULL_TILE: Case = Case {
        label: "full_tile_128x32x96",
        shape: Shape {
            m: 128,
            k: 32,
            n: 96,
        },
    };
    const TAIL: Case = Case {
        label: "tail_129x36x100",
        shape: Shape {
            m: 129,
            k: 36,
            n: 100,
        },
    };
    const K0: Case = Case {
        label: "k0_129x0x100",
        shape: Shape {
            m: 129,
            k: 0,
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
                beta: 0.0,
                m: i32::try_from(shape.m).map_err(|_| "M exceeds i32")?,
                k: i32::try_from(shape.k).map_err(|_| "K exceeds i32")?,
                n: i32::try_from(shape.n).map_err(|_| "N exceeds i32")?,
                lda: i32::try_from(shape.k).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(shape.n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(shape.n).map_err(|_| "ldc exceeds i32")?,
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
                Self::Candidate => "direct_epilogue_nobarrier_n96",
                Self::Retained => "measured_direct_epilogue_n96",
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
    }

    struct Fixture {
        a: GuardedF32,
        b: GuardedF32,
        candidate: GuardedF32,
        retained: GuardedF32,
        fast: GuardedF32,
    }

    impl Fixture {
        fn new(runtime: &Runtime, case: Case, exceptional: bool) -> Result<Self, String> {
            let shape = case.shape;
            let mut a = full_mantissa::finite_full_mantissa_values(shape.m * shape.k, 0xb196_a001);
            let mut b = full_mantissa::finite_full_mantissa_values(shape.k * shape.n, 0xb196_b002);
            if exceptional && shape.k > 0 {
                a.fill(0.0);
                b.fill(0.0);
                for row in 0..shape.m {
                    a[row * shape.k] = 1.0;
                }
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
                    if column < shape.n {
                        b[column] = f32::from_bits(bits);
                    }
                }
            }
            let output = vec![f32::from_bits(POISON_BITS); shape.m * shape.n];
            Ok(Self {
                a: GuardedF32::new(&runtime.ctx.stream, a, "A")?,
                b: GuardedF32::new(&runtime.ctx.stream, b, "B")?,
                candidate: GuardedF32::new(&runtime.ctx.stream, output.clone(), "candidate C")?,
                retained: GuardedF32::new(&runtime.ctx.stream, output.clone(), "retained C")?,
                fast: GuardedF32::new(&runtime.ctx.stream, output, "Fast C")?,
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
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _retained_module: Arc<CudaModule>,
        _candidate_module: Arc<CudaModule>,
        retained: CudaFunction,
        candidate: CudaFunction,
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
            return Err("N96 direct-epilogue timing requires --release".into());
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
        let triad_body = retained_source::compose_triad_nn_n96_source(FIXED_N96_SOURCE)?;
        let retained_body = direct_source::compose_candidate_source(&triad_body)?;
        let candidate_body = candidate_source::compose_candidate_source(&retained_body)?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) = compile_module(
            &device,
            module_source(&retained_body),
            direct_source::SYMBOL,
        )?;
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_module(
                &device,
                module_source(&candidate_body),
                candidate_source::SYMBOL,
            )?;
        let runtime = Runtime {
            _device: device,
            ctx,
            _retained_module: retained_module,
            _candidate_module: candidate_module,
            retained,
            candidate,
            retained_source_sha,
            candidate_source_sha,
            retained_ptx_sha,
            candidate_ptx_sha,
        };
        resource_gate(&runtime.retained, direct_source::SYMBOL, 128)?;
        resource_gate(&runtime.candidate, candidate_source::SYMBOL, 128)?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierIdentityV1",
                "retained_symbol":direct_source::SYMBOL,
                "candidate_symbol":candidate_source::SYMBOL,
                "retained_source_sha":runtime.retained_source_sha,
                "candidate_source_sha":runtime.candidate_source_sha,
                "retained_ptx_sha":runtime.retained_ptx_sha,
                "candidate_ptx_sha":runtime.candidate_ptx_sha,
                "conversion":"add_half_ulp_tf32_v1",
                "change":"skip_final_cta_barrier_for_full_tile_direct_float2_only",
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
                "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierResourceV1","symbol":symbol,
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

    fn launch(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
    ) -> Result<(), String> {
        let shape = case.shape;
        let output = fixture.output(arm).ptr(&runtime.ctx.stream);
        let (a, b) = if shape.k == 0 {
            (0, 0)
        } else {
            (
                fixture.a.ptr(&runtime.ctx.stream),
                fixture.b.ptr(&runtime.ctx.stream),
            )
        };
        match arm {
            Arm::Candidate | Arm::Retained => {
                let function = if arm == Arm::Candidate {
                    &runtime.candidate
                } else {
                    &runtime.retained
                };
                let bias = 0_u64;
                let params = Params::new(shape)?;
                let mut builder = runtime.ctx.stream.launch_builder(function);
                builder.arg(&output).arg(&a).arg(&b).arg(&bias).arg(&params);
                unsafe { builder.launch(shape.config()?) }
                    .map(|_| ())
                    .map_err(|e| format!("launch {}: {e:?}", arm.name()))
            }
            Arm::Fast => {
                if shape.k == 0 {
                    return Err("Fast comparator excludes K0".into());
                }
                let alpha = 1.0_f32;
                let beta = 0.0_f32;
                let dtype = WeightDtype::F32.cuda_data_type();
                unsafe {
                    blas_result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        shape.n as i32,
                        shape.m as i32,
                        shape.k as i32,
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
                    json!({"schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierGraphV1","arm":arm.name(),"nodes":count,"abi":"opaque","timing":"whole_graph"})
                );
                return Ok(());
            }
            if count != 1 {
                return Err(format!("{} graph has {count} nodes", arm.name()));
            }
            let mut node = std::ptr::null_mut();
            if sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count)
                != sys::CUresult::CUDA_SUCCESS
            {
                return Err(format!("{} graph node query failed", arm.name()));
            }
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
            if sys::cuGraphKernelNodeGetParams_v2(node, &mut params) != sys::CUresult::CUDA_SUCCESS
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
            let expected_symbol = if arm == Arm::Candidate {
                candidate_source::SYMBOL
            } else {
                direct_source::SYMBOL
            };
            let expected_grid = case.shape.grid()?;
            if symbol != expected_symbol
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != expected_grid
                || (params.blockDimX, params.blockDimY, params.blockDimZ) != (256, 1, 1)
                || params.sharedMemBytes != SHARED as u32
            {
                return Err(format!(
                    "{} graph identity changed: {symbol} grid={:?} block={:?} shared={}",
                    arm.name(),
                    (params.gridDimX, params.gridDimY, params.gridDimZ),
                    (params.blockDimX, params.blockDimY, params.blockDimZ),
                    params.sharedMemBytes
                ));
            }
            println!(
                "{}",
                json!({"schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierGraphV1","arm":arm.name(),"symbol":symbol,"grid":expected_grid,"block":[256,1,1],"shared":SHARED,"timing":"whole_graph"})
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
        if bits.iter().any(|word| *word == POISON_BITS) {
            return Err(format!("{} left an unwritten output word", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
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
        if case.shape.k == 0 && golden.iter().any(|word| *word != 0) {
            return Err(format!(
                "{} alpha1/beta0/no-bias K0 oracle is not positive zero",
                case.label
            ));
        }
        println!(
            "{}",
            json!({"schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierBitsV1","case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],"exceptional":exceptional,"candidate_retained_exact":true,"eager_repeats":2,"graph_repeats":2,"guards":true})
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
                "{} candidate differs from measured direct epilogue",
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
                "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierTargetBitsV1",
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
                "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierFastBitsV1",
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
        order: retained_source::BracketOrder,
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
        let mut raw = Vec::with_capacity(WINDOWS);
        let mut ratios = Vec::with_capacity(WINDOWS);
        for _ in 0..WINDOWS {
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
            ratios.push(retained_source::candidate_over_auto_ratio(
                order,
                observation,
            )?);
            raw.push(observation);
        }
        let result = [quantile(&ratios, 0.5), quantile(&ratios, 0.95)];
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierScreenV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid()?,
                "candidate":"direct_epilogue_nobarrier_n96","comparator":comparator.name(),"path":path.name(),
                "order":order.name(),"windows":WINDOWS,"warmups_per_arm":WARMUPS,
                "logical_gemms_per_observation":OPS,"raw_observations_us":raw,
                "ratio_direction":"candidate_over_comparator","ratio_p50":result[0],"ratio_p95":result[1],
            })
        );
        Ok(result)
    }

    #[test]
    #[ignore = "requires exclusive quiet CC8.9/142-SM CUDA13.2; isolated TF32 NN N96 direct-epilogue no-barrier once7"]
    fn ada_tf32_nn_n96_direct_epilogue_nobarrier_prism_once7() -> Result<(), String> {
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        let idle_resident_mode = common::gpu_quiet::idle_resident_mode_enabled();
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-nn-n96-direct-epilogue-nobarrier/pre", 1_800)?;
        } else {
            quiet.require_pre_context("tf32-nn-n96-direct-epilogue-nobarrier/pre")?;
        }
        let runtime = new_runtime()?;
        assert_eq!(PRISM.shape.grid()?, (777, 1, 1));
        check_exact_case(&runtime, FULL_TILE, false)?;
        check_exact_case(&runtime, FULL_TILE, true)?;
        check_exact_case(&runtime, TAIL, false)?;
        check_exact_case(&runtime, TAIL, true)?;
        check_exact_case(&runtime, K0, false)?;
        for case in [PRISM] {
            let mut prepared = prepare_target(&runtime, case)?;
            let mut retained_strata = Vec::with_capacity(4);
            let mut fast_strata = Vec::with_capacity(4);
            if idle_resident_mode {
                quiet
                    .require_idle_resident("tf32-nn-n96-direct-epilogue-nobarrier/retained", 256)?;
            } else {
                quiet.require_cohort("tf32-nn-n96-direct-epilogue-nobarrier/retained")?;
            }
            for path in [Path::Eager, Path::Graph] {
                for order in [
                    retained_source::BracketOrder::Abba,
                    retained_source::BracketOrder::Baab,
                ] {
                    retained_strata.push(screen(
                        &runtime,
                        case,
                        &mut prepared,
                        Arm::Retained,
                        path,
                        order,
                    )?);
                }
            }
            let retained_win = strict_once7_win(&retained_strata);
            if retained_win {
                prepare_fast(&runtime, case, &mut prepared)?;
                if idle_resident_mode {
                    quiet
                        .require_idle_resident("tf32-nn-n96-direct-epilogue-nobarrier/fast", 256)?;
                } else {
                    quiet.require_cohort("tf32-nn-n96-direct-epilogue-nobarrier/fast")?;
                }
                for path in [Path::Eager, Path::Graph] {
                    for order in [
                        retained_source::BracketOrder::Abba,
                        retained_source::BracketOrder::Baab,
                    ] {
                        fast_strata.push(screen(
                            &runtime,
                            case,
                            &mut prepared,
                            Arm::Fast,
                            path,
                            order,
                        )?);
                    }
                }
            }
            let fast_win = retained_win && strict_once7_win(&fast_strata);
            println!(
                "{}",
                json!({
                    "schema":"MambaBiTriadTf32NnN96DirectEpilogueNoBarrierDecisionV1","cell":case.label,
                    "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid()?,
                    "retained_strata":retained_strata,"fast_strata":fast_strata,
                    "strata_order":["eager/ABBA","eager/BAAB","graph/ABBA","graph/BAAB"],
                    "retained_win":retained_win,"fast_win":fast_win,
                    "decision":if !retained_win {
                        "stop_no_retry"
                    } else if fast_win {
                        "shortlist_strict_fast_win"
                    } else {
                        "retain_candidate_fast_miss"
                    },
                    "fast_screened":retained_win,
                    "fast_qualified":fast_win,"idle_resident_mode":idle_resident_mode,
                    "promotion":false,
                })
            );
        }
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-nn-n96-direct-epilogue-nobarrier/post", 256)?;
        } else {
            quiet.verify_post_cohort("tf32-nn-n96-direct-epilogue-nobarrier/post")?;
        }
        Ok(())
    }
}
