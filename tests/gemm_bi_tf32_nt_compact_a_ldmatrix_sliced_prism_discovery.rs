//! Test-only Ada TF32 NT Prism stage-sliced A-ldmatrix discovery.

#[path = "support/triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs"]
mod candidate_source;
#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;

const PRISM_TARGET: (usize, usize, usize) = (4_096, 3_072, 1_536);
const PRISM_GRID: u32 = 1_536;
const PRODUCTION_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");
const COMPACT_LAYOUT_SOURCE: &str = include_str!("gemm_bi_tf32_nt_compact_xor.cu");

#[test]
fn prism_target_grid_and_training_triad_family_are_frozen() {
    assert_eq!(PRISM_TARGET, (4_096, 3_072, 1_536));
    assert_eq!(
        4_096usize.div_ceil(128) * 3_072usize.div_ceil(64),
        PRISM_GRID as usize
    );
    assert!(candidate_source::SYMBOL.starts_with("gemm_bi_nt_"));
    assert!(candidate_source::RETAINED_SYMBOL.starts_with("gemm_bi_nt_"));
    let source =
        candidate_source::candidate_source(PRODUCTION_SOURCE, COMPACT_LAYOUT_SOURCE).unwrap();
    for anchor in [
        "#define GEMM_BI_TF32_DEFINE_KERNEL",
        "float* output, const float* a, const float* b, const float* bias,",
        "Sm80Tf32KernelParams params)",
        "using SgbTf32KernelSignature = void (*)(",
        &format!(
            "GEMM_BI_TF32_DEFINE_KERNEL({}, SgbTf32Nt, 128, 64, 2, 256, 1)",
            candidate_source::SYMBOL
        ),
        &format!(
            "TF32_ASSERT_KERNEL_SIGNATURE({});",
            candidate_source::SYMBOL
        ),
    ] {
        assert!(
            source.contains(anchor),
            "missing training-Triad anchor {anchor:?}"
        );
    }
    assert!(!source.contains("gemm_bi_nn_fixed"));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingPhase {
    Scout,
    Qualification,
}

const fn windows_for(phase: TimingPhase) -> usize {
    match phase {
        TimingPhase::Scout => 3,
        TimingPhase::Qualification => 7,
    }
}

fn may_run_fast(retained_strata: &[[f64; 2]]) -> bool {
    retained_strata.len() == 4
        && retained_strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < 0.99
                && *p95 < 0.99
        })
}

#[test]
fn timing_policy_is_once3_then_once7_retained_first() {
    assert_eq!(windows_for(TimingPhase::Scout), 3);
    assert_eq!(windows_for(TimingPhase::Qualification), 7);
    assert!(!may_run_fast(&[[0.98, 0.98]; 3]));
    assert!(!may_run_fast(&[[0.98, 0.99]; 4]));
    assert!(may_run_fast(&[[0.98, 0.989]; 4]));
}

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::ffi::{CStr, c_void};
    use std::sync::Arc;

    use cudarc::cublas::{result as blas_result, sys as blas};
    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use serde_json::json;
    use sha2::{Digest as _, Sha256};

    use super::candidate_source;
    use super::common::gpu_quiet::QuietGpu;
    use super::full_mantissa;
    use super::{PRISM_GRID, PRISM_TARGET, TimingPhase, may_run_fast, windows_for};

    const ENV: &str = "MAMBA_TRIAD_TF32_NT_A_LDMATRIX_SLICED_PRISM_DISCOVERY";
    const OPS: usize = 20;
    const WARMUPS: usize = 3;
    const GUARD: usize = 64;
    const ALIGNMENT: u64 = 256;
    const GUARD_BITS: u32 = 0x7fc0_a5d2;
    const POISON_BITS: u32 = 0x7fc0_c5d2;
    const PRODUCTION: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");
    const COMPACT_LAYOUT: &str = include_str!("gemm_bi_tf32_nt_compact_xor.cu");

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Shape {
        m: usize,
        k_out: usize,
        reduction: usize,
        alpha: f32,
        label: &'static str,
    }

    impl Shape {
        const fn new(
            m: usize,
            k_out: usize,
            reduction: usize,
            alpha: f32,
            label: &'static str,
        ) -> Self {
            Self {
                m,
                k_out,
                reduction,
                alpha,
                label,
            }
        }

        fn grid(self) -> Result<(u32, u32, u32), String> {
            let blocks = self
                .m
                .div_ceil(128)
                .checked_mul(self.k_out.div_ceil(64))
                .ok_or("grid overflow")?;
            Ok((u32::try_from(blocks).map_err(|_| "grid exceeds u32")?, 1, 1))
        }

        fn config(self) -> Result<LaunchConfig, String> {
            Ok(LaunchConfig {
                grid_dim: self.grid()?,
                block_dim: (candidate_source::BLOCK_THREADS, 1, 1),
                shared_mem_bytes: candidate_source::DYNAMIC_SHARED_BYTES,
            })
        }
    }

    const TARGET: Shape = Shape::new(4_096, 3_072, 1_536, 1.0, "prism");
    const M_TAIL: Shape = Shape::new(129, 64, 64, -0.75, "m_tail_negative_alpha");
    const N_TAIL: Shape = Shape::new(128, 67, 64, 1.0, "n_tail");
    const K_TAIL: Shape = Shape::new(128, 64, 71, 1.0, "k_tail");
    const EXCEPTION: Shape = Shape::new(128, 64, 64, 1.0, "exception");
    const K0: Shape = Shape::new(129, 67, 0, 1.0, "k0");

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
                alpha: shape.alpha,
                beta: 0.0,
                m: i32::try_from(shape.m).map_err(|_| "m exceeds i32")?,
                k: i32::try_from(shape.k_out).map_err(|_| "k_out exceeds i32")?,
                n: i32::try_from(shape.reduction).map_err(|_| "reduction exceeds i32")?,
                lda: i32::try_from(shape.reduction).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(shape.reduction).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(shape.k_out).map_err(|_| "ldc exceeds i32")?,
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
                Self::Candidate => "a_ldmatrix_stage_sliced",
                Self::Retained => "retained_a_ldmatrix",
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

    #[derive(Clone, Copy)]
    enum Order {
        Abba,
        Baab,
    }

    impl Order {
        const fn name(self) -> &'static str {
            match self {
                Self::Abba => "ABBA",
                Self::Baab => "BAAB",
            }
        }

        fn arms(self, comparator: Arm) -> [Arm; 4] {
            match self {
                Self::Abba => [Arm::Candidate, comparator, comparator, Arm::Candidate],
                Self::Baab => [comparator, Arm::Candidate, Arm::Candidate, comparator],
            }
        }
    }

    struct GuardedF32 {
        buffer: GpuBuffer,
        baseline: Vec<f32>,
        active: usize,
        label: &'static str,
    }

    impl GuardedF32 {
        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            label: &'static str,
        ) -> Result<Self, String> {
            let active_len = active.len();
            let mut baseline = vec![f32::from_bits(GUARD_BITS); GUARD + active_len + GUARD];
            baseline[GUARD..GUARD + active_len].copy_from_slice(&active);
            let buffer = GpuBuffer::from_cpu(stream, &baseline)?;
            let result = Self {
                buffer,
                baseline,
                active: active_len,
                label,
            };
            if result.buffer.cached_ptr() % ALIGNMENT != 0 || result.ptr(stream) % ALIGNMENT != 0 {
                return Err(format!(
                    "{} base/logical pointer is not 256B aligned",
                    label
                ));
            }
            Ok(result)
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
                .chain(&values[GUARD + self.active..])
                .any(|value| value.to_bits() != GUARD_BITS)
            {
                return Err(format!("{} red zone changed", self.label));
            }
            Ok(values[GUARD..GUARD + self.active]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn unchanged(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            let values = self.buffer.to_cpu(stream)?;
            if values
                .iter()
                .zip(&self.baseline)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{} input or guard changed", self.label));
            }
            Ok(())
        }
    }

    struct Fixture {
        shape: Shape,
        a: GuardedF32,
        b: GuardedF32,
        candidate: GuardedF32,
        retained: GuardedF32,
        fast: GuardedF32,
    }

    impl Fixture {
        fn new(runtime: &Runtime, shape: Shape, exceptional: bool) -> Result<Self, String> {
            let mut a =
                full_mantissa::finite_full_mantissa_values(shape.m * shape.reduction, 0xa5d2_a001);
            let mut b = full_mantissa::finite_full_mantissa_values(
                shape.k_out * shape.reduction,
                0xa5d2_b002,
            );
            if exceptional && shape.reduction != 0 {
                a.fill(0.0);
                b.fill(0.0);
                for row in 0..shape.m {
                    a[row * shape.reduction] = 1.0;
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
                    if column < shape.k_out {
                        b[column * shape.reduction] = f32::from_bits(bits);
                    }
                }
            }
            let output = vec![f32::from_bits(POISON_BITS); shape.m * shape.k_out];
            Ok(Self {
                shape,
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
        _candidate_module: Arc<CudaModule>,
        _retained_module: Arc<CudaModule>,
        candidate: CudaFunction,
        retained: CudaFunction,
        candidate_source_sha: String,
        retained_source_sha: String,
        candidate_ptx_sha: String,
        retained_ptx_sha: String,
    }

    fn strip_typed_include(source: &str) -> String {
        source
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn composed_source(transformed: &str) -> String {
        [
            include_str!("../kernels/_typed_prelude.cuh").to_owned(),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/contract.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/common.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/epilogue.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/mma16.cuh")),
            transformed.to_owned(),
        ]
        .join("\n")
    }

    fn compile_ptx_only(
        transformed: &str,
        label: &str,
        supports_random_seed: bool,
    ) -> Result<(cudarc::nvrtc::Ptx, String), String> {
        let source = composed_source(transformed);
        let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let mut options = vec![
            "--fmad=true".into(),
            "--extra-device-vectorization".into(),
            "-DNDEBUG".into(),
            "-DGEMM_BI_GROUP_M=16".into(),
            "-DMAMBA_RS_STATE_CAP=256".into(),
        ];
        if supports_random_seed {
            options.push("--frandom-seed=1295072049".into());
        }
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("compute_89"),
                options,
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("NVRTC compile {label}: {error:?}"))?;
        Ok((ptx, source_sha))
    }

    fn compile_source(
        device: &GpuDevice,
        transformed: &str,
        symbol: &str,
        supports_random_seed: bool,
    ) -> Result<(Arc<CudaModule>, CudaFunction, String, String), String> {
        let (ptx, source_sha) = compile_ptx_only(transformed, symbol, supports_random_seed)?;
        let ptx_sha = format!("{:x}", Sha256::digest(ptx.to_src().as_bytes()));
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load module {symbol}: {error:?}"))?;
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                candidate_source::DYNAMIC_SHARED_BYTES as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        Ok((module, function, source_sha, ptx_sha))
    }

    #[test]
    #[ignore = "CUDA13.2 NVRTC compile-only; creates no device or CUDA context"]
    fn cuda132_nvrtc_source_compile_only() -> Result<(), String> {
        let candidate = candidate_source::candidate_source(PRODUCTION, COMPACT_LAYOUT)?;
        let retained = candidate_source::retained_source(PRODUCTION, COMPACT_LAYOUT)?;
        let (candidate_ptx, candidate_source_sha) =
            compile_ptx_only(&candidate, "TF32 NT sliced candidate", true)?;
        let (retained_ptx, retained_source_sha) =
            compile_ptx_only(&retained, "TF32 NT retained A-only", true)?;
        let candidate_ptx = candidate_ptx.to_src();
        let retained_ptx = retained_ptx.to_src();
        for anchor in [
            candidate_source::SYMBOL,
            "cp.async.cg.shared.global",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ] {
            if !candidate_ptx.contains(anchor) {
                return Err(format!("candidate PTX missing {anchor:?}"));
            }
        }
        for anchor in [
            candidate_source::RETAINED_SYMBOL,
            "cp.async.cg.shared.global",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ] {
            if !retained_ptx.contains(anchor) {
                return Err(format!("retained PTX missing {anchor:?}"));
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismCompileV1",
                "candidate_source_sha":candidate_source_sha,
                "retained_source_sha":retained_source_sha,
                "candidate_ptx_sha":format!("{:x}", Sha256::digest(candidate_ptx.as_bytes())),
                "retained_ptx_sha":format!("{:x}", Sha256::digest(retained_ptx.as_bytes())),
            })
        );
        Ok(())
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var(ENV).as_deref() != Ok("1") {
            return Err(format!("set {ENV}=1"));
        }
        if cfg!(debug_assertions) {
            return Err("TF32 NT sliced timing requires --release".into());
        }
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
            return Err(format!(
                "requires CC8.9/142-SM Ada, found {:?}/{}",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        let compiler = ctx.kernels.compiler_identity();
        if !matches!(compiler.nvrtc_version, (12, 8) | (13, 0) | (13, 2))
            || compiler.target.as_str() != "sm_89"
        {
            return Err(format!(
                "requires supported CUDA12.8/13.0/13.2 on sm_89, found {compiler:?}"
            ));
        }
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
        let candidate_body = candidate_source::candidate_source(PRODUCTION, COMPACT_LAYOUT)?;
        let retained_body = candidate_source::retained_source(PRODUCTION, COMPACT_LAYOUT)?;
        let supports_random_seed = compiler.nvrtc_version >= (13, 0);
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_source(
                &device,
                &candidate_body,
                candidate_source::SYMBOL,
                supports_random_seed,
            )?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) = compile_source(
            &device,
            &retained_body,
            candidate_source::RETAINED_SYMBOL,
            supports_random_seed,
        )?;
        let runtime = Runtime {
            _device: device,
            ctx,
            _candidate_module: candidate_module,
            _retained_module: retained_module,
            candidate,
            retained,
            candidate_source_sha,
            retained_source_sha,
            candidate_ptx_sha,
            retained_ptx_sha,
        };
        resource_gate(&runtime.candidate, candidate_source::SYMBOL)?;
        resource_gate(&runtime.retained, candidate_source::RETAINED_SYMBOL)?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismIdentityV1",
                "candidate_symbol":candidate_source::SYMBOL,
                "retained_symbol":candidate_source::RETAINED_SYMBOL,
                "candidate_source_sha":runtime.candidate_source_sha,
                "retained_source_sha":runtime.retained_source_sha,
                "candidate_ptx_sha":runtime.candidate_ptx_sha,
                "retained_ptx_sha":runtime.retained_ptx_sha,
                "change":"four_next_stage_copy_slices_interleaved_before_ascending_k8_issues",
            })
        );
        Ok(runtime)
    }

    fn resource_gate(function: &CudaFunction, symbol: &str) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|e| format!("{symbol} regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("{symbol} local: {e:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("{symbol} static shared: {e:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|e| format!("{symbol} max threads: {e:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|e| format!("{symbol} max dynamic: {e:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                candidate_source::BLOCK_THREADS,
                candidate_source::DYNAMIC_SHARED_BYTES as usize,
                None,
            )
            .map_err(|e| format!("{symbol} occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismResourceV1",
                "symbol":symbol,"threads":candidate_source::BLOCK_THREADS,
                "registers":registers,"local_bytes":local,
                "static_shared_bytes":static_shared,
                "dynamic_shared_bytes":candidate_source::DYNAMIC_SHARED_BYTES,
                "max_dynamic_shared_bytes":max_dynamic,
                "max_threads":max_threads,"occupancy":occupancy,
            })
        );
        if registers <= 0
            || registers > candidate_source::REGISTER_CAP
            || local != 0
            || static_shared != 0
            || max_threads < candidate_source::BLOCK_THREADS as i32
            || max_dynamic < candidate_source::DYNAMIC_SHARED_BYTES as i32
            || occupancy < candidate_source::REQUIRED_OCCUPANCY
        {
            return Err(format!(
                "{symbol} resource gate failed: regs={registers} local={local} static={static_shared} max_dynamic={max_dynamic} occupancy={occupancy}"
            ));
        }
        Ok(())
    }

    fn launch(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let shape = fixture.shape;
        let output = fixture.output(arm).ptr(&runtime.ctx.stream);
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
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
                if shape.reduction == 0 {
                    return Err("Fast comparator excludes K0".into());
                }
                let beta = 0.0_f32;
                let dtype = WeightDtype::F32.cuda_data_type();
                unsafe {
                    blas_result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_T,
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        shape.k_out as i32,
                        shape.m as i32,
                        shape.reduction as i32,
                        (&shape.alpha as *const f32).cast::<c_void>(),
                        b as *const c_void,
                        dtype,
                        shape.reduction as i32,
                        a as *const c_void,
                        dtype,
                        shape.reduction as i32,
                        (&beta as *const f32).cast::<c_void>(),
                        output as *mut c_void,
                        dtype,
                        shape.k_out as i32,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|e| format!("cuBLAS Fast TF32: {e:?}"))
            }
        }
    }

    fn capture(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<CudaGraph, String> {
        launch(runtime, fixture, arm)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} warmup sync: {e:?}", arm.name()))?;
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, arm)) }
    }

    fn graph_identity(graph: &CudaGraph, shape: Shape, arm: Arm) -> Result<(), String> {
        unsafe {
            let mut count = 0usize;
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count == 0
            {
                return Err(format!("{} graph empty/unqueryable", arm.name()));
            }
            if arm == Arm::Fast {
                println!(
                    "{}",
                    json!({"schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismGraphV1","arm":arm.name(),"nodes":count,"abi":"opaque","timing":"whole_graph"})
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
                candidate_source::RETAINED_SYMBOL
            };
            if symbol != expected_symbol
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != shape.grid()?
                || (params.blockDimX, params.blockDimY, params.blockDimZ)
                    != (candidate_source::BLOCK_THREADS, 1, 1)
                || params.sharedMemBytes != candidate_source::DYNAMIC_SHARED_BYTES
            {
                return Err(format!(
                    "{} graph identity changed: {symbol} grid={:?} block={:?} shared={}",
                    arm.name(),
                    (params.gridDimX, params.gridDimY, params.gridDimZ),
                    (params.blockDimX, params.blockDimY, params.blockDimZ),
                    params.sharedMemBytes
                ));
            }
            for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
                .into_iter()
                .enumerate()
            {
                let mut offset = 0usize;
                let mut size = 0usize;
                if sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size)
                    != sys::CUresult::CUDA_SUCCESS
                    || (offset, size) != expected
                {
                    return Err(format!(
                        "{} graph ABI parameter {index} changed: {:?}",
                        arm.name(),
                        (offset, size)
                    ));
                }
            }
            let mut offset = 0usize;
            let mut size = 0usize;
            if sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size)
                != sys::CUresult::CUDA_ERROR_INVALID_VALUE
            {
                return Err(format!(
                    "{} graph ABI accepted a sixth argument",
                    arm.name()
                ));
            }
        }
        Ok(())
    }

    fn output_after(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: Option<&CudaGraph>,
    ) -> Result<Vec<u32>, String> {
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        if let Some(graph) = graph {
            graph
                .launch()
                .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?;
        } else {
            launch(runtime, fixture, arm)?;
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} sync: {e:?}", arm.name()))?;
        let bits = fixture.output(arm).bits(&runtime.ctx.stream)?;
        if bits.iter().any(|word| *word == POISON_BITS) {
            return Err(format!("{} left an unwritten output", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        Ok(bits)
    }

    fn check_exact_case(runtime: &Runtime, shape: Shape, exceptional: bool) -> Result<(), String> {
        let mut fixture = Fixture::new(runtime, shape, exceptional)?;
        let golden = output_after(runtime, &mut fixture, Arm::Retained, None)?;
        let candidate_graph = capture(runtime, &fixture, Arm::Candidate)?;
        let retained_graph = capture(runtime, &fixture, Arm::Retained)?;
        graph_identity(&candidate_graph, shape, Arm::Candidate)?;
        graph_identity(&retained_graph, shape, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (Arm::Candidate, &candidate_graph),
                (Arm::Retained, &retained_graph),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, arm, path)? != golden {
                        return Err(format!(
                            "{} {} repeat {repeat} changed exact bits",
                            shape.label,
                            arm.name()
                        ));
                    }
                }
            }
        }
        if shape.reduction == 0 && golden.iter().any(|word| *word != 0) {
            return Err(format!("{} K0 oracle is not positive zero", shape.label));
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismBitsV1",
                "case":shape.label,"shape":[shape.m,shape.k_out,shape.reduction],
                "exceptional":exceptional,"candidate_retained_exact":true,
                "eager_repeats":2,"graph_repeats":2,"guards":true,
            })
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

    fn prepare_target(runtime: &Runtime) -> Result<Prepared, String> {
        let mut fixture = Fixture::new(runtime, TARGET, false)?;
        let exact_bits = output_after(runtime, &mut fixture, Arm::Retained, None)?;
        if output_after(runtime, &mut fixture, Arm::Candidate, None)? != exact_bits {
            return Err("target candidate differs from retained A-only".into());
        }
        let candidate_graph = capture(runtime, &fixture, Arm::Candidate)?;
        let retained_graph = capture(runtime, &fixture, Arm::Retained)?;
        graph_identity(&candidate_graph, TARGET, Arm::Candidate)?;
        graph_identity(&retained_graph, TARGET, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph, expected) in [
                (Arm::Candidate, &candidate_graph, exact_bits.as_slice()),
                (Arm::Retained, &retained_graph, exact_bits.as_slice()),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, arm, path)? != expected {
                        return Err(format!(
                            "target {} {} repeat {repeat} changed bits",
                            arm.name(),
                            if path.is_some() { "graph" } else { "eager" }
                        ));
                    }
                }
            }
        }
        Ok(Prepared {
            fixture,
            candidate_graph,
            retained_graph,
            fast_graph: None,
            exact_bits,
            fast_bits: None,
        })
    }

    fn prepare_fast(runtime: &Runtime, prepared: &mut Prepared) -> Result<(), String> {
        if prepared.fast_graph.is_some() || prepared.fast_bits.is_some() {
            return Err("Fast comparator was prepared more than once".into());
        }
        let fast_bits = output_after(runtime, &mut prepared.fixture, Arm::Fast, None)?;
        if fast_bits.is_empty()
            || fast_bits.iter().all(|word| word & 0x7fff_ffff == 0)
            || fast_bits
                .iter()
                .any(|word| !f32::from_bits(*word).is_finite())
        {
            return Err("Fast comparator produced invalid target output".into());
        }
        let fast_graph = capture(runtime, &prepared.fixture, Arm::Fast)?;
        graph_identity(&fast_graph, TARGET, Arm::Fast)?;
        for repeat in 0..2 {
            for path in [None, Some(&fast_graph)] {
                if output_after(runtime, &mut prepared.fixture, Arm::Fast, path)? != fast_bits {
                    return Err(format!(
                        "target Fast {} repeat {repeat} changed bits",
                        if path.is_some() { "graph" } else { "eager" }
                    ));
                }
            }
        }
        prepared.fast_graph = Some(fast_graph);
        prepared.fast_bits = Some(fast_bits);
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        prepared: &mut Prepared,
        arm: Arm,
        path: Path,
    ) -> Result<f64, String> {
        prepared
            .fixture
            .output_mut(arm)
            .reset(&runtime.ctx.stream)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("{} start event: {e:?}", arm.name()))?;
        for _ in 0..OPS {
            match path {
                Path::Eager => launch(runtime, &prepared.fixture, arm)?,
                Path::Graph => {
                    let graph = match arm {
                        Arm::Candidate => &prepared.candidate_graph,
                        Arm::Retained => &prepared.retained_graph,
                        Arm::Fast => prepared
                            .fast_graph
                            .as_ref()
                            .ok_or("Fast graph used before retained qualification")?,
                    };
                    graph
                        .launch()
                        .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?;
                }
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
        let actual = prepared.fixture.output(arm).bits(&runtime.ctx.stream)?;
        let expected = if arm == Arm::Fast {
            prepared
                .fast_bits
                .as_ref()
                .ok_or("Fast bits used before retained qualification")?
        } else {
            &prepared.exact_bits
        };
        if actual != *expected {
            return Err(format!("{} timed bits changed", arm.name()));
        }
        prepared.fixture.validate_inputs(runtime)?;
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
        prepared: &mut Prepared,
        comparator: Arm,
        path: Path,
        order: Order,
        windows: usize,
        phase: TimingPhase,
    ) -> Result<[f64; 2], String> {
        if comparator == Arm::Candidate {
            return Err("candidate cannot compare against itself".into());
        }
        for _ in 0..WARMUPS {
            measure(runtime, prepared, Arm::Candidate, path)?;
            measure(runtime, prepared, comparator, path)?;
        }
        let mut raw = Vec::with_capacity(windows);
        let mut ratios = Vec::with_capacity(windows);
        for _ in 0..windows {
            let mut observation = [0.0; 4];
            for (index, arm) in order.arms(comparator).into_iter().enumerate() {
                observation[index] = measure(runtime, prepared, arm, path)?;
            }
            let (candidate_us, comparator_us) = match order {
                Order::Abba => (
                    (observation[0] + observation[3]) * 0.5,
                    (observation[1] + observation[2]) * 0.5,
                ),
                Order::Baab => (
                    (observation[1] + observation[2]) * 0.5,
                    (observation[0] + observation[3]) * 0.5,
                ),
            };
            ratios.push(candidate_us / comparator_us);
            raw.push(observation);
        }
        let result = [quantile(&ratios, 0.5), quantile(&ratios, 0.95)];
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismScreenV1",
                "cell":TARGET.label,"shape":[TARGET.m,TARGET.k_out,TARGET.reduction],
                "grid":TARGET.grid()?,"candidate":Arm::Candidate.name(),
                "comparator":comparator.name(),"path":path.name(),"order":order.name(),
                "phase":format!("{phase:?}"),"windows":windows,
                "warmups_per_arm":WARMUPS,"logical_gemms_per_observation":OPS,
                "raw_observations_us":raw,"ratio_direction":"candidate_over_comparator",
                "ratio_p50":result[0],"ratio_p95":result[1],
            })
        );
        Ok(result)
    }

    fn all_strata(
        runtime: &Runtime,
        prepared: &mut Prepared,
        comparator: Arm,
        windows: usize,
        phase: TimingPhase,
    ) -> Result<Vec<[f64; 2]>, String> {
        let mut strata = Vec::with_capacity(4);
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                strata.push(screen(
                    runtime, prepared, comparator, path, order, windows, phase,
                )?);
            }
        }
        Ok(strata)
    }

    #[test]
    #[ignore = "requires exclusive quiet CC8.9/142-SM CUDA12.8/13.0/13.2; TF32 NT sliced Prism once3/once7"]
    fn ada_tf32_nt_a_ldmatrix_sliced_prism_scout3_then_once7() -> Result<(), String> {
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast TF32 disabled".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("tf32-nt-a-ldmatrix-sliced-prism/pre")?;
        let runtime = new_runtime()?;
        assert_eq!((TARGET.m, TARGET.k_out, TARGET.reduction), PRISM_TARGET);
        assert_eq!(TARGET.grid()?, (PRISM_GRID, 1, 1));
        check_exact_case(&runtime, TARGET, false)?;
        check_exact_case(&runtime, M_TAIL, false)?;
        check_exact_case(&runtime, N_TAIL, false)?;
        check_exact_case(&runtime, K_TAIL, false)?;
        check_exact_case(&runtime, EXCEPTION, true)?;
        check_exact_case(&runtime, K0, false)?;
        let mut prepared = prepare_target(&runtime)?;
        quiet.require_cohort("tf32-nt-a-ldmatrix-sliced-prism/scout")?;
        let scout = all_strata(
            &runtime,
            &mut prepared,
            Arm::Retained,
            windows_for(TimingPhase::Scout),
            TimingPhase::Scout,
        )?;
        let scout_pass = may_run_fast(&scout);
        if !scout_pass {
            println!(
                "{}",
                json!({
                    "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismDecisionV1",
                    "cell":TARGET.label,"retained_scout":scout,
                    "decision":"stop_scout_retained_loss","promotion":false,
                })
            );
            quiet.verify_post_cohort("tf32-nt-a-ldmatrix-sliced-prism/scout-post")?;
            return Ok(());
        }
        quiet.require_cohort("tf32-nt-a-ldmatrix-sliced-prism/qualification")?;
        let retained = all_strata(
            &runtime,
            &mut prepared,
            Arm::Retained,
            windows_for(TimingPhase::Qualification),
            TimingPhase::Qualification,
        )?;
        if !may_run_fast(&retained) {
            println!(
                "{}",
                json!({
                    "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismDecisionV1",
                    "cell":TARGET.label,"retained_scout":scout,"retained_once7":retained,
                    "decision":"stop_retained_loss","promotion":false,
                })
            );
            quiet.verify_post_cohort("tf32-nt-a-ldmatrix-sliced-prism/retained-post")?;
            return Ok(());
        }
        prepare_fast(&runtime, &mut prepared)?;
        let fast = all_strata(
            &runtime,
            &mut prepared,
            Arm::Fast,
            windows_for(TimingPhase::Qualification),
            TimingPhase::Qualification,
        )?;
        let fast_win = may_run_fast(&fast);
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NtALdmatrixSlicedPrismDecisionV1",
                "cell":TARGET.label,"shape":[TARGET.m,TARGET.k_out,TARGET.reduction],
                "retained_scout":scout,"retained_once7":retained,"fast_once7":fast,
                "strata_order":["eager/ABBA","eager/BAAB","graph/ABBA","graph/BAAB"],
                "retained_win":true,"fast_win":fast_win,
                "decision":if fast_win { "shortlist_fast_win" } else { "shortlist_retained_only" },
                "promotion":false,
            })
        );
        quiet.verify_post_cohort("tf32-nt-a-ldmatrix-sliced-prism/post")?;
        Ok(())
    }
}
