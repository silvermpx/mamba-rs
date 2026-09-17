use mamba_rs::mamba_ssm::gpu::GemmMode;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QualificationPhase {
    ModuleAbiResources,
    FiniteEager,
    FiniteGraph,
    ExceptionalEager,
    ExceptionalGraph,
    NonUnitAlpha,
    K0,
    RejectedNeighbors,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CaseReceipt {
    phases: Vec<QualificationPhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScreenOutcome {
    Once3Stop,
    Once7Stop,
    Retained,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchReceipt {
    cases: Vec<CaseReceipt>,
    screen: ScreenOutcome,
    fast_context_labelled: bool,
}

const REQUIRED_PHASES: [QualificationPhase; 8] = [
    QualificationPhase::ModuleAbiResources,
    QualificationPhase::FiniteEager,
    QualificationPhase::FiniteGraph,
    QualificationPhase::ExceptionalEager,
    QualificationPhase::ExceptionalGraph,
    QualificationPhase::NonUnitAlpha,
    QualificationPhase::K0,
    QualificationPhase::RejectedNeighbors,
];

fn validate_batch_receipt(receipt: &BatchReceipt) -> Result<(), String> {
    if receipt.cases.len() != 2 {
        return Err(format!(
            "exact-F32 d128 admission requires two cases, got {}",
            receipt.cases.len()
        ));
    }
    for (index, case) in receipt.cases.iter().enumerate() {
        if case.phases != REQUIRED_PHASES {
            return Err(format!(
                "exact-F32 d128 case {index} phase order changed: {:?}",
                case.phases
            ));
        }
    }
    if receipt.fast_context_labelled != matches!(receipt.screen, ScreenOutcome::Retained) {
        return Err("Fast context may be labelled only after once3 and once7 retain".into());
    }
    Ok(())
}

fn splitm64_delta(dims: (usize, usize, usize), a: &[f32], b: &[f32], alpha: f32) -> Vec<f32> {
    let (m, k, n) = dims;
    assert_eq!(m, 1_024);
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), m * n);
    let mut delta = vec![0.0_f32; k * n];
    for row in 0..k {
        for column in 0..n {
            let mut sum = 0.0_f64;
            for chunk in 0..64 {
                let mut partial = 0.0_f32;
                for offset in 0..16 {
                    let reduction = chunk * 16 + offset;
                    partial = a[reduction * k + row].mul_add(b[reduction * n + column], partial);
                }
                if chunk == 0 {
                    sum = partial as f64;
                } else {
                    sum += partial as f64;
                }
            }
            delta[row * n + column] = ((alpha as f64) * sum) as f32;
        }
    }
    delta
}

fn apply_delta_bits(initial: &[f32], delta: &[f32], repeats: usize) -> Vec<u32> {
    assert_eq!(initial.len(), delta.len());
    let mut output = initial.to_vec();
    for _ in 0..repeats {
        for (output, delta) in output.iter_mut().zip(delta) {
            *output += *delta;
        }
    }
    output.into_iter().map(f32::to_bits).collect()
}

#[cfg(feature = "cuda")]
#[path = "../../tests/support/fixed_full_mantissa.rs"]
mod full_mantissa;

#[cfg(feature = "cuda")]
mod live {
    use super::*;
    use crate::full_mantissa::finite_full_mantissa_values;
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dw_grad;
    use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        D128_IN_SYMBOL, D128_OUT_SYMBOL, PhysicalQualificationF32Epilogue,
        PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
        SM89_EXACT_F32_D128_KERNEL_SPECS, Sm89ExactF32D128Route, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp, ResolvedNumericContract,
        ResolvedOutputOwnership,
    };
    use serde_json::json;

    const OPS_PER_WINDOW: usize = 20;
    const RAW_GUARD_ELEMENTS: usize = 64;
    const RAW_GUARD_BITS: u32 = 0x7fc0_d128;
    const PRIOR_SYMBOLS: [&str; 2] = ["tn_splitm_partial_aligned", "splitm_reduce"];

    #[derive(Clone, Copy)]
    struct Case {
        route: Sm89ExactF32D128Route,
        dims: (usize, usize, usize),
        symbol: &'static str,
        tile: (u32, u32),
        dynamic_shared_bytes: u32,
        prior_grids: [(u32, u32, u32); 2],
    }

    const CASES: [Case; 2] = [
        Case {
            route: Sm89ExactF32D128Route::D128InDirectFold,
            dims: (1_024, 128, 512),
            symbol: D128_IN_SYMBOL,
            tile: (16, 16),
            dynamic_shared_bytes: 4_096,
            prior_grids: [(4, 1, 64), (256, 1, 1)],
        },
        Case {
            route: Sm89ExactF32D128Route::D128OutDirectFold,
            dims: (1_024, 256, 128),
            symbol: D128_OUT_SYMBOL,
            tile: (8, 16),
            dynamic_shared_bytes: 3_072,
            prior_grids: [(2, 1, 64), (128, 1, 1)],
        },
    ];

    #[derive(Clone)]
    struct Fixture {
        output: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        delta: Vec<f32>,
    }

    impl Fixture {
        fn words(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            (
                self.output.iter().copied().map(f32::to_bits).collect(),
                self.a.iter().copied().map(f32::to_bits).collect(),
                self.b.iter().copied().map(f32::to_bits).collect(),
            )
        }

        fn expected(&self, repeats: usize) -> Vec<u32> {
            apply_delta_bits(&self.output, &self.delta, repeats)
        }
    }

    struct RawGuardedBuffer {
        gpu: GpuBuffer,
        seed: Vec<f32>,
        active: usize,
        label: &'static str,
    }

    impl RawGuardedBuffer {
        fn new(ctx: &GpuCtx, active: &[f32], label: &'static str) -> Result<Self, String> {
            let mut seed = vec![f32::from_bits(RAW_GUARD_BITS); RAW_GUARD_ELEMENTS];
            seed.extend_from_slice(active);
            seed.resize(
                RAW_GUARD_ELEMENTS + active.len() + RAW_GUARD_ELEMENTS,
                f32::from_bits(RAW_GUARD_BITS),
            );
            let gpu = GpuBuffer::from_cpu(&ctx.stream, &seed)?;
            let active_pointer =
                gpu.cached_ptr() + (RAW_GUARD_ELEMENTS * std::mem::size_of::<f32>()) as u64;
            if !active_pointer.is_multiple_of(256) {
                return Err(format!("{label} active pointer is not 256-byte aligned"));
            }
            Ok(Self {
                gpu,
                seed,
                active: active.len(),
                label,
            })
        }

        fn ptr(&self) -> u64 {
            self.gpu.cached_ptr() + (RAW_GUARD_ELEMENTS * std::mem::size_of::<f32>()) as u64
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.gpu.upload(&ctx.stream, &self.seed)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize {} reset: {error:?}", self.label))
        }

        fn active_bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let values = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize {} readback: {error:?}", self.label))?;
            let active_end = RAW_GUARD_ELEMENTS + self.active;
            if values[..RAW_GUARD_ELEMENTS]
                .iter()
                .chain(&values[active_end..])
                .any(|value| value.to_bits() != RAW_GUARD_BITS)
            {
                return Err(format!(
                    "{} leading/trailing 256-byte guard changed",
                    self.label
                ));
            }
            Ok(values[RAW_GUARD_ELEMENTS..active_end]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
            let actual = self.active_bits(ctx)?;
            let expected = self.seed[RAW_GUARD_ELEMENTS..RAW_GUARD_ELEMENTS + self.active]
                .iter()
                .map(|value| value.to_bits());
            if actual.into_iter().ne(expected) {
                return Err(format!("{} active input changed", self.label));
            }
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum TimedPath {
        Eager,
        Graph,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum AdmissionMode {
        PreAdmission,
        PostAdmission,
    }

    impl AdmissionMode {
        fn require(self) -> Result<(), String> {
            let expected = match self {
                Self::PreAdmission => "0",
                Self::PostAdmission => "1",
            };
            if std::env::var("MAMBA_SM89_EXACT_F32_D128_EXPECT_AUTO")
                .ok()
                .as_deref()
                != Some(expected)
            {
                return Err(format!(
                    "{self:?} requires MAMBA_SM89_EXACT_F32_D128_EXPECT_AUTO={expected}"
                ));
            }
            Ok(())
        }

        fn label(self) -> &'static str {
            match self {
                Self::PreAdmission => "pre_admission",
                Self::PostAdmission => "post_admission",
            }
        }
    }

    fn runtime() -> Result<GpuCtx, String> {
        let device = GpuDevice::new(0)?;
        let ctx = GpuCtx::new(&device)?;
        let identity = ctx.gemm_route();
        if identity.device.compute_capability != (8, 9)
            || identity.device.multiprocessor_count != 142
        {
            return Err(format!(
                "SM89 exact-F32 d128 qualification requires CC8.9/142SM, got {:?}/{}SM",
                identity.device.compute_capability, identity.device.multiprocessor_count
            ));
        }
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.route_controls().set_family(BiGemmFamily::Triad);
        ctx.route_controls().set_tensor_cores(false);
        ctx.route_controls()
            .set_f32_policy(F32TriadPolicy::ExactScalarFma);
        Ok(ctx)
    }

    fn request(
        route: PhysicalQualificationRoute,
        dims: (usize, usize, usize),
        alpha: f32,
        beta: f32,
        bias: bool,
    ) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            dims,
            route,
            PhysicalQualificationF32Epilogue::new(alpha, beta, bias),
        )
    }

    fn finite_fixture(case: Case, alpha: f32) -> Fixture {
        let (m, k, n) = case.dims;
        let output = finite_full_mantissa_values(k * n, 0xc128_0003);
        let a = finite_full_mantissa_values(m * k, 0xa128_0001);
        let b = finite_full_mantissa_values(m * n, 0xb128_0002);
        let delta = splitm64_delta(case.dims, &a, &b, alpha);
        Fixture {
            output,
            a,
            b,
            delta,
        }
    }

    fn exceptional_fixture(case: Case) -> Fixture {
        let mut fixture = finite_fixture(case, 1.0);
        let (_, k, n) = case.dims;
        let exceptional = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7fa1_2345,
            0xffc5_4321,
            0xffa5_4321,
        ];
        for chunk in 0..64 {
            let boundary = chunk * 16;
            for (index, bits) in exceptional.into_iter().enumerate() {
                let reduction = boundary + index;
                fixture.a[reduction * k + index % k] = f32::from_bits(bits);
                fixture.b[reduction * n + index % n] =
                    f32::from_bits(exceptional[exceptional.len() - 1 - index]);
            }
        }
        fixture.output[0] = -0.0;
        fixture.output[1] = f32::from_bits(0x8000_0001);
        fixture.output[2] = f32::from_bits(0x7fc0_d128);
        fixture.delta = splitm64_delta(case.dims, &fixture.a, &fixture.b, 1.0);
        fixture
    }

    fn require_module_abi_resources(ctx: &GpuCtx) -> Result<(), String> {
        let artifact = ctx
            .kernels
            .triad_sm89_exact_f32_d128_artifact_identity()
            .ok_or_else(|| "exact-F32 d128 module has no artifact identity".to_string())?;
        let compiler = ctx
            .kernels
            .triad_sm89_exact_f32_d128_compiler_identity()
            .ok_or_else(|| "exact-F32 d128 module has no compiler identity".to_string())?;
        if artifact.module_kind != ModuleKind::TriadSm89ExactF32D128
            || compiler.target.as_str() != "sm_89"
            || ctx.kernels.artifact_set_identity().sm89_exact_f32_d128 != Some(artifact)
        {
            return Err("exact-F32 d128 module/compiler/artifact identity changed".into());
        }
        let exclusions = ctx.kernels.triad_sm89_exact_f32_d128_exclusions();
        if !exclusions.is_empty() {
            return Err(format!(
                "exact-F32 d128 ABI/resource inventory excluded symbols: {exclusions:?}"
            ));
        }
        for spec in SM89_EXACT_F32_D128_KERNEL_SPECS {
            let function = ctx
                .kernels
                .triad_sm89_exact_f32_d128_function(spec.symbol)
                .ok_or_else(|| format!("missing qualified function {}", spec.symbol))?;
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", spec.symbol))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", spec.symbol))?;
            let shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} shared bytes: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("{} occupancy: {error:?}", spec.symbol))?;
            if registers < 0
                || registers as u32 > spec.register_cap
                || local != 0
                || shared != 0
                || occupancy < spec.occupancy_gate
            {
                return Err(format!(
                    "{} resource stop: regs={registers}/{} local={local} static_shared={shared}/0 occupancy={occupancy}/{}",
                    spec.symbol, spec.register_cap, spec.occupancy_gate
                ));
            }
        }
        Ok(())
    }

    fn launch_raw_d128(
        ctx: &GpuCtx,
        case: Case,
        output: u64,
        a: u64,
        b: u64,
        alpha: f32,
    ) -> Result<(), String> {
        let function = ctx
            .kernels
            .triad_sm89_exact_f32_d128_function(case.symbol)
            .ok_or_else(|| format!("missing raw d128 function {}", case.symbol))?;
        let (m, k, n) = (case.dims.0 as i32, case.dims.1 as i32, case.dims.2 as i32);
        let mut builder = ctx.stream.launch_builder(function);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&alpha);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (256, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: case.dynamic_shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|error| format!("launch raw {}: {error:?}", case.symbol))
    }

    fn launch_prior_splitm64(
        ctx: &GpuCtx,
        case: Case,
        output: u64,
        scratch: u64,
        a: u64,
        b: u64,
        alpha: f32,
    ) -> Result<(), String> {
        let (m, k, n) = (case.dims.0 as i32, case.dims.1 as i32, case.dims.2 as i32);
        let m_chunk = 16_i32;
        let mut partial = ctx
            .stream
            .launch_builder(&ctx.kernels.gemm_bi_tn_splitm_partial_aligned);
        partial.arg(&scratch);
        partial.arg(&a);
        partial.arg(&b);
        partial.arg(&m);
        partial.arg(&k);
        partial.arg(&n);
        partial.arg(&m_chunk);
        unsafe {
            partial.launch(LaunchConfig {
                grid_dim: case.prior_grids[0],
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| format!("launch prior d128 SplitM64 partial: {error:?}"))?;

        let chunks = 64_i32;
        let mut reducer = ctx
            .stream
            .launch_builder(&ctx.kernels.gemm_bi_splitm_reduce);
        reducer.arg(&output);
        reducer.arg(&scratch);
        reducer.arg(&alpha);
        reducer.arg(&k);
        reducer.arg(&n);
        reducer.arg(&chunks);
        unsafe {
            reducer.launch(LaunchConfig {
                grid_dim: case.prior_grids[1],
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| format!("launch prior d128 SplitM64 reducer: {error:?}"))
    }

    fn compare_words(label: &str, actual: &[u32], expected: &[u32]) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!(
                "{label} word count changed: {} != {}",
                actual.len(),
                expected.len()
            ));
        }
        if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            return Err(format!(
                "{label} mismatch at {index}: {:08x} != {:08x}",
                actual[index], expected[index]
            ));
        }
        Ok(())
    }

    fn run_raw_guarded_probe(
        ctx: &GpuCtx,
        case: Case,
        fixture: &Fixture,
        alpha: f32,
        exceptional: bool,
    ) -> Result<Vec<u32>, String> {
        let mut output = RawGuardedBuffer::new(ctx, &fixture.output, "raw d128 output")?;
        let mut reference_output =
            RawGuardedBuffer::new(ctx, &fixture.output, "raw prior SplitM64 output")?;
        let a = RawGuardedBuffer::new(ctx, &fixture.a, "raw d128 A")?;
        let b = RawGuardedBuffer::new(ctx, &fixture.b, "raw d128 B")?;
        let scratch_values = vec![f32::from_bits(0x7fc0_bbbb); 64 * case.dims.1 * case.dims.2];
        let scratch = RawGuardedBuffer::new(ctx, &scratch_values, "raw prior SplitM64 scratch")?;
        let (output_ptr, reference_output_ptr, scratch_ptr, a_ptr, b_ptr) = (
            output.ptr(),
            reference_output.ptr(),
            scratch.ptr(),
            a.ptr(),
            b.ptr(),
        );
        let launch = || launch_raw_d128(ctx, case, output_ptr, a_ptr, b_ptr, alpha);
        let launch_reference = || {
            launch_prior_splitm64(
                ctx,
                case,
                reference_output_ptr,
                scratch_ptr,
                a_ptr,
                b_ptr,
                alpha,
            )
        };

        launch_reference()?;
        let reference_eager = reference_output.active_bits(ctx)?;
        if !exceptional {
            compare_words(
                "prior raw SplitM64 vs independent CPU oracle",
                &reference_eager,
                &fixture.expected(1),
            )?;
        }
        reference_output.reset(ctx)?;
        let reference_graph = unsafe { capture_into_graph(&ctx.stream, launch_reference) }?;
        reference_graph
            .launch()
            .map_err(|error| format!("prior SplitM64 graph launch: {error:?}"))?;
        compare_words(
            "prior raw SplitM64 eager/graph",
            &reference_output.active_bits(ctx)?,
            &reference_eager,
        )?;

        launch()?;
        compare_words(
            "raw d128 eager",
            &output.active_bits(ctx)?,
            &reference_eager,
        )?;
        a.unchanged(ctx)?;
        b.unchanged(ctx)?;

        output.reset(ctx)?;
        let graph = unsafe { capture_into_graph(&ctx.stream, launch) }?;
        graph
            .launch()
            .map_err(|error| format!("raw {} graph launch: {error:?}", case.symbol))?;
        compare_words(
            "raw d128 graph",
            &output.active_bits(ctx)?,
            &reference_eager,
        )?;
        a.unchanged(ctx)?;
        b.unchanged(ctx)?;
        let _ = scratch.active_bits(ctx)?;
        println!(
            "{}",
            json!({
                "schema":"MambaTriadSm89ExactF32D128RawGuardedV1",
                "route":format!("{:?}",case.route),
                "dims":case.dims,
                "symbol":case.symbol,
                "alpha_bits":alpha.to_bits(),
                "exceptional":exceptional,
                "guarded_allocations":5,
                "leading_guard_bytes":256,
                "trailing_guard_bytes":256,
                "active_alignment_bytes":256,
                "eager_graph_bits_exact":true,
                "inputs_immutable":true
            })
        );
        Ok(reference_eager)
    }

    fn require_direct_identity(
        ctx: &GpuCtx,
        launch: &QualifiedPhysicalLaunch<'_>,
        case: Case,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let node = evidence
            .nodes()
            .first()
            .ok_or_else(|| "d128 direct route has no node".to_string())?;
        let (_, k, n) = case.dims;
        if evidence.route_identity() != &ctx.gemm_route()
            || evidence.evidence_scope() != "eager_preflight_same_launcher"
            || !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || evidence.single_launch_symbol() != Some(case.symbol)
            || evidence.single_launch_tile() != Some(case.tile)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm89ExactF32D128)
            || node.kind != PhysicalLaunchKind::Gemm
            || node.symbol != case.symbol
            || node.module_kind != ModuleKind::TriadSm89ExactF32D128
            || node.logical_op != ResolvedGemmOp::Tn
            || node.logical_dtype != PolicyDtype::F32
            || node.execution_dtype != PolicyDtype::F32
            || node.shape != case.dims
            || node.strides != (k, n, n)
            || node.tile != Some(case.tile)
            || node.numeric_contract != Some(ResolvedNumericContract::ScalarFmaTnSplitMF64Reduce)
            || node.ownership != Some(ResolvedOutputOwnership::OneCtaPerOutputTile)
            || node.launch.grid_dim != (256, 1, 1)
            || node.launch.block_dim != (64, 1, 1)
            || node.launch.shared_mem_bytes != case.dynamic_shared_bytes
        {
            return Err(format!(
                "{:?} direct identity changed: {evidence:?}",
                case.route
            ));
        }
        Ok(())
    }

    fn require_prior_auto_identity(
        ctx: &GpuCtx,
        launch: &QualifiedPhysicalLaunch<'_>,
        case: Case,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let (_, k, n) = case.dims;
        let expected_tiles = [(128, 128), (1, 1)];
        let expected_numeric = [
            ResolvedNumericContract::ScalarFma,
            ResolvedNumericContract::ScalarFmaTnSplitMF64Reduce,
        ];
        let expected_ownership = [
            ResolvedOutputOwnership::OneCtaPerOutputTile,
            ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitMReduce,
        ];
        if evidence.route_identity() != &ctx.gemm_route()
            || evidence.evidence_scope() != "eager_preflight_same_launcher"
            || !evidence.eager_graph_equal()
            || evidence.launch_count() != 2
            || evidence.single_launch_symbol().is_some()
            || evidence.single_launch_tile().is_some()
            || evidence.nodes().iter().enumerate().any(|(index, node)| {
                node.kind != PhysicalLaunchKind::Gemm
                    || node.symbol != PRIOR_SYMBOLS[index]
                    || node.module_kind != ModuleKind::TriadScalar
                    || node.logical_op != ResolvedGemmOp::Tn
                    || node.logical_dtype != PolicyDtype::F32
                    || node.execution_dtype != PolicyDtype::F32
                    || node.shape != case.dims
                    || node.strides != (k, n, n)
                    || node.tile != Some(expected_tiles[index])
                    || node.numeric_contract != Some(expected_numeric[index])
                    || node.ownership != Some(expected_ownership[index])
                    || node.launch.grid_dim != case.prior_grids[index]
                    || node.launch.block_dim != (256, 1, 1)
                    || node.launch.shared_mem_bytes != 0
            })
        {
            return Err(format!(
                "{:?} prior actual AUTO identity changed: {:?}",
                case.route,
                evidence.nodes()
            ));
        }
        Ok(())
    }

    fn exercise_path(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        fixture: &Fixture,
        expected: &[u32],
        path: TimedPath,
        forced: bool,
        repeats: usize,
    ) -> Result<(), String> {
        let words = fixture.words();
        for repeat in 0..repeats {
            launch.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
            match (path, forced) {
                (TimedPath::Eager, true) => {
                    launch.measure_prevalidated_forced_eager_window_ms(ctx, 1)?;
                }
                (TimedPath::Eager, false) => {
                    launch.measure_eager_window_ms(ctx, 1)?;
                }
                (TimedPath::Graph, _) => {
                    launch.measure_graph_window_ms(ctx, 1)?;
                }
            }
            if launch.f32_output_bits(ctx)? != expected {
                return Err(format!("qualified d128 output differs at repeat {repeat}"));
            }
            if launch.f32_operand_bits(ctx)? != (words.1.clone(), words.2.clone()) {
                return Err(format!(
                    "qualified d128 launch modified A/B at repeat {repeat}"
                ));
            }
            let guards = launch.validate_red_zones(ctx)?;
            if guards.allocation_count() != 3 || guards.element_count() < 96 {
                return Err("qualified d128 output/A/B guard inventory changed".into());
            }
        }
        Ok(())
    }

    fn forced_request(case: Case, alpha: f32) -> PhysicalQualificationRequest {
        request(
            PhysicalQualificationRoute::Sm89ExactF32D128TnForced(case.route),
            case.dims,
            alpha,
            1.0,
            false,
        )
    }

    fn auto_request(case: Case) -> PhysicalQualificationRequest {
        request(
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFma),
            case.dims,
            1.0,
            1.0,
            false,
        )
    }

    fn reject_contract_neighbors(ctx: &GpuCtx, case: Case) -> Result<(), String> {
        for dims in [
            (case.dims.0 - 1, case.dims.1, case.dims.2),
            (case.dims.0 + 1, case.dims.1, case.dims.2),
            (case.dims.0, case.dims.1 - 1, case.dims.2),
            (case.dims.0, case.dims.1 + 1, case.dims.2),
            (case.dims.0, case.dims.1, case.dims.2 - 1),
            (case.dims.0, case.dims.1, case.dims.2 + 1),
            (0, case.dims.1, case.dims.2),
        ] {
            if qualify_physical_launch(
                ctx,
                request(
                    PhysicalQualificationRoute::Sm89ExactF32D128TnForced(case.route),
                    dims,
                    1.0,
                    1.0,
                    false,
                ),
            )
            .is_ok()
            {
                return Err(format!(
                    "forced {:?} admitted neighbor {dims:?}",
                    case.route
                ));
            }
        }
        for (beta, bias) in [(0.0, false), (-0.0, false), (1.0, true)] {
            if qualify_physical_launch(
                ctx,
                request(
                    PhysicalQualificationRoute::Sm89ExactF32D128TnForced(case.route),
                    case.dims,
                    1.0,
                    beta,
                    bias,
                ),
            )
            .is_ok()
            {
                return Err(format!(
                    "forced {:?} admitted beta={beta:?} bias={bias}",
                    case.route
                ));
            }
        }
        Ok(())
    }

    fn exercise_k0(ctx: &GpuCtx, case: Case) -> Result<(), String> {
        let dims = (0, case.dims.1, case.dims.2);
        let output = finite_full_mantissa_values(dims.1 * dims.2, 0xc000_0128);
        let expected = output
            .iter()
            .map(|value| 1.0_f32.mul_add(0.0, *value).to_bits())
            .collect::<Vec<_>>();
        let mut output = RawGuardedBuffer::new(ctx, &output, "d128 K0 output")?;
        let a = RawGuardedBuffer::new(ctx, &[], "d128 K0 A")?;
        let b = RawGuardedBuffer::new(ctx, &[], "d128 K0 B")?;
        let (output_ptr, k, n) = (output.ptr(), dims.1, dims.2);
        let launch = || {
            gpu_gemm_bi_backward_dw_grad(
                ctx,
                &GradSlice::from_raw(output_ptr, k * n),
                &b.gpu,
                &a.gpu,
                0,
                k,
                n,
            )
        };
        launch()?;
        compare_words("d128 K0 eager", &output.active_bits(ctx)?, &expected)?;
        a.unchanged(ctx)?;
        b.unchanged(ctx)?;
        output.reset(ctx)?;
        let graph = unsafe { capture_into_graph(&ctx.stream, launch) }?;
        graph
            .launch()
            .map_err(|error| format!("d128 K0 graph launch: {error:?}"))?;
        compare_words("d128 K0 graph", &output.active_bits(ctx)?, &expected)?;
        a.unchanged(ctx)?;
        b.unchanged(ctx)?;
        Ok(())
    }

    fn qualify_case(
        forced_ctx: &GpuCtx,
        auto_ctx: &GpuCtx,
        case: Case,
    ) -> Result<CaseReceipt, String> {
        let finite = finite_fixture(case, 1.0);
        let finite_expected = run_raw_guarded_probe(forced_ctx, case, &finite, 1.0, false)?;
        let mut forced = qualify_physical_launch(forced_ctx, forced_request(case, 1.0))?;
        require_direct_identity(forced_ctx, &forced, case)?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &finite,
            &finite_expected,
            TimedPath::Eager,
            true,
            2,
        )?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &finite,
            &finite_expected,
            TimedPath::Graph,
            true,
            2,
        )?;
        let exceptional = exceptional_fixture(case);
        drop(forced);
        let exceptional_expected =
            run_raw_guarded_probe(forced_ctx, case, &exceptional, 1.0, true)?;
        let mut forced = qualify_physical_launch(forced_ctx, forced_request(case, 1.0))?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &exceptional,
            &exceptional_expected,
            TimedPath::Eager,
            true,
            2,
        )?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &exceptional,
            &exceptional_expected,
            TimedPath::Graph,
            true,
            2,
        )?;
        drop(forced);

        let nonunit = finite_fixture(case, -0.75);
        let nonunit_expected = run_raw_guarded_probe(forced_ctx, case, &nonunit, -0.75, false)?;
        let mut forced = qualify_physical_launch(forced_ctx, forced_request(case, -0.75))?;
        require_direct_identity(forced_ctx, &forced, case)?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &nonunit,
            &nonunit_expected,
            TimedPath::Eager,
            true,
            2,
        )?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &nonunit,
            &nonunit_expected,
            TimedPath::Graph,
            true,
            2,
        )?;
        drop(forced);

        if qualify_physical_launch(
            auto_ctx,
            request(
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFma),
                case.dims,
                -0.75,
                1.0,
                false,
            ),
        )
        .is_ok()
        {
            return Err("public AUTO admitted non-unit TN alpha".into());
        }

        let mut auto = qualify_physical_launch(auto_ctx, auto_request(case))?;
        require_prior_auto_identity(auto_ctx, &auto, case)?;
        exercise_path(
            auto_ctx,
            &mut auto,
            &finite,
            &finite_expected,
            TimedPath::Eager,
            false,
            2,
        )?;
        exercise_path(
            auto_ctx,
            &mut auto,
            &finite,
            &finite_expected,
            TimedPath::Graph,
            false,
            2,
        )?;
        drop(auto);

        exercise_k0(auto_ctx, case)?;
        reject_contract_neighbors(forced_ctx, case)?;
        Ok(CaseReceipt {
            phases: REQUIRED_PHASES.to_vec(),
        })
    }

    fn quantile(values: &[f64], q: f64) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[((values.len() - 1) as f64 * q).ceil() as usize]
    }

    fn timed_window(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        fixture: &Fixture,
        path: TimedPath,
        forced: bool,
    ) -> Result<f64, String> {
        let words = fixture.words();
        launch.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
        let total = match (path, forced) {
            (TimedPath::Eager, true) => {
                launch.measure_prevalidated_forced_eager_window_ms(ctx, OPS_PER_WINDOW)?
            }
            (TimedPath::Eager, false) => launch.measure_eager_window_ms(ctx, OPS_PER_WINDOW)?,
            (TimedPath::Graph, _) => launch.measure_graph_window_ms(ctx, OPS_PER_WINDOW)?,
        };
        if !total.is_finite() || total <= 0.0 {
            return Err("invalid exact-F32 d128 timing window".into());
        }
        if launch.f32_output_bits(ctx)? != fixture.expected(OPS_PER_WINDOW) {
            return Err("timed d128 output differs from the independent repeated oracle".into());
        }
        if launch.f32_operand_bits(ctx)? != (words.1, words.2) {
            return Err("timed d128 launch modified A/B".into());
        }
        launch.validate_red_zones(ctx)?;
        Ok(total / OPS_PER_WINDOW as f64)
    }

    fn screen_batch(
        forced_ctx: &GpuCtx,
        auto_ctx: &GpuCtx,
        windows: usize,
    ) -> Result<bool, String> {
        let mut all_strict = true;
        for case in CASES {
            let fixture = finite_fixture(case, 1.0);
            let mut forced = qualify_physical_launch(forced_ctx, forced_request(case, 1.0))?;
            let mut auto = qualify_physical_launch(auto_ctx, auto_request(case))?;
            require_direct_identity(forced_ctx, &forced, case)?;
            require_prior_auto_identity(auto_ctx, &auto, case)?;
            for (path_name, path) in [("eager", TimedPath::Eager), ("graph", TimedPath::Graph)] {
                for (order_name, candidate_first) in [("ABBA", true), ("BAAB", false)] {
                    let mut ratios = Vec::with_capacity(windows);
                    let mut observations = Vec::with_capacity(windows);
                    for _ in 0..windows {
                        let raw = if candidate_first {
                            [
                                timed_window(forced_ctx, &mut forced, &fixture, path, true)?,
                                timed_window(auto_ctx, &mut auto, &fixture, path, false)?,
                                timed_window(auto_ctx, &mut auto, &fixture, path, false)?,
                                timed_window(forced_ctx, &mut forced, &fixture, path, true)?,
                            ]
                        } else {
                            [
                                timed_window(auto_ctx, &mut auto, &fixture, path, false)?,
                                timed_window(forced_ctx, &mut forced, &fixture, path, true)?,
                                timed_window(forced_ctx, &mut forced, &fixture, path, true)?,
                                timed_window(auto_ctx, &mut auto, &fixture, path, false)?,
                            ]
                        };
                        let candidate = if candidate_first {
                            raw[0] + raw[3]
                        } else {
                            raw[1] + raw[2]
                        };
                        let reference = if candidate_first {
                            raw[1] + raw[2]
                        } else {
                            raw[0] + raw[3]
                        };
                        observations.push(raw);
                        ratios.push(candidate / reference);
                    }
                    let p50 = quantile(&ratios, 0.50);
                    let p95 = quantile(&ratios, 0.95);
                    let strict = p50 < 0.99 && p95 < 0.99;
                    all_strict &= strict;
                    println!(
                        "{}",
                        json!({
                            "schema":"MambaTriadSm89ExactF32D128PairedScreenV1",
                            "route":format!("{:?}",case.route),
                            "dims":case.dims,
                            "path":path_name,
                            "order":order_name,
                            "windows":windows,
                            "ops_per_window":OPS_PER_WINDOW,
                            "observations_ms_per_op":observations,
                            "ratios":ratios,
                            "ratio_p50":p50,
                            "ratio_p95":p95,
                            "strict_099":strict,
                            "comparator":"literal_current_actual_AUTO_TnSplitM_16_64"
                        })
                    );
                }
            }
        }
        Ok(all_strict)
    }

    #[test]
    #[ignore = "requires an exclusive SM89 GPU; root is the sole GPU owner"]
    fn sm89_exact_f32_d128_pre_admission_forced_vs_prior_actual_auto() -> Result<(), String> {
        let mode = AdmissionMode::PreAdmission;
        mode.require()?;
        let forced_ctx = runtime()?;
        let auto_ctx = runtime()?;
        require_module_abi_resources(&forced_ctx)?;
        let mut cases = Vec::with_capacity(CASES.len());
        for case in CASES {
            cases.push(qualify_case(&forced_ctx, &auto_ctx, case)?);
        }
        let once3 = screen_batch(&forced_ctx, &auto_ctx, 3)?;
        let (screen, fast_context_labelled) = if once3 {
            if screen_batch(&forced_ctx, &auto_ctx, 7)? {
                println!(
                    "{}",
                    json!({
                        "schema":"MambaTriadSm89ExactF32D128FastContextV1",
                        "status":"separately_labelled_discovery_context_only",
                        "admission_comparator":"literal_current_actual_AUTO_TnSplitM_16_64"
                    })
                );
                (ScreenOutcome::Retained, true)
            } else {
                (ScreenOutcome::Once7Stop, false)
            }
        } else {
            (ScreenOutcome::Once3Stop, false)
        };
        let receipt = BatchReceipt {
            cases,
            screen,
            fast_context_labelled,
        };
        validate_batch_receipt(&receipt)?;
        println!(
            "{}",
            json!({
                "schema":"MambaTriadSm89ExactF32D128AdmissionReceiptV1",
                "mode":mode.label(),
                "screen":format!("{:?}",receipt.screen),
                "fast_context_labelled":receipt.fast_context_labelled,
                "auto_admitted":false
            })
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive SM89 GPU; root is the sole GPU owner"]
    fn sm89_exact_f32_d128_post_admission_actual_auto_exactness() -> Result<(), String> {
        let mode = AdmissionMode::PostAdmission;
        mode.require()?;
        let ctx = runtime()?;
        require_module_abi_resources(&ctx)?;
        for case in CASES {
            let fixture = finite_fixture(case, 1.0);
            let finite_expected = run_raw_guarded_probe(&ctx, case, &fixture, 1.0, false)?;
            let mut auto = qualify_physical_launch(&ctx, auto_request(case))?;
            require_direct_identity(&ctx, &auto, case)?;
            exercise_path(
                &ctx,
                &mut auto,
                &fixture,
                &finite_expected,
                TimedPath::Eager,
                false,
                2,
            )?;
            exercise_path(
                &ctx,
                &mut auto,
                &fixture,
                &finite_expected,
                TimedPath::Graph,
                false,
                2,
            )?;
            let exceptional = exceptional_fixture(case);
            drop(auto);
            let exceptional_expected = run_raw_guarded_probe(&ctx, case, &exceptional, 1.0, true)?;
            let mut auto = qualify_physical_launch(&ctx, auto_request(case))?;
            require_direct_identity(&ctx, &auto, case)?;
            exercise_path(
                &ctx,
                &mut auto,
                &exceptional,
                &exceptional_expected,
                TimedPath::Eager,
                false,
                2,
            )?;
            exercise_path(
                &ctx,
                &mut auto,
                &exceptional,
                &exceptional_expected,
                TimedPath::Graph,
                false,
                2,
            )?;
            println!(
                "{}",
                json!({
                    "schema":"MambaTriadSm89ExactF32D128PostAdmissionAutoV1",
                    "mode":mode.label(),
                    "route":format!("{:?}",case.route),
                    "dims":case.dims,
                    "one_node_no_split_scratch":true,
                    "eager_graph_identity_exact":true,
                    "repeated_bits_exact":true,
                    "guards_exact":true,
                    "inputs_immutable":true,
                    "auto_admitted":true
                })
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod native_contract_tests {
    use super::*;

    fn complete_case() -> CaseReceipt {
        CaseReceipt {
            phases: REQUIRED_PHASES.to_vec(),
        }
    }

    #[test]
    fn receipt_requires_exactly_two_complete_cases_and_ordered_screen() {
        let receipt = BatchReceipt {
            cases: vec![complete_case(), complete_case()],
            screen: ScreenOutcome::Retained,
            fast_context_labelled: true,
        };
        validate_batch_receipt(&receipt).unwrap();
        for case_index in 0..2 {
            for phase_index in 0..REQUIRED_PHASES.len() {
                let mut mutated = receipt.clone();
                mutated.cases[case_index].phases.remove(phase_index);
                assert!(validate_batch_receipt(&mutated).is_err());
            }
        }
        for (screen, labelled, accepted) in [
            (ScreenOutcome::Once3Stop, false, true),
            (ScreenOutcome::Once3Stop, true, false),
            (ScreenOutcome::Once7Stop, false, true),
            (ScreenOutcome::Once7Stop, true, false),
            (ScreenOutcome::Retained, false, false),
            (ScreenOutcome::Retained, true, true),
        ] {
            assert_eq!(
                validate_batch_receipt(&BatchReceipt {
                    cases: vec![complete_case(), complete_case()],
                    screen,
                    fast_context_labelled: labelled,
                })
                .is_ok(),
                accepted
            );
        }
    }

    #[test]
    fn independent_oracle_uses_positive_zero_chunks_and_ascending_f64_fold() {
        let dims = (1_024, 1, 1);
        let mut a = vec![0.0_f32; 1_024];
        let mut b = vec![0.0_f32; 1_024];
        for reduction in 0..1_024 {
            let mantissa = 1.0 + ((reduction * 13 % 127) as f32) * f32::EPSILON;
            a[reduction] = if reduction & 1 == 0 {
                mantissa
            } else {
                -mantissa
            };
            b[reduction] = 2.0_f32.powi(((reduction * 17 + 3) % 23) as i32 - 11);
        }
        let delta = splitm64_delta(dims, &a, &b, 0.75);
        let split = apply_delta_bits(&[-0.0], &delta, 1)[0];
        let mut direct = 0.0_f32;
        for reduction in 0..1_024 {
            direct = a[reduction].mul_add(b[reduction], direct);
        }
        assert_ne!(split, (-0.0_f32 + 0.75 * direct).to_bits());
    }

    #[test]
    fn independent_oracle_retains_exceptional_and_repeated_add_semantics() {
        let dims = (1_024, 1, 1);
        let mut a = vec![0.0_f32; 1_024];
        let mut b = vec![0.0_f32; 1_024];
        a[15] = f32::from_bits(0x7fc0_0128);
        b[15] = 1.0;
        assert!(splitm64_delta(dims, &a, &b, 1.0)[0].is_nan());

        a.fill(0.0);
        b.fill(0.0);
        a[16] = -0.0;
        b[16] = 1.0;
        let delta = splitm64_delta(dims, &a, &b, 1.0);
        assert_eq!(apply_delta_bits(&[-0.0], &delta, 1), [0x0000_0000]);
        assert_eq!(
            apply_delta_bits(
                &[f32::from_bits(0x0000_0001)],
                &[f32::from_bits(0x0000_0001)],
                2
            ),
            [0x0000_0003]
        );
    }
}
