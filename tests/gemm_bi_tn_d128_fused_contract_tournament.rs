const PRIMARY: &str = "gemm_bi_tn_d128_fused_m8n32_bk16_exp";
const SENSITIVITY: &str = "gemm_bi_tn_d128_fused_m16n32_bk16_exp";
const SOURCE: &str = include_str!("gemm_bi_tn_d128_fused_contract_tournament.cu");

const M: usize = 1_024;
const K: usize = 128;
const N: usize = 512;
const CHUNK: usize = 16;
const CHUNKS: usize = 64;

#[cfg(feature = "cuda")]
mod common;
#[path = "support/triad_tn_d128_direct_source.rs"]
mod triad_tn_d128_direct_source;

fn short_pair_quantiles(raw: &[[f64; 4]], candidate_first: bool) -> Result<[f64; 2], String> {
    if raw.len() != 7 || raw.iter().flatten().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err("short screen requires seven valid four-observation brackets".into());
    }
    let mut ratios: Vec<_> = raw
        .iter()
        .map(|v| {
            let outside = v[0] + v[3];
            let inside = v[1] + v[2];
            if candidate_first {
                outside / inside
            } else {
                inside / outside
            }
        })
        .collect();
    ratios.sort_by(f64::total_cmp);
    Ok([ratios[3], ratios[6]])
}

#[test]
fn short_pair_rejects_invalid_times_and_respects_arm_order() {
    let abba = [[4.0, 10.0, 10.0, 6.0]; 7];
    assert_eq!(short_pair_quantiles(&abba, true).unwrap(), [0.5, 0.5]);
    assert_eq!(short_pair_quantiles(&abba, false).unwrap(), [2.0, 2.0]);
    assert!(short_pair_quantiles(&abba[..6], true).is_err());
    let mut invalid = abba;
    invalid[4][2] = f64::NAN;
    assert!(short_pair_quantiles(&invalid, true).is_err());
    invalid[4][2] = 0.0;
    assert!(short_pair_quantiles(&invalid, true).is_err());
}

fn parameters(symbol: &str) -> &str {
    let (_, signature) = SOURCE
        .split_once(&format!("void {symbol}("))
        .unwrap_or_else(|| panic!("missing {symbol}"));
    signature
        .split_once(") {")
        .map(|(parameters, _)| parameters)
        .unwrap_or_else(|| panic!("missing parameter list for {symbol}"))
}

fn splitm64_oracle(a: &[f32], b: &[f32], initial: &[f32], alpha: f32) -> Vec<u32> {
    assert_eq!(a.len(), M * K);
    assert_eq!(b.len(), M * N);
    assert_eq!(initial.len(), K * N);
    let mut output = initial.to_vec();
    for row in 0..K {
        for column in 0..N {
            let mut sum = 0.0_f64;
            for chunk in 0..CHUNKS {
                let mut partial = 0.0_f32;
                for offset in 0..CHUNK {
                    let reduction = chunk * CHUNK + offset;
                    partial = a[reduction * K + row].mul_add(b[reduction * N + column], partial);
                }
                if chunk == 0 {
                    sum = partial as f64;
                } else {
                    sum += partial as f64;
                }
            }
            output[row * N + column] += ((alpha as f64) * sum) as f32;
        }
    }
    output.into_iter().map(f32::to_bits).collect()
}

fn first_bit_mismatch(actual: &[u32], expected: &[u32], allow_nan_payload: bool) -> Option<usize> {
    actual.iter().zip(expected).position(|(actual, expected)| {
        actual != expected
            && !(allow_nan_payload
                && f32::from_bits(*actual).is_nan()
                && f32::from_bits(*expected).is_nan())
    })
}

#[test]
fn candidates_are_test_only_and_keep_the_bounded_abi() {
    let production = [
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs"),
    ];
    for symbol in [PRIMARY, SENSITIVITY] {
        assert_eq!(SOURCE.matches(&format!("void {symbol}(")).count(), 1);
        assert!(parameters(symbol).matches(',').count() < 7);
        assert!(production.iter().all(|source| !source.contains(symbol)));
    }
    assert!(!SOURCE.contains("atomic"));
}

#[test]
fn candidates_pin_the_exact_cell_and_one_owner_grids() {
    for required in [
        "M == 1024 && K == 128 && N == 512",
        "constexpr int CHUNK = 16",
        "constexpr int CHUNKS = 64",
        "blockIdx.x * BM",
        "blockIdx.y * 32",
        "gemm_bi_tn_d128_fused_contract_body<8>",
        "gemm_bi_tn_d128_fused_contract_body<16>",
        "__launch_bounds__(128",
    ] {
        assert!(SOURCE.contains(required), "missing {required}");
    }
}

#[test]
fn candidates_spell_the_current_splitm64_rounding_contract() {
    for required in [
        "partial[owned] = 0.0f",
        "partial[owned] = __fmaf_rn(",
        "sums[owned] = (double)partial[owned]",
        "sum = __dadd_rn(sum, (double)partial[owned])",
        "double scaled = __dmul_rn((double)alpha, sum)",
        "output += __double2float_rn(scaled)",
    ] {
        assert!(SOURCE.contains(required), "missing {required}");
    }
    assert_eq!(SOURCE.matches("partial[owned] = __fmaf_rn(").count(), 1);
}

#[test]
fn oracle_distinguishes_splitm64_from_a_single_chain() {
    let mut a = vec![0.0_f32; M * K];
    let mut b = vec![0.0_f32; M * N];
    let initial = vec![-0.0_f32; K * N];
    for reduction in 0..M {
        let exponent = ((reduction * 17 + 3) % 23) as i32 - 11;
        let mantissa = 1.0 + ((reduction * 13 % 127) as f32) * f32::EPSILON;
        a[reduction * K] = if reduction & 1 == 0 {
            mantissa
        } else {
            -mantissa
        };
        b[reduction * N] = 2.0_f32.powi(exponent);
    }
    let split = splitm64_oracle(&a, &b, &initial, 0.75)[0];
    let mut direct = 0.0_f32;
    for reduction in 0..M {
        direct = a[reduction * K].mul_add(b[reduction * N], direct);
    }
    let direct = (-0.0_f32 + 0.75 * direct).to_bits();
    assert_ne!(split, direct, "corpus no longer discriminates the contract");
}

#[test]
fn oracle_preserves_exceptional_float_behavior() {
    let mut a = vec![0.0_f32; M * K];
    let mut b = vec![0.0_f32; M * N];
    let initial = vec![-0.0_f32; K * N];
    a[0] = f32::INFINITY;
    b[0] = 0.0;
    assert!(f32::from_bits(splitm64_oracle(&a, &b, &initial, 1.0)[0]).is_nan());

    a[0] = -0.0;
    b[0] = 1.0;
    let bits = splitm64_oracle(&a, &b, &initial, 1.0)[0];
    assert_eq!(bits, 0x0000_0000, "final f32 addition semantics changed");
}

#[test]
fn production_nan_classification_does_not_weaken_candidate_bit_exactness() {
    let production = [0x7fff_ffff, (-0.0_f32).to_bits()];
    let cpu = [0x7fc0_0128, (-0.0_f32).to_bits()];
    assert_eq!(first_bit_mismatch(&production, &cpu, true), None);
    assert_eq!(first_bit_mismatch(&production, &cpu, false), Some(0));

    let payload_mutation = [0x7fc0_0128, (-0.0_f32).to_bits()];
    assert_eq!(
        first_bit_mismatch(&payload_mutation, &production, false),
        Some(0)
    );
    let zero_sign_mutation = [0x7fff_ffff, 0.0_f32.to_bits()];
    assert_eq!(
        first_bit_mismatch(&zero_sign_mutation, &production, false),
        Some(1)
    );
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use std::sync::Arc;
    use std::time::Instant;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
    };
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dw_grad;
    use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

    use super::{K, M, N, PRIMARY, SENSITIVITY, SOURCE, first_bit_mismatch, splitm64_oracle};

    const GUARD: usize = 64;
    const GUARD_BITS: u32 = 0x7fc0_d128;
    const REPEATS: usize = 3;
    const WINDOWS: usize = 101;
    const WINDOW_US: f64 = 10_000.0;
    const MAX_REGISTERS: usize = 96;
    const MIN_OCCUPANCY: usize = 2;
    const MIN_SPEEDUP: f64 = 1.05;

    #[derive(Clone, Copy, Debug)]
    enum Arm {
        Production,
        M8N32,
        M16N32,
    }

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
        symbol: &'static str,
        static_shared: usize,
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _module: Arc<CudaModule>,
        primary: Kernel,
        sensitivity: Kernel,
        dims: (usize, usize, usize),
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        offset: usize,
        len: usize,
    }

    impl GuardedBuffer {
        fn new(stream: &Arc<CudaStream>, active: Vec<f32>) -> Result<Self, String> {
            let len = active.len();
            let mut expected = vec![f32::from_bits(GUARD_BITS); GUARD + len + GUARD];
            expected[GUARD..GUARD + len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                offset: GUARD,
                len,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.offset)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.expected)
        }

        fn bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.buffer.to_cpu(stream)?;
            for (index, value) in actual[..self.offset]
                .iter()
                .chain(&actual[self.offset + self.len..])
                .enumerate()
            {
                if value.to_bits() != GUARD_BITS {
                    return Err(format!("{label} red zone changed at {index}"));
                }
            }
            Ok(actual[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.buffer.to_cpu(stream)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        production_a: GpuBuffer,
        production_b: GpuBuffer,
        production_output: GpuBuffer,
        production_initial: Vec<f32>,
        primary_output: GuardedBuffer,
        sensitivity_output: GuardedBuffer,
        oracle: Vec<u32>,
        alpha: f32,
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
                    let exponent = (119 + ((state >> 29) as u32 % 12)) << 23;
                    let mantissa = (state as u32 & 0x007f_ffff) | 1;
                    f32::from_bits(sign | exponent | mantissa)
                }
            })
            .collect()
    }

    fn runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0)
            || identity.multiprocessor_count != 170
            || device.nvrtc_target() != "compute_120"
        {
            return Err(format!(
                "TN d128 tournament requires CC12.0/170 SM compute_120, found {:?}/{} SM/{}",
                identity.compute_capability,
                identity.multiprocessor_count,
                device.nvrtc_target()
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
        if compiler.target.as_str() != "compute_120"
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(
                "TN d128 tournament requires the qualified scalar CUDA 13.2 artifact domain".into(),
            );
        }
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, options)
            .map_err(|error| format!("compile TN d128 candidates: {error:?}"))?;
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load TN d128 candidates: {error:?}"))?;
        let load = |symbol, grid, static_shared| -> Result<Kernel, String> {
            Ok(Kernel {
                function: module
                    .load_function(symbol)
                    .map_err(|error| format!("load {symbol}: {error:?}"))?,
                config: LaunchConfig {
                    grid_dim: grid,
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                symbol,
                static_shared,
            })
        };
        let primary = load(PRIMARY, (16, 16, 1), 2_560)?;
        let sensitivity = load(SENSITIVITY, (8, 16, 1), 3_072)?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            primary,
            sensitivity,
            dims: (M, K, N),
        })
    }

    fn fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let mut a = values(M * K, 0xa128_0001);
        let mut b = values(M * N, 0xb128_0002);
        a[0] = f32::from_bits(0x7fc0_0128);
        a[1] = -0.0;
        b[1] = 1.0;
        let initial = values(K * N, 0xc128_0003);
        let alpha = 1.0_f32;
        let oracle = splitm64_oracle(&a, &b, &initial, alpha);
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a.clone())?,
            b: GuardedBuffer::new(&runtime.ctx.stream, b.clone())?,
            production_a: GpuBuffer::from_cpu(&runtime.ctx.stream, &a)?,
            production_b: GpuBuffer::from_cpu(&runtime.ctx.stream, &b)?,
            production_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &initial)?,
            production_initial: initial.clone(),
            primary_output: GuardedBuffer::new(&runtime.ctx.stream, initial.clone())?,
            sensitivity_output: GuardedBuffer::new(&runtime.ctx.stream, initial)?,
            oracle,
            alpha,
        })
    }

    fn kernel(runtime: &Runtime, arm: Arm) -> &Kernel {
        match arm {
            Arm::M8N32 => &runtime.primary,
            Arm::M16N32 => &runtime.sensitivity,
            Arm::Production => panic!("production has no experimental kernel"),
        }
    }

    fn launch_candidate(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let kernel = kernel(runtime, arm);
        let destination = match arm {
            Arm::M8N32 => fixture.primary_output.ptr(&runtime.ctx.stream),
            Arm::M16N32 => fixture.sensitivity_output.ptr(&runtime.ctx.stream),
            Arm::Production => unreachable!(),
        };
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
        let m = runtime.dims.0 as i32;
        let k = runtime.dims.1 as i32;
        let n = runtime.dims.2 as i32;
        let mut builder = runtime.ctx.stream.launch_builder(&kernel.function);
        builder.arg(&destination);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&fixture.alpha);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", kernel.symbol))
    }

    fn launch_production(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let (m, k, n) = runtime.dims;
        gpu_gemm_bi_backward_dw_grad(
            &runtime.ctx,
            &GradSlice::from_raw(fixture.production_output.cached_ptr(), k * n),
            &fixture.production_b,
            &fixture.production_a,
            m,
            k,
            n,
        )
    }

    fn capture(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<CudaGraph, String> {
        if matches!(arm, Arm::Production) {
            launch_production(runtime, fixture)?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize production graph warmup: {error:?}"))?;
        }
        unsafe {
            capture_into_graph(&runtime.ctx.stream, || match arm {
                Arm::Production => launch_production(runtime, fixture),
                _ => launch_candidate(runtime, fixture, arm),
            })
        }
    }

    fn reset_output(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<(), String> {
        match arm {
            Arm::Production => fixture
                .production_output
                .upload(&runtime.ctx.stream, &fixture.production_initial),
            Arm::M8N32 => fixture.primary_output.reset(&runtime.ctx.stream),
            Arm::M16N32 => fixture.sensitivity_output.reset(&runtime.ctx.stream),
        }
    }

    fn output_bits(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<Vec<u32>, String> {
        match arm {
            Arm::Production => Ok(fixture
                .production_output
                .to_cpu(&runtime.ctx.stream)?
                .into_iter()
                .map(f32::to_bits)
                .collect()),
            Arm::M8N32 => fixture
                .primary_output
                .bits(&runtime.ctx.stream, "M8N32 output"),
            Arm::M16N32 => fixture
                .sensitivity_output
                .bits(&runtime.ctx.stream, "M16N32 output"),
        }
    }

    fn correctness(runtime: &Runtime, fixture: &mut Fixture) -> Result<(), String> {
        let mut production_oracle = None;
        for arm in [Arm::Production, Arm::M8N32, Arm::M16N32] {
            reset_output(runtime, fixture, arm)?;
            let graph = capture(runtime, fixture, arm)?;
            let mut previous = None;
            for repeat in 0..REPEATS {
                reset_output(runtime, fixture, arm)?;
                match arm {
                    Arm::Production => launch_production(runtime, fixture)?,
                    _ => launch_candidate(runtime, fixture, arm)?,
                }
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("sync eager: {error:?}"))?;
                let eager = output_bits(runtime, fixture, arm)?;
                let expected = production_oracle.as_ref().unwrap_or(&fixture.oracle);
                let mismatch = first_bit_mismatch(&eager, expected, matches!(arm, Arm::Production));
                if let Some(index) = mismatch {
                    return Err(format!(
                        "{arm:?} differs from SplitM64 oracle at {index}: actual=0x{:08x} expected=0x{:08x}",
                        eager[index], expected[index]
                    ));
                }
                reset_output(runtime, fixture, arm)?;
                graph
                    .launch()
                    .map_err(|error| format!("launch {arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("sync graph: {error:?}"))?;
                let graph_bits = output_bits(runtime, fixture, arm)?;
                if graph_bits != eager {
                    return Err(format!(
                        "{arm:?} graph differs from eager at repeat {repeat}"
                    ));
                }
                if previous.as_ref().is_some_and(|bits| bits != &eager) {
                    return Err(format!("{arm:?} repeated eager bits changed"));
                }
                previous.get_or_insert(eager);
            }
            if matches!(arm, Arm::Production) {
                production_oracle = previous.clone();
            }
        }
        fixture.a.unchanged(&runtime.ctx.stream, "A and guards")?;
        fixture.b.unchanged(&runtime.ctx.stream, "B and guards")?;
        Ok(())
    }

    fn resources(runtime: &Runtime) -> Result<(), String> {
        for kernel in [&runtime.primary, &runtime.sensitivity] {
            let local = kernel
                .function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", kernel.symbol))?
                as usize;
            let registers = kernel
                .function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", kernel.symbol))?
                as usize;
            let shared = kernel
                .function
                .shared_size_bytes()
                .map_err(|error| format!("{} shared bytes: {error:?}", kernel.symbol))?
                as usize;
            let occupancy = kernel
                .function
                .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
                .map_err(|error| format!("{} occupancy: {error:?}", kernel.symbol))?
                as usize;
            if local != 0
                || registers > MAX_REGISTERS
                || shared != kernel.static_shared
                || occupancy < MIN_OCCUPANCY
            {
                return Err(format!(
                    "{} resource contract failed: local={local} regs={registers} shared={shared} occupancy={occupancy}",
                    kernel.symbol
                ));
            }
            eprintln!(
                "RESOURCE symbol={} local={} regs={} static_shared={} occupancy={}",
                kernel.symbol, local, registers, shared, occupancy
            );
        }
        Ok(())
    }

    fn production_identity(runtime: &Runtime) -> Result<(), String> {
        let (m, k, n) = runtime.dims;
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Tn,
            (m, k, n),
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let qualified = qualify_physical_launch(&runtime.ctx, request)?;
        let nodes = qualified.evidence().nodes();
        if nodes.len() != 2
            || nodes[0].symbol != "gemm_bi_tn_splitm_partial_aligned"
            || nodes[0].launch.grid_dim != ((k.div_ceil(128) * n.div_ceil(128)) as u32, 1, 64)
            || nodes[0].launch.block_dim != (256, 1, 1)
            || nodes[0].launch.shared_mem_bytes != 0
            || nodes[1].symbol != "gemm_bi_splitm_reduce"
            || nodes[1].launch.grid_dim != ((k * n).div_ceil(256) as u32, 1, 1)
            || nodes[1].launch.block_dim != (256, 1, 1)
            || nodes[1].launch.shared_mem_bytes != 0
            || nodes[0].launch.arguments_digest == [0; 32]
            || nodes[1].launch.arguments_digest == [0; 32]
            || nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest
            || !qualified.evidence().eager_graph_equal()
        {
            return Err(format!("production SplitM64 identity changed: {nodes:?}"));
        }
        qualified.validate_red_zones(&runtime.ctx)?;
        Ok(())
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    fn time_graph(runtime: &Runtime, graph: &CudaGraph, iterations: usize) -> Result<f64, String> {
        let start = Instant::now();
        for _ in 0..iterations {
            graph
                .launch()
                .map_err(|error| format!("timed graph launch: {error:?}"))?;
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("timed sync: {error:?}"))?;
        Ok(start.elapsed().as_secs_f64() * 1.0e6 / iterations as f64)
    }

    fn benchmark(runtime: &Runtime, fixture: &mut Fixture) -> Result<(), String> {
        let arms = [Arm::Production, Arm::M8N32, Arm::M16N32];
        let mut graphs = Vec::new();
        for arm in arms {
            reset_output(runtime, fixture, arm)?;
            graphs.push(capture(runtime, fixture, arm)?);
        }
        let mut iterations = [1_usize; 3];
        for index in 0..3 {
            let probe = time_graph(runtime, &graphs[index], 16)?;
            iterations[index] = ((WINDOW_US / probe).ceil() as usize).max(1);
        }
        let mut samples = [
            Vec::with_capacity(WINDOWS),
            Vec::with_capacity(WINDOWS),
            Vec::with_capacity(WINDOWS),
        ];
        for window in 0..WINDOWS {
            let order = if window & 1 == 0 {
                [0, 1, 2, 2, 1, 0]
            } else {
                [2, 1, 0, 0, 1, 2]
            };
            let mut paired = [[0.0_f64; 2]; 3];
            let mut occurrence = [0_usize; 3];
            for index in order {
                paired[index][occurrence[index]] =
                    time_graph(runtime, &graphs[index], iterations[index])?;
                occurrence[index] += 1;
            }
            for index in 0..3 {
                samples[index].push((paired[index][0] + paired[index][1]) * 0.5);
            }
        }
        let production = median(&mut samples[0]);
        let primary = median(&mut samples[1]);
        let sensitivity = median(&mut samples[2]);
        eprintln!(
            "RESULT production_p50_us={production:.9} m8n32_p50_us={primary:.9} m16n32_p50_us={sensitivity:.9} production_over_m8n32={:.6} m16n32_over_m8n32={:.6}",
            production / primary,
            sensitivity / primary
        );
        if !(primary.is_finite()
            && production.is_finite()
            && sensitivity.is_finite()
            && production / primary >= MIN_SPEEDUP
            && sensitivity / primary >= 1.0)
        {
            return Err(format!(
                "hard performance gate failed: production={production} M8N32={primary} M16N32={sensitivity}"
            ));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive CC 12.0/170-SM CUDA 13.2 GPU"]
    fn qualify_fused_tn_d128_exactness_and_resources() -> Result<(), String> {
        let runtime = runtime()?;
        let mut fixture = fixture(&runtime)?;
        production_identity(&runtime)?;
        resources(&runtime)?;
        correctness(&runtime, &mut fixture)
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC 12.0/170-SM GPU"]
    fn benchmark_fused_tn_d128_after_qualification() -> Result<(), String> {
        let runtime = runtime()?;
        let mut fixture = fixture(&runtime)?;
        benchmark(&runtime, &mut fixture)
    }

    // Bounded Ada discovery. The existing SM120 tournament is unchanged.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum AdaDirectVariant {
        Base,
        MoreWaves,
        MoreWarps,
        D128Out,
        InFoldPipeline,
        OutFoldPipeline,
    }

    fn ada_direct_runtime(variant: AdaDirectVariant) -> Result<Runtime, String> {
        use super::triad_tn_d128_direct_source as source;
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err("Ada direct TN discovery requires CC8.9/142 SMs".into());
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(false);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        if compiler.nvrtc_version != (13, 2) || compiler.target.as_str() != "sm_89" {
            return Err(format!("unexpected Ada compiler {compiler:?}"));
        }
        let original = include_str!("gemm_bi_scalar_tn_underfill_direct_experiment.cu");
        let cuda = match variant {
            AdaDirectVariant::Base => source::compose_source(original)?,
            AdaDirectVariant::MoreWaves => source::compose_wave_source(original)?,
            AdaDirectVariant::MoreWarps => source::compose_warp_source(original)?,
            AdaDirectVariant::D128Out => source::compose_out_source(original)?,
            AdaDirectVariant::InFoldPipeline => source::compose_foldpipe_source(original)?,
            AdaDirectVariant::OutFoldPipeline => source::compose_out_foldpipe_source(original)?,
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            cuda,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec![
                    "--fmad=true".into(),
                    "--extra-device-vectorization".into(),
                    "-DNDEBUG".into(),
                ],
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|e| format!("Ada TN direct compile: {e:?}"))?;
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|e| format!("Ada TN module: {e:?}"))?;
        let load = |symbol, dynamic, grid, threads| -> Result<Kernel, String> {
            let function = module
                .load_function(symbol)
                .map_err(|e| format!("load {symbol}: {e:?}"))?;
            let registers = function.num_regs().map_err(|e| format!("regs: {e:?}"))?;
            let local = function
                .local_size_bytes()
                .map_err(|e| format!("local: {e:?}"))?;
            let shared = function
                .shared_size_bytes()
                .map_err(|e| format!("shared: {e:?}"))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, dynamic, None)
                .map_err(|e| format!("occupancy: {e:?}"))?;
            println!(
                "{}",
                serde_json::json!({"schema":"AdaTnDirectResourceV1", "symbol":symbol,
                "registers":registers,"local_bytes":local,"static_shared":shared,"dynamic_shared":dynamic,
                "threads":threads,"grid":[grid,1,1],"occupancy":occupancy})
            );
            if local != 0 || shared != 0 || occupancy == 0 {
                return Err(format!("unviable Ada direct resources for {symbol}"));
            }
            Ok(Kernel {
                function,
                config: LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: u32::try_from(dynamic)
                        .map_err(|_| "dynamic shared size exceeds u32")?,
                },
                symbol,
                static_shared: 0,
            })
        };
        let (primary, sensitivity) = match variant {
            AdaDirectVariant::InFoldPipeline => (
                load(source::IN_FOLDPIPE_SYMBOL, 4096, 256, 64)?,
                load(source::M16N16_SYMBOL, 4096, 256, 64)?,
            ),
            AdaDirectVariant::OutFoldPipeline => (
                load(source::OUT_FOLDPIPE_SYMBOL, 3072, 256, 64)?,
                load(source::OUT_M8N16_SYMBOL, 3072, 256, 64)?,
            ),
            AdaDirectVariant::D128Out => (
                load(source::OUT_M8N16_SYMBOL, 3072, 256, 64)?,
                load(source::OUT_M16N16_SYMBOL, 4096, 128, 64)?,
            ),
            AdaDirectVariant::MoreWaves => (
                load(source::M8N16_SYMBOL, 3072, 512, 64)?,
                load(source::M16N16_SYMBOL, 4096, 256, 64)?,
            ),
            AdaDirectVariant::MoreWarps => (
                load(source::M16N16_T128_SYMBOL, 4096, 256, 128)?,
                load(source::M16N16_SYMBOL, 4096, 256, 64)?,
            ),
            AdaDirectVariant::Base => (
                load(source::M16N16_SYMBOL, 4096, 256, 64)?,
                load(source::M8N32_SYMBOL, 5120, 256, 64)?,
            ),
        };
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            primary,
            sensitivity,
            dims: if matches!(
                variant,
                AdaDirectVariant::D128Out | AdaDirectVariant::OutFoldPipeline
            ) {
                (1024, 256, 128)
            } else {
                (M, K, N)
            },
        })
    }

    fn ada_finite_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k, n) = runtime.dims;
        let a = values(m * k, 0xa128_5101);
        let b = values(m * n, 0xb128_5102);
        let initial = values(k * n, 0xc128_5103);
        // Normative bit oracle is the real public SplitM64 route below.
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a.clone())?,
            b: GuardedBuffer::new(&runtime.ctx.stream, b.clone())?,
            production_a: GpuBuffer::from_cpu(&runtime.ctx.stream, &a)?,
            production_b: GpuBuffer::from_cpu(&runtime.ctx.stream, &b)?,
            production_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &initial)?,
            production_initial: initial.clone(),
            primary_output: GuardedBuffer::new(&runtime.ctx.stream, initial.clone())?,
            sensitivity_output: GuardedBuffer::new(&runtime.ctx.stream, initial)?,
            oracle: Vec::new(),
            alpha: 1.0,
        })
    }

    fn ada_fast_tn(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        use cudarc::cublas::{result, sys};
        use std::ffi::c_void;
        // dW^T = dY^T * X: row-major [K,N] becomes column-major [N,K].
        let one = 1.0_f32;
        let (m, k, n) = runtime.dims;
        let dtype = mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::F32.cuda_data_type();
        unsafe {
            result::gemm_ex(
                *runtime.ctx.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                k as i32,
                m as i32,
                (&one as *const f32).cast::<c_void>(),
                fixture.production_b.cached_ptr() as *const c_void,
                dtype,
                n as i32,
                fixture.production_a.cached_ptr() as *const c_void,
                dtype,
                k as i32,
                (&one as *const f32).cast::<c_void>(),
                fixture.production_output.cached_ptr() as *mut c_void,
                dtype,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| format!("explicit TN Fast: {e:?}"))
    }

    fn ada_observe(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        fast: bool,
        graph: &CudaGraph,
        graph_path: bool,
        expected: Option<&[u32]>,
    ) -> Result<(f64, Vec<u32>), String> {
        reset_output(runtime, fixture, arm)?;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = runtime
            .ctx
            .stream
            .record_event(flags)
            .map_err(|e| format!("start: {e:?}"))?;
        if graph_path {
            graph.launch().map_err(|e| format!("graph: {e:?}"))?;
        } else if fast {
            ada_fast_tn(runtime, fixture)?;
        } else if matches!(arm, Arm::Production) {
            launch_production(runtime, fixture)?;
        } else {
            launch_candidate(runtime, fixture, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(flags)
            .map_err(|e| format!("end: {e:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("elapsed: {e:?}"))?,
        ) * 1000.0;
        let actual = output_bits(runtime, fixture, arm)?;
        if !us.is_finite() || us <= 0.0 || expected.is_some_and(|want| want != actual) {
            return Err(format!(
                "Ada TN {arm:?} fast={fast} invalid timing or changed bits"
            ));
        }
        fixture.a.unchanged(&runtime.ctx.stream, "direct A")?;
        fixture.b.unchanged(&runtime.ctx.stream, "direct B")?;
        for (buffer, source) in [
            (&fixture.production_a, &fixture.a),
            (&fixture.production_b, &fixture.b),
        ] {
            let actual = buffer.to_cpu(&runtime.ctx.stream)?;
            if actual
                .iter()
                .zip(&source.expected[source.offset..source.offset + source.len])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("public TN input mutated".into());
            }
        }
        Ok((us, actual))
    }

    fn ada_candidate_graph(graph: &CudaGraph, kernel: &Kernel) -> Result<(), String> {
        use cudarc::driver::sys;
        use std::ffi::CStr;
        let mut count = 0;
        let mut node = std::ptr::null_mut();
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe {
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count != 1
            {
                return Err("direct TN graph must have one node".into());
            }
            if sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || sys::cuGraphKernelNodeGetParams_v2(node, &mut params)
                    != sys::CUresult::CUDA_SUCCESS
            {
                return Err("query direct TN graph".into());
            }
            let mut name = std::ptr::null();
            if sys::cuFuncGetName(&mut name, params.func) != sys::CUresult::CUDA_SUCCESS
                || name.is_null()
                || CStr::from_ptr(name).to_bytes() != kernel.symbol.as_bytes()
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != kernel.config.grid_dim
                || (params.blockDimX, params.blockDimY, params.blockDimZ) != kernel.config.block_dim
                || params.sharedMemBytes != kernel.config.shared_mem_bytes
            {
                return Err("wrong direct TN graph".into());
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; two new direct-fold TN candidates, short once7"]
    fn ada_d128_direct_fold_two_arm_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::Base)
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; NCU-guided M8N16 vs retained M16N16 and Fast, once7"]
    fn ada_d128_direct_fold_more_waves_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::MoreWaves)
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; double warps without extra tile traffic, once7"]
    fn ada_d128_direct_fold_more_warps_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::MoreWarps)
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; reuse direct fold on d128-out, two tiles vs AUTO/Fast"]
    fn ada_d128_out_direct_fold_two_arm_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::D128Out)
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; delayed FP64 fold vs retained d128-in and Fast"]
    fn ada_d128_in_fold_pipeline_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::InFoldPipeline)
    }

    #[test]
    #[ignore = "quiet Ada13.2 only; delayed FP64 fold vs retained d128-out and Fast"]
    fn ada_d128_out_fold_pipeline_once7() -> Result<(), String> {
        ada_direct_fold_screen(AdaDirectVariant::OutFoldPipeline)
    }

    fn ada_fold_pipeline_corner_bits(runtime: &Runtime) -> Result<(), String> {
        let mut fixture = ada_finite_fixture(runtime)?;
        let (_, k, n) = runtime.dims;
        // Separate output rows exercise payload/sign/subnormal behavior. The
        // timing fixture remains finite and is created after this probe.
        for (row, word) in [
            0x0000_0000u32,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7fa1_2345,
            0xffc5_4321,
            0x7f7f_ffff,
        ]
        .into_iter()
        .enumerate()
        {
            fixture.a.expected[fixture.a.offset + row] = f32::from_bits(word);
        }
        fixture.a.expected[fixture.a.offset + 15 * k + 15] = 2.0f32.powi(25);
        fixture.a.expected[fixture.a.offset + 16 * k + 15] = -2.0f32.powi(25);
        fixture.b.expected[fixture.b.offset + 15 * n] = 1.0;
        fixture.b.expected[fixture.b.offset + 16 * n] = 1.0;
        fixture.a.reset(&runtime.ctx.stream)?;
        fixture.b.reset(&runtime.ctx.stream)?;
        fixture.production_a.upload(
            &runtime.ctx.stream,
            &fixture.a.expected[fixture.a.offset..fixture.a.offset + fixture.a.len],
        )?;
        fixture.production_b.upload(
            &runtime.ctx.stream,
            &fixture.b.expected[fixture.b.offset..fixture.b.offset + fixture.b.len],
        )?;
        let auto_graph = capture(runtime, &fixture, Arm::Production)?;
        let (_, expected) = ada_observe(
            runtime,
            &mut fixture,
            Arm::Production,
            false,
            &auto_graph,
            false,
            None,
        )?;
        for arm in [Arm::Production, Arm::M8N32, Arm::M16N32] {
            let graph = capture(runtime, &fixture, arm)?;
            if !matches!(arm, Arm::Production) {
                ada_candidate_graph(&graph, kernel(runtime, arm))?;
            }
            for graph_path in [false, true] {
                for _ in 0..2 {
                    ada_observe(
                        runtime,
                        &mut fixture,
                        arm,
                        false,
                        &graph,
                        graph_path,
                        Some(&expected),
                    )?;
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "schema":"AdaTnFoldPipelineCornerBitsV1", "shape":runtime.dims,
                "words":expected.len(), "eager_repeats":2, "graph_repeats":2,
                "oracle":"actual_auto_raw_bits", "candidate":runtime.primary.symbol,
                "retained":runtime.sensitivity.symbol,
                "fixture":"signed_zero_subnormal_inf_nan_payload_and_chunk_boundary"
            })
        );
        Ok(())
    }

    fn ada_direct_fold_screen(variant: AdaDirectVariant) -> Result<(), String> {
        let more_waves = matches!(
            variant,
            AdaDirectVariant::MoreWaves
                | AdaDirectVariant::MoreWarps
                | AdaDirectVariant::InFoldPipeline
                | AdaDirectVariant::OutFoldPipeline
        );
        assert!(!cfg!(debug_assertions), "use release");
        let quiet = super::common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("ada-direct-tn/pre")?;
        let runtime = ada_direct_runtime(variant)?;
        production_identity(&runtime)?; // Isolated holder is dropped on return.
        if matches!(
            variant,
            AdaDirectVariant::InFoldPipeline | AdaDirectVariant::OutFoldPipeline
        ) {
            ada_fold_pipeline_corner_bits(&runtime)?;
        }
        let mut fixture = ada_finite_fixture(&runtime)?;
        let auto_graph = capture(&runtime, &fixture, Arm::Production)?;
        let (_, exact) = ada_observe(
            &runtime,
            &mut fixture,
            Arm::Production,
            false,
            &auto_graph,
            false,
            None,
        )?;
        ada_fast_tn(&runtime, &fixture)?;
        let fast_graph =
            unsafe { capture_into_graph(&runtime.ctx.stream, || ada_fast_tn(&runtime, &fixture)) }?;
        let (_, fast_bits) = ada_observe(
            &runtime,
            &mut fixture,
            Arm::Production,
            true,
            &fast_graph,
            false,
            None,
        )?;
        if fast_bits.iter().any(|x| !f32::from_bits(*x).is_finite())
            || fast_bits.iter().all(|x| *x == 0)
        {
            return Err("invalid Fast initial output".into());
        }
        for graph_path in [false, true] {
            for _ in 0..2 {
                ada_observe(
                    &runtime,
                    &mut fixture,
                    Arm::Production,
                    false,
                    &auto_graph,
                    graph_path,
                    Some(&exact),
                )?;
                ada_observe(
                    &runtime,
                    &mut fixture,
                    Arm::Production,
                    true,
                    &fast_graph,
                    graph_path,
                    Some(&fast_bits),
                )?;
            }
        }
        quiet.require_cohort("ada-direct-tn/timed")?;
        let reference_arm = if more_waves {
            Arm::M16N32
        } else {
            Arm::Production
        };
        let reference_name = if variant == AdaDirectVariant::OutFoldPipeline {
            "retained_m8n16"
        } else if more_waves {
            "retained_m16n16"
        } else {
            "actual_auto"
        };
        let reference_graph = if more_waves {
            let graph = capture(&runtime, &fixture, reference_arm)?;
            ada_candidate_graph(&graph, kernel(&runtime, reference_arm))?;
            graph
        } else {
            auto_graph
        };
        let arms: &[Arm] = if more_waves {
            &[Arm::M8N32]
        } else {
            &[Arm::M8N32, Arm::M16N32]
        };
        for &arm in arms {
            // Existing arm slots are reused, but real symbol/config are emitted.
            let candidate = kernel(&runtime, arm);
            let graph = capture(&runtime, &fixture, arm)?;
            ada_candidate_graph(&graph, candidate)?;
            for graph_path in [false, true] {
                for _ in 0..2 {
                    ada_observe(
                        &runtime,
                        &mut fixture,
                        arm,
                        false,
                        &graph,
                        graph_path,
                        Some(&exact),
                    )?;
                }
            }
            for fast in [false, true] {
                let mut strata = Vec::new();
                for graph_path in [false, true] {
                    for candidate_first in [true, false] {
                        let sequence = if candidate_first {
                            [true, false, false, true]
                        } else {
                            [false, true, true, false]
                        };
                        let mut raw = Vec::new();
                        for bracket in 0..11 {
                            // Four warmup brackets, seven measured.
                            let mut times = [0.0; 4];
                            for (slot, is_candidate) in sequence.into_iter().enumerate() {
                                let (a, f, g, bits) = if is_candidate {
                                    (arm, false, &graph, &exact)
                                } else if fast {
                                    (Arm::Production, true, &fast_graph, &fast_bits)
                                } else {
                                    (reference_arm, false, &reference_graph, &exact)
                                };
                                times[slot] = ada_observe(
                                    &runtime,
                                    &mut fixture,
                                    a,
                                    f,
                                    g,
                                    graph_path,
                                    Some(bits),
                                )?
                                .0;
                            }
                            if bracket >= 4 {
                                raw.push(times);
                            }
                        }
                        let pair = super::short_pair_quantiles(&raw, candidate_first)?;
                        strata.push(pair);
                        println!(
                            "{}",
                            serde_json::json!({"schema":"AdaTnDirectScreenV1","shape":runtime.dims,
                        "candidate":candidate.symbol,"comparator":if fast{"cublas_fast_tf32"}else{reference_name},
                        "path":if graph_path{"graph"}else{"eager"},"order":if candidate_first{"ABBA"}else{"BAAB"},
                        "candidate_first":candidate_first,"windows":7,"warmup_brackets":4,"raw_observations_us":raw,
                        "ratio_p50":pair[0],"ratio_p95":pair[1],"ratio_direction":"candidate_over_reference",
                        "reseed":"same_nonzero_C_before_event","bit_oracle":if fast{"Fast self bits"}else{"public SplitM64"}})
                        );
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"schema":"AdaTnDirectDecisionV1","shape":runtime.dims,"candidate":candidate.symbol,
                    "comparator":if fast{"cublas_fast_tf32"}else{reference_name},"strata":strata,
                    "retain":strata.iter().flatten().all(|x|*x<0.99),"promotion":false})
                );
            }
        }
        quiet.verify_post_cohort("ada-direct-tn/post")?;
        Ok(()) // Valid speed losses are recorded; they do not discard later arms.
    }
}
