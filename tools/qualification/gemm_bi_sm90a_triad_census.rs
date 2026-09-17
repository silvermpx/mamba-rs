//! Real-Hopper qualification for the deterministic SM90a triad.
//!
//! This is deliberately separate from the Inference family's Hopper check. It
//! exercises only the triad's exact-sm_90a module through its public tensor-map
//! preparation and forced-launch APIs. Automatic routing remains disabled until
//! this test has passed on the target Hopper/toolchain pair.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM90A_STAGES, SM90A_TILE, Sm90aForcedRoute, Sm90aLaunchOperands, Sm90aMapRequest,
    Sm90aNumericContract, Sm90aOp, Sm90aPreparedTensorMaps, Sm90aShape, Sm90aWarpgroupSchedule,
    launch_sm90a_wgmma_forced, prepare_sm90a_tensor_maps, resolve_sm90a_forced,
    validate_sm90a_graph_replay,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::ModuleKind;

const EDGE_DIMENSIONS: &[usize] = &[1, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129];
const ASYMMETRIC: (usize, usize, usize) = (65, 127, 129);
const DETERMINISM_RERUNS: usize = 100;

fn round_up_8(value: usize) -> usize {
    value.div_ceil(8) * 8
}

fn quantize(dtype: WeightDtype, value: f32) -> f32 {
    match dtype {
        WeightDtype::Bf16 => half::bf16::from_f32(value).to_f32(),
        WeightDtype::F16 => half::f16::from_f32(value).to_f32(),
        WeightDtype::F32 => value,
    }
}

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let base = ((state & 0xFFFFFF) as f32 / 16_777_216.0) * 0.4 - 0.2;
            match index % 8 {
                0 => base * 8.0,
                1 => -base * 4.0,
                _ => base,
            }
        })
        .collect()
}

enum CensusOutput {
    Typed(DtypedBuf),
    F32(GpuBuffer),
}

impl CensusOutput {
    fn new(ctx: &GpuCtx, op: Sm90aOp, dtype: WeightDtype, len: usize) -> Self {
        if op == Sm90aOp::Tn {
            Self::F32(GpuBuffer::zeros(&ctx.stream, len).expect("f32 output"))
        } else {
            Self::Typed(DtypedBuf::zeros(&ctx.stream, len, dtype).expect("typed output"))
        }
    }

    fn ptr(&self) -> u64 {
        match self {
            Self::Typed(buffer) => buffer.cached_ptr(),
            Self::F32(buffer) => buffer.cached_ptr(),
        }
    }

    fn reset(&mut self, ctx: &GpuCtx, values: &[f32]) {
        match self {
            Self::Typed(buffer) => buffer
                .upload_f32(&ctx.stream, values)
                .expect("reset typed output"),
            Self::F32(buffer) => buffer
                .upload(&ctx.stream, values)
                .expect("reset f32 output"),
        }
    }

    fn download(&self, ctx: &GpuCtx, len: usize) -> Vec<f32> {
        match self {
            Self::Typed(buffer) => {
                let mut values = vec![0.0; len];
                buffer
                    .download_f32(&ctx.stream, &mut values)
                    .expect("download typed output");
                values
            }
            Self::F32(buffer) => buffer.to_cpu(&ctx.stream).expect("download f32 output"),
        }
    }
}

struct CensusCase {
    op: Sm90aOp,
    dtype: WeightDtype,
    shape: Sm90aShape,
    a: DtypedBuf,
    b: DtypedBuf,
    bias: GpuBuffer,
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    bias_host: Vec<f32>,
    initial: Vec<f32>,
    expected: Vec<f32>,
    output_rows: usize,
    output_cols: usize,
    nn_full_epilogue: bool,
}

impl CensusCase {
    fn new(
        ctx: &GpuCtx,
        op: Sm90aOp,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        nn_full_epilogue: bool,
    ) -> Self {
        let (m, k, n) = dims;
        let a_width = if op == Sm90aOp::Nt { n } else { k };
        let output_rows = if op == Sm90aOp::Tn { k } else { m };
        let output_cols = if op == Sm90aOp::Nt { k } else { n };
        let shape = Sm90aShape {
            m,
            k,
            n,
            lda: round_up_8(a_width),
            ldb: round_up_8(n),
            ldc: output_cols + 3,
        };
        let b_rows = if op == Sm90aOp::Tn { m } else { k };
        let mut a_host = vec![13.0; m * shape.lda];
        let mut b_host = vec![-11.0; b_rows * shape.ldb];
        let a_values = synth(m * a_width, 0xA11CE ^ m as u64 ^ (k as u64) << 8);
        let b_values = synth(b_rows * n, 0xB0B ^ n as u64 ^ (m as u64) << 8);
        for row in 0..m {
            for col in 0..a_width {
                a_host[row * shape.lda + col] = quantize(dtype, a_values[row * a_width + col]);
            }
        }
        for row in 0..b_rows {
            for col in 0..n {
                b_host[row * shape.ldb + col] = quantize(dtype, b_values[row * n + col]);
            }
        }
        let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), dtype).expect("A");
        a.upload_f32(&ctx.stream, &a_host).expect("upload A");
        let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), dtype).expect("B");
        b.upload_f32(&ctx.stream, &b_host).expect("upload B");
        let bias_host = synth(output_cols, 0xB1A5 ^ output_cols as u64);
        let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias");
        let output_dtype = if op == Sm90aOp::Tn {
            WeightDtype::F32
        } else {
            dtype
        };
        let initial = synth(output_rows * shape.ldc, 0xC001 ^ output_rows as u64)
            .into_iter()
            .map(|value| quantize(output_dtype, value))
            .collect::<Vec<_>>();
        let mut case = Self {
            op,
            dtype,
            shape,
            a,
            b,
            bias,
            a_host,
            b_host,
            bias_host,
            initial,
            expected: Vec::new(),
            output_rows,
            output_cols,
            nn_full_epilogue,
        };
        case.expected = case.reference();
        case
    }

    fn request(&self) -> Sm90aMapRequest {
        Sm90aMapRequest {
            op: self.op,
            dtype: self.dtype,
            a_ptr: self.a.cached_ptr(),
            b_ptr: self.b.cached_ptr(),
            shape: self.shape,
        }
    }

    fn operands(&self, output_ptr: u64) -> Sm90aLaunchOperands {
        let simple_nn = self.op == Sm90aOp::Nn && !self.nn_full_epilogue;
        Sm90aLaunchOperands {
            output_ptr,
            bias_ptr: if self.op == Sm90aOp::Nn && !simple_nn {
                self.bias.cached_ptr()
            } else {
                0
            },
            alpha: if simple_nn { 1.0 } else { 0.75 },
            beta: match self.op {
                Sm90aOp::Nn if simple_nn => 0.0,
                Sm90aOp::Nn => -0.25,
                Sm90aOp::Tn => 1.0,
                Sm90aOp::Nt => 0.0,
            },
        }
    }

    fn reference(&self) -> Vec<f32> {
        let mut output = self.initial.clone();
        for row in 0..self.output_rows {
            for col in 0..self.output_cols {
                let reduction = match self.op {
                    Sm90aOp::Nn => self.shape.k,
                    Sm90aOp::Tn => self.shape.m,
                    Sm90aOp::Nt => self.shape.n,
                };
                let mut sum = 0.0f32;
                for inner in 0..reduction {
                    let (a, b) = match self.op {
                        Sm90aOp::Nn => (
                            self.a_host[row * self.shape.lda + inner],
                            self.b_host[inner * self.shape.ldb + col],
                        ),
                        Sm90aOp::Tn => (
                            self.a_host[inner * self.shape.lda + row],
                            self.b_host[inner * self.shape.ldb + col],
                        ),
                        Sm90aOp::Nt => (
                            self.a_host[row * self.shape.lda + inner],
                            self.b_host[col * self.shape.ldb + inner],
                        ),
                    };
                    sum = a.mul_add(b, sum);
                }
                let index = row * self.shape.ldc + col;
                let value = match self.op {
                    Sm90aOp::Nn => {
                        let operands = self.operands(1);
                        let product = operands.alpha * sum;
                        let product = if operands.bias_ptr == 0 {
                            product
                        } else {
                            product + self.bias_host[col]
                        };
                        if operands.beta == 0.0 {
                            product
                        } else {
                            operands.beta.mul_add(self.initial[index], product)
                        }
                    }
                    Sm90aOp::Tn => 0.75f32.mul_add(sum, self.initial[index]),
                    Sm90aOp::Nt => 0.75 * sum,
                };
                output[index] = if self.op == Sm90aOp::Tn {
                    value
                } else {
                    quantize(self.dtype, value)
                };
            }
        }
        output
    }

    fn assert_correct(&self, actual: &[f32], schedule: Sm90aWarpgroupSchedule) {
        let (absolute_tolerance, relative_tolerance) = match (self.op, self.dtype) {
            (Sm90aOp::Tn, _) => (5.0e-4, 5.0e-4),
            (_, WeightDtype::Bf16) => (2.0e-2, 2.0e-2),
            (_, WeightDtype::F16) => (3.0e-3, 3.0e-3),
            (_, WeightDtype::F32) => unreachable!(),
        };
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for row in 0..self.output_rows {
            for col in 0..self.output_cols {
                let index = row * self.shape.ldc + col;
                let got = actual[index];
                let want = self.expected[index];
                let tolerance = absolute_tolerance + relative_tolerance * want.abs();
                assert!(
                    got.is_finite() && (got - want).abs() <= tolerance,
                    "{:?}/{:?}/{schedule:?} M{} K{} N{} at ({row},{col}): got {got}, expected {want}",
                    self.op,
                    self.dtype,
                    self.shape.m,
                    self.shape.k,
                    self.shape.n,
                );
                dot += f64::from(got) * f64::from(want);
                actual_norm += f64::from(got) * f64::from(got);
                expected_norm += f64::from(want) * f64::from(want);
            }
            for col in self.output_cols..self.shape.ldc {
                let index = row * self.shape.ldc + col;
                assert_eq!(
                    actual[index].to_bits(),
                    self.initial[index].to_bits(),
                    "{:?}/{:?}/{schedule:?} overwrote ldc padding at ({row},{col})",
                    self.op,
                    self.dtype,
                );
            }
        }
        if actual_norm > 0.0 && expected_norm > 0.0 {
            let cosine = dot / (actual_norm * expected_norm).sqrt();
            assert!(
                cosine > 0.9999,
                "{:?}/{:?}/{schedule:?} cosine {cosine} at M{} K{} N{}",
                self.op,
                self.dtype,
                self.shape.m,
                self.shape.k,
                self.shape.n,
            );
        }
    }
}

fn forced_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    schedule: Sm90aWarpgroupSchedule,
) -> Sm90aForcedRoute {
    resolve_sm90a_forced(
        (9, 0),
        ctx.kernels.has_sm90a_wgmma(),
        case.op,
        case.dtype,
        schedule,
        case.shape,
    )
    .expect("valid forced route")
    .expect("exact-sm_90a route loaded")
}

fn run_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    maps: &Sm90aPreparedTensorMaps,
    schedule: Sm90aWarpgroupSchedule,
    reruns: usize,
) {
    let route = forced_route(ctx, case, schedule);
    let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.output_rows * case.shape.ldc);
    let mut baseline_bits = Vec::new();
    let mut baseline_identity = None;
    for repetition in 0..=reruns {
        output.reset(ctx, &case.initial);
        let identity = launch_sm90a_wgmma_forced(
            &ctx.stream,
            &ctx.kernels,
            route,
            maps,
            case.operands(output.ptr()),
        )
        .expect("forced SM90a launch");
        let actual = output.download(ctx, case.initial.len());
        let bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(identity.numeric_contract, Sm90aNumericContract::Wgmma);
        assert_eq!(identity.op, case.op);
        assert_eq!(identity.dtype, case.dtype);
        assert_eq!(identity.schedule, schedule);
        assert_eq!(identity.shape, case.shape);
        assert_eq!(identity.tile, SM90A_TILE);
        assert_eq!(identity.stages, SM90A_STAGES);
        assert_eq!(identity.cluster, (1, 1, 1));
        assert_eq!(identity.symbol, route.symbol());
        assert_eq!(identity.module_kind, ModuleKind::TriadSm90a);
        assert_eq!(identity.exact_target, "sm_90a");
        assert_eq!(identity.compiler.target.as_str(), "sm_90a");
        assert_eq!(identity.device.compute_capability, (9, 0));
        assert_eq!(identity.device.target.as_str(), "sm_90a");
        assert_eq!(identity.tensor_maps_digest, maps.identity_digest());
        assert_eq!(identity.tuning_revision, 0);
        if repetition == 0 {
            case.assert_correct(&actual, schedule);
            baseline_bits = bits;
            baseline_identity = Some(identity);
        } else {
            assert_eq!(
                bits, baseline_bits,
                "{:?}/{:?}/{schedule:?} changed bits on repetition {repetition} at M{} K{} N{}",
                case.op, case.dtype, case.shape.m, case.shape.k, case.shape.n,
            );
            baseline_identity
                .expect("baseline identity")
                .ensure_current(identity, "SM90a check")
                .expect("route identity remains current");
        }
    }
}

fn run_case(
    ctx: &GpuCtx,
    op: Sm90aOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
    reruns: usize,
    nn_full_epilogue: bool,
) {
    let case = CensusCase::new(ctx, op, dtype, dims, nn_full_epilogue);
    let maps = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, case.request())
        .expect("prepare tensor maps");
    let cached = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, case.request())
        .expect("reuse tensor maps");
    assert_eq!(maps.identity_digest(), cached.identity_digest());
    for schedule in [Sm90aWarpgroupSchedule::Wg1, Sm90aWarpgroupSchedule::Wg2] {
        run_route(ctx, &case, &maps, schedule, reruns);
    }
}

fn run_graph_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    maps: &Sm90aPreparedTensorMaps,
    schedule: Sm90aWarpgroupSchedule,
) {
    let route = forced_route(ctx, case, schedule);
    let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let operands = case.operands(output.ptr());
    output.reset(ctx, &case.initial);
    let eager_identity =
        launch_sm90a_wgmma_forced(&ctx.stream, &ctx.kernels, route, maps, operands)
            .expect("eager graph baseline");
    let eager = output.download(ctx, case.initial.len());
    case.assert_correct(&eager, schedule);
    let eager_bits = eager
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();

    let alternate_a =
        DtypedBuf::zeros(&ctx.stream, case.a_host.len(), case.dtype).expect("alternate graph A");
    alternate_a
        .upload_f32(&ctx.stream, &case.a_host)
        .expect("upload alternate graph A");
    let mut miss_request = case.request();
    miss_request.a_ptr = alternate_a.cached_ptr();
    output.reset(ctx, &case.initial);
    let mut captured_identity = None;
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            let cached = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, case.request())?;
            if cached.identity_digest() != maps.identity_digest() {
                return Err("captured tensor-map cache hit changed identity".into());
            }
            let miss = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, miss_request)
                .expect_err("capture-time tensor-map cache miss must be rejected");
            if !miss.contains("cache miss during graph capture") {
                return Err(format!("unexpected capture-time cache miss error: {miss}"));
            }
            captured_identity = Some(launch_sm90a_wgmma_forced(
                &ctx.stream,
                &ctx.kernels,
                route,
                maps,
                operands,
            )?);
            Ok(())
        })
    }
    .expect("capture forced SM90a route");
    let captured_identity = captured_identity.expect("captured route identity");
    eager_identity
        .ensure_current(captured_identity, "SM90a graph check")
        .expect("eager and captured route identity agree");

    let mut changed_scalars = operands;
    changed_scalars.alpha = f32::from_bits(operands.alpha.to_bits() ^ 1);
    let error = validate_sm90a_graph_replay(
        &ctx.stream,
        &ctx.kernels,
        route,
        maps,
        changed_scalars,
        captured_identity,
    )
    .expect_err("changed graph scalar must invalidate replay");
    assert!(error.contains("route changed since capture"), "{error}");

    for replay in 0..3 {
        output.reset(ctx, &case.initial);
        validate_sm90a_graph_replay(
            &ctx.stream,
            &ctx.kernels,
            route,
            maps,
            operands,
            captured_identity,
        )
        .expect("validate graph replay resources");
        graph.launch().expect("launch captured SM90a graph");
        let actual = output.download(ctx, case.initial.len());
        let bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            bits, eager_bits,
            "{:?}/{:?}/{schedule:?} graph replay {replay} changed output bits",
            case.op, case.dtype,
        );
    }
}

fn run_graph_case(ctx: &GpuCtx, op: Sm90aOp, dtype: WeightDtype) {
    let case = CensusCase::new(ctx, op, dtype, ASYMMETRIC, true);
    let maps = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, case.request())
        .expect("prepare graph tensor maps");
    for schedule in [Sm90aWarpgroupSchedule::Wg1, Sm90aWarpgroupSchedule::Wg2] {
        run_graph_route(ctx, &case, &maps, schedule);
    }
}

fn assert_semantic_rejections(ctx: &GpuCtx) {
    for op in [Sm90aOp::Tn, Sm90aOp::Nt] {
        let case = CensusCase::new(ctx, op, WeightDtype::Bf16, ASYMMETRIC, true);
        let maps = prepare_sm90a_tensor_maps(&ctx.stream, &ctx.kernels, case.request())
            .expect("prepare rejection maps");
        let route = forced_route(ctx, &case, Sm90aWarpgroupSchedule::Wg1);
        let output = CensusOutput::new(ctx, case.op, case.dtype, case.output_rows * case.shape.ldc);
        let operands = case.operands(output.ptr());
        let reject = |operands, contract: &str| {
            let error =
                launch_sm90a_wgmma_forced(&ctx.stream, &ctx.kernels, route, &maps, operands)
                    .expect_err("invalid epilogue contract must fail before launch");
            assert!(error.contains(contract), "{error}");
        };
        if op == Sm90aOp::Tn {
            reject(
                Sm90aLaunchOperands {
                    bias_ptr: case.bias.cached_ptr(),
                    ..operands
                },
                "SM90a TN",
            );
            reject(
                Sm90aLaunchOperands {
                    beta: 0.0,
                    ..operands
                },
                "SM90a TN",
            );
        } else {
            reject(
                Sm90aLaunchOperands {
                    bias_ptr: case.bias.cached_ptr(),
                    ..operands
                },
                "SM90a NT",
            );
            reject(
                Sm90aLaunchOperands {
                    beta: 1.0,
                    ..operands
                },
                "SM90a NT",
            );
        }
    }
}

#[test]
#[ignore = "needs a real exact-sm_90a Hopper device"]
fn qualifies_all_forced_sm90a_triad_routes() {
    let device = match GpuDevice::new(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping the SM90a triad check without a CUDA device: {error}");
            return;
        }
    };
    if device.compute_capability != (9, 0) {
        eprintln!(
            "skipping the SM90a triad check on compute capability {}.{}",
            device.compute_capability.0, device.compute_capability.1
        );
        return;
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(
        ctx.kernels.has_sm90a_wgmma(),
        "exact Hopper detected but the SM90a triad module failed qualification"
    );

    for op in [Sm90aOp::Nn, Sm90aOp::Tn, Sm90aOp::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for &dimension in EDGE_DIMENSIONS {
                let nn_full_epilogue = op != Sm90aOp::Nn || !dimension.is_multiple_of(2);
                run_case(
                    &ctx,
                    op,
                    dtype,
                    (dimension, dimension, dimension),
                    0,
                    nn_full_epilogue,
                );
            }
            run_case(&ctx, op, dtype, ASYMMETRIC, DETERMINISM_RERUNS, true);
            run_graph_case(&ctx, op, dtype);
        }
    }
    assert_semantic_rejections(&ctx);
}
