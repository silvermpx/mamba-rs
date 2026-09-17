//! Real-Blackwell qualification for the deterministic SM100 triad.
//!
//! The check keeps automatic routing disabled. It exercises every forced
//! physical route, descriptor cache reuse, prepared graph capture, replay
//! identity, output padding, and bitwise repeatability on CC 10.0 or 10.3.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM100_AUTO_CELLS_CC100, SM100_AUTO_CELLS_CC103, SM100_KERNEL_SPECS, SM100_TENSOR_MAP_REVISION,
    SM100_TUNING_REVISION, Sm100ForcedRoute, Sm100LaunchOperands, Sm100MapRequest,
    Sm100NumericContract, Sm100Op, Sm100PhysicalRoute, Sm100PreparedLaunch,
    Sm100PreparedTensorMaps, Sm100Schedule, Sm100Shape, Sm100Stages, Sm100Tile,
    launch_sm100_tcgen_prepared, prepare_sm100_tcgen_forced, prepare_sm100_tensor_maps,
    resolve_sm100_forced, validate_sm100_graph_replay, validate_sm100_map_request,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{FramedSha256, ModuleKind, Sha256Digest};
use std::collections::BTreeSet;
use std::sync::Arc;

const EDGE_DIMENSIONS: &[usize] = &[1, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129];
const ASYMMETRIC: (usize, usize, usize) = (65, 127, 129);
const DETERMINISM_RERUNS: usize = 100;
const GRAPH_REPLAYS: usize = 100;

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

fn output_digest(values: &[f32]) -> Sha256Digest {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    FramedSha256::bytes(&bytes)
}

fn independent_axis_shapes() -> BTreeSet<(usize, usize, usize)> {
    let mut shapes = BTreeSet::new();
    for &edge in EDGE_DIMENSIONS {
        shapes.insert((edge, ASYMMETRIC.1, ASYMMETRIC.2));
        shapes.insert((ASYMMETRIC.0, edge, ASYMMETRIC.2));
        shapes.insert((ASYMMETRIC.0, ASYMMETRIC.1, edge));
    }
    shapes
}

enum CensusOutput {
    Typed(DtypedBuf),
    F32(GpuBuffer),
}

impl CensusOutput {
    fn new(ctx: &GpuCtx, op: Sm100Op, dtype: WeightDtype, len: usize) -> Self {
        if op == Sm100Op::Tn {
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
        self.reset_on(&ctx.stream, values);
    }

    fn reset_on(&mut self, stream: &Arc<cudarc::driver::CudaStream>, values: &[f32]) {
        match self {
            Self::Typed(buffer) => buffer
                .upload_f32(stream, values)
                .expect("reset typed output"),
            Self::F32(buffer) => buffer.upload(stream, values).expect("reset f32 output"),
        }
    }

    fn download(&self, ctx: &GpuCtx, len: usize) -> Vec<f32> {
        self.download_on(&ctx.stream, len)
    }

    fn download_on(&self, stream: &Arc<cudarc::driver::CudaStream>, len: usize) -> Vec<f32> {
        match self {
            Self::Typed(buffer) => {
                let mut values = vec![0.0; len];
                buffer
                    .download_f32(stream, &mut values)
                    .expect("download typed output");
                values
            }
            Self::F32(buffer) => buffer.to_cpu(stream).expect("download f32 output"),
        }
    }
}

struct CensusCase {
    op: Sm100Op,
    dtype: WeightDtype,
    shape: Sm100Shape,
    a: DtypedBuf,
    b: DtypedBuf,
    a_offset: usize,
    b_offset: usize,
    a_alternate_offset: usize,
    b_alternate_offset: usize,
    bias: GpuBuffer,
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    bias_host: Vec<f32>,
    initial: Vec<f32>,
    expected: Vec<f32>,
    output_rows: usize,
    output_columns: usize,
}

impl CensusCase {
    fn new(ctx: &GpuCtx, op: Sm100Op, dtype: WeightDtype, dims: (usize, usize, usize)) -> Self {
        Self::with_origins(ctx, op, dtype, dims, (0, 0), (0, 0))
    }

    fn with_origins(
        ctx: &GpuCtx,
        op: Sm100Op,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        a_origin: (usize, usize),
        b_origin: (usize, usize),
    ) -> Self {
        let (m, k, n) = dims;
        let a_width = if op == Sm100Op::Nt { n } else { k };
        let b_rows = if op == Sm100Op::Tn { m } else { k };
        let output_rows = if op == Sm100Op::Tn { k } else { m };
        let output_columns = if op == Sm100Op::Nt { k } else { n };
        let shape = Sm100Shape {
            m,
            k,
            n,
            lda: round_up_8(a_width),
            ldb: round_up_8(n),
            ldc: output_columns + 3,
        };
        let mut a_host = vec![13.0; m * shape.lda];
        let mut b_host = vec![-11.0; b_rows * shape.ldb];
        let a_values = synth(m * a_width, 0xA11CE ^ m as u64 ^ (k as u64) << 8);
        let b_values = synth(b_rows * n, 0xB0B ^ n as u64 ^ (m as u64) << 8);
        for row in 0..m {
            for column in 0..a_width {
                a_host[row * shape.lda + column] =
                    quantize(dtype, a_values[row * a_width + column]);
            }
        }
        for row in 0..b_rows {
            for column in 0..n {
                b_host[row * shape.ldb + column] = quantize(dtype, b_values[row * n + column]);
            }
        }
        assert!(a_origin.0 + a_width <= shape.lda);
        assert!(b_origin.0 + n <= shape.ldb);
        let a_offset = a_origin.1 * shape.lda + a_origin.0;
        let b_offset = b_origin.1 * shape.ldb + b_origin.0;
        let a_alternate_offset = a_offset + a_host.len();
        let b_alternate_offset = b_offset + b_host.len();
        let mut a_storage = vec![17.0; a_alternate_offset + a_host.len()];
        a_storage[a_offset..a_offset + a_host.len()].copy_from_slice(&a_host);
        a_storage[a_alternate_offset..].copy_from_slice(&a_host);
        let mut b_storage = vec![-19.0; b_alternate_offset + b_host.len()];
        b_storage[b_offset..b_offset + b_host.len()].copy_from_slice(&b_host);
        b_storage[b_alternate_offset..].copy_from_slice(&b_host);
        let a = DtypedBuf::zeros(&ctx.stream, a_storage.len(), dtype).expect("A");
        a.upload_f32(&ctx.stream, &a_storage).expect("upload A");
        let b = DtypedBuf::zeros(&ctx.stream, b_storage.len(), dtype).expect("B");
        b.upload_f32(&ctx.stream, &b_storage).expect("upload B");
        let bias_host = synth(output_columns, 0xB1A5 ^ output_columns as u64);
        let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias");
        let output_dtype = if op == Sm100Op::Tn {
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
            a_offset,
            b_offset,
            a_alternate_offset,
            b_alternate_offset,
            bias,
            a_host,
            b_host,
            bias_host,
            initial,
            expected: Vec::new(),
            output_rows,
            output_columns,
        };
        case.expected = case.reference();
        case
    }

    fn request(&self, tile: Sm100Tile) -> Sm100MapRequest {
        self.request_at_offsets(tile, self.a_offset, self.b_offset)
    }

    fn alternate_request(&self, tile: Sm100Tile) -> Sm100MapRequest {
        self.request_at_offsets(tile, self.a_alternate_offset, self.b_alternate_offset)
    }

    fn request_at_offsets(
        &self,
        tile: Sm100Tile,
        a_offset: usize,
        b_offset: usize,
    ) -> Sm100MapRequest {
        Sm100MapRequest {
            op: self.op,
            dtype: self.dtype,
            tile,
            a_ptr: self.a.cached_ptr() + (a_offset * 2) as u64,
            b_ptr: self.b.cached_ptr() + (b_offset * 2) as u64,
            shape: self.shape,
        }
    }

    fn operands(&self, output_ptr: u64) -> Sm100LaunchOperands {
        Sm100LaunchOperands {
            output_ptr,
            bias_ptr: if self.op == Sm100Op::Nn {
                self.bias.cached_ptr()
            } else {
                0
            },
            alpha: if self.op == Sm100Op::Nn { 1.0 } else { 0.75 },
            beta: match self.op {
                Sm100Op::Nn => -0.25,
                Sm100Op::Tn => 1.0,
                Sm100Op::Nt => 0.0,
            },
        }
    }

    fn reference(&self) -> Vec<f32> {
        self.reference_for(self.operands(0))
    }

    fn reference_for(&self, operands: Sm100LaunchOperands) -> Vec<f32> {
        let mut output = self.initial.clone();
        for row in 0..self.output_rows {
            for column in 0..self.output_columns {
                let reduction = match self.op {
                    Sm100Op::Nn => self.shape.k,
                    Sm100Op::Tn => self.shape.m,
                    Sm100Op::Nt => self.shape.n,
                };
                let mut sum = 0.0f32;
                for inner in 0..reduction {
                    let (a, b) = match self.op {
                        Sm100Op::Nn => (
                            self.a_host[row * self.shape.lda + inner],
                            self.b_host[inner * self.shape.ldb + column],
                        ),
                        Sm100Op::Tn => (
                            self.a_host[inner * self.shape.lda + row],
                            self.b_host[inner * self.shape.ldb + column],
                        ),
                        Sm100Op::Nt => (
                            self.a_host[row * self.shape.lda + inner],
                            self.b_host[column * self.shape.ldb + inner],
                        ),
                    };
                    sum = a.mul_add(b, sum);
                }
                let index = row * self.shape.ldc + column;
                let value = match self.op {
                    Sm100Op::Nn => {
                        let seeded = if operands.bias_ptr == 0 {
                            sum
                        } else {
                            sum + self.bias_host[column]
                        };
                        let scaled = if operands.alpha == 1.0 {
                            seeded
                        } else {
                            operands.alpha * seeded
                        };
                        operands.beta.mul_add(self.initial[index], scaled)
                    }
                    Sm100Op::Tn => operands.alpha.mul_add(sum, self.initial[index]),
                    Sm100Op::Nt => operands.alpha * sum,
                };
                output[index] = if self.op == Sm100Op::Tn {
                    value
                } else {
                    quantize(self.dtype, value)
                };
            }
        }
        output
    }

    fn assert_correct(&self, actual: &[f32], route: Sm100ForcedRoute) {
        self.assert_correct_for(actual, route, &self.expected);
    }

    fn assert_correct_for(&self, actual: &[f32], route: Sm100ForcedRoute, expected: &[f32]) {
        let (absolute_tolerance, relative_tolerance) = match (self.op, self.dtype) {
            (Sm100Op::Tn, _) => (8.0e-4, 8.0e-4),
            (_, WeightDtype::Bf16) => (3.0e-2, 3.0e-2),
            (_, WeightDtype::F16) => (5.0e-3, 5.0e-3),
            (_, WeightDtype::F32) => unreachable!(),
        };
        for row in 0..self.output_rows {
            for column in 0..self.output_columns {
                let index = row * self.shape.ldc + column;
                let got = actual[index];
                let want = expected[index];
                let tolerance = absolute_tolerance + relative_tolerance * want.abs();
                assert!(
                    got.is_finite() && (got - want).abs() <= tolerance,
                    "{:?}/{:?}/{:?} M{} K{} N{} at ({row},{column}): got {got}, expected {want}",
                    self.op,
                    self.dtype,
                    route.physical,
                    self.shape.m,
                    self.shape.k,
                    self.shape.n,
                );
            }
            for column in self.output_columns..self.shape.ldc {
                let index = row * self.shape.ldc + column;
                assert_eq!(
                    actual[index].to_bits(),
                    self.initial[index].to_bits(),
                    "{:?}/{:?}/{:?} overwrote ldc padding at ({row},{column})",
                    self.op,
                    self.dtype,
                    route.physical,
                );
            }
        }
    }
}

fn forced_route(ctx: &GpuCtx, case: &CensusCase, physical: Sm100PhysicalRoute) -> Sm100ForcedRoute {
    let requested = Sm100ForcedRoute {
        op: case.op,
        dtype: case.dtype,
        physical,
        shape: case.shape,
    };
    let target = ctx
        .kernels
        .sm100_target_candidate()
        .expect("SM100 target identity");
    resolve_sm100_forced(target.device_cc, Some(target), requested)
        .expect("valid forced route")
        .expect("exact SM100 route loaded")
}

fn assert_identity(prepared: &Sm100PreparedLaunch, route: Sm100ForcedRoute) {
    let identity = prepared.identity();
    assert_eq!(identity.numeric_contract, Sm100NumericContract::Tcgen05F32);
    assert_eq!(identity.op, route.op);
    assert_eq!(identity.dtype, route.dtype);
    assert_eq!(identity.physical, route.physical);
    assert_eq!(identity.shape, route.shape);
    assert_eq!(identity.symbol, route.kernel_spec().unwrap().symbol);
    assert_eq!(identity.module_kind, ModuleKind::TriadSm100);
    assert_eq!(
        identity.target.device_cc,
        (
            identity.device.compute_capability.0 as i32,
            identity.device.compute_capability.1 as i32,
        )
    );
    assert_eq!(
        identity.target.nvrtc_arch,
        identity.compiler.target.as_str()
    );
    let global_target = match identity.device.compute_capability {
        (10, 0) => "sm_100a",
        (10, 3) => "sm_103a",
        cc => panic!("unexpected SM100 device identity {cc:?}"),
    };
    assert_eq!(identity.device.target.as_str(), global_target);
    assert!(matches!(
        identity.target.ptx_target,
        "sm_100f" | "sm_100a" | "sm_103f" | "sm_103a"
    ));
    assert_eq!(identity.tensor_map_revision, SM100_TENSOR_MAP_REVISION);
    assert_eq!(identity.tuning_revision, SM100_TUNING_REVISION);
}

fn run_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    maps: &Sm100PreparedTensorMaps,
    physical: Sm100PhysicalRoute,
    reruns: usize,
) {
    let route = forced_route(ctx, case, physical);
    let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let prepared = prepare_sm100_tcgen_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        maps,
        case.operands(output.ptr()),
    )
    .expect("prepare forced SM100 launch");
    assert_identity(&prepared, route);
    assert_eq!(
        prepared.identity().tensor_maps_digest,
        maps.identity_digest()
    );

    let mut baseline_bits = Vec::new();
    for repetition in 0..=reruns {
        output.reset(ctx, &case.initial);
        let identity = launch_sm100_tcgen_prepared(&ctx.stream, &ctx.kernels, &prepared)
            .expect("launch prepared SM100 route");
        assert_eq!(identity, prepared.identity());
        let actual = output.download(ctx, case.initial.len());
        let bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        if repetition == 0 {
            case.assert_correct(&actual, route);
            baseline_bits = bits;
        } else {
            assert_eq!(
                bits, baseline_bits,
                "{:?}/{:?}/{physical:?} changed bits on repetition {repetition}",
                case.op, case.dtype,
            );
        }
    }
}

fn run_case(ctx: &GpuCtx, case: &CensusCase, reruns: usize) {
    for tile in [Sm100Tile::M128N64, Sm100Tile::M128N128] {
        let maps = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(tile))
            .expect("prepare SM100 tensor maps");
        let cached = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(tile))
            .expect("reuse SM100 tensor maps");
        assert_eq!(maps.identity_digest(), cached.identity_digest());
        for stages in [Sm100Stages::S2, Sm100Stages::S3, Sm100Stages::S4] {
            for schedule in [Sm100Schedule::C4, Sm100Schedule::P8] {
                run_route(
                    ctx,
                    case,
                    &maps,
                    Sm100PhysicalRoute {
                        tile,
                        stages,
                        schedule,
                    },
                    reruns,
                );
            }
        }
    }
}

fn run_alternate_subview_case(ctx: &GpuCtx, case: &CensusCase) {
    for tile in [Sm100Tile::M128N64, Sm100Tile::M128N128] {
        let primary = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(tile))
            .expect("prepare primary subview maps");
        let alternate =
            prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.alternate_request(tile))
                .expect("prepare alternate subview maps");
        let alternate_cached =
            prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.alternate_request(tile))
                .expect("reuse alternate subview maps");
        assert_ne!(primary.identity_digest(), alternate.identity_digest());
        assert_eq!(
            alternate.identity_digest(),
            alternate_cached.identity_digest()
        );
        for stages in [Sm100Stages::S2, Sm100Stages::S3, Sm100Stages::S4] {
            for schedule in [Sm100Schedule::C4, Sm100Schedule::P8] {
                run_route(
                    ctx,
                    case,
                    &alternate,
                    Sm100PhysicalRoute {
                        tile,
                        stages,
                        schedule,
                    },
                    1,
                );
            }
        }
    }
}

fn changed_operands(op: Sm100Op, mut operands: Sm100LaunchOperands) -> Sm100LaunchOperands {
    if op == Sm100Op::Nn {
        operands.beta = f32::from_bits(operands.beta.to_bits() ^ 1);
    } else {
        operands.alpha = f32::from_bits(operands.alpha.to_bits() ^ 1);
    }
    operands
}

fn run_graph_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    maps: &Sm100PreparedTensorMaps,
    physical: Sm100PhysicalRoute,
) {
    let route = forced_route(ctx, case, physical);
    let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let operands = case.operands(output.ptr());
    let prepared = prepare_sm100_tcgen_forced(&ctx.stream, &ctx.kernels, route, maps, operands)
        .expect("prepare graph launch");
    output.reset(ctx, &case.initial);
    let eager_identity = launch_sm100_tcgen_prepared(&ctx.stream, &ctx.kernels, &prepared)
        .expect("eager graph baseline");
    let eager = output.download(ctx, case.initial.len());
    case.assert_correct(&eager, route);
    let eager_digest = output_digest(&eager);

    output.reset(ctx, &case.initial);
    let mut captured_identity = None;
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            let prepare_error =
                prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical.tile))
                    .expect_err("tensor-map preparation during capture must fail");
            if !prepare_error.contains("prepared before graph capture") {
                return Err(format!(
                    "unexpected capture-time preparation error: {prepare_error}"
                ));
            }
            captured_identity = Some(launch_sm100_tcgen_prepared(
                &ctx.stream,
                &ctx.kernels,
                &prepared,
            )?);
            Ok(())
        })
    }
    .expect("capture prepared SM100 route");
    let captured_identity = captured_identity.expect("captured route identity");
    eager_identity
        .ensure_current(captured_identity, "SM100 graph check")
        .expect("eager and captured identity agree");

    let changed = prepare_sm100_tcgen_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        maps,
        changed_operands(case.op, operands),
    )
    .expect("prepare changed-scalar identity");
    captured_identity
        .ensure_current(changed.identity(), "SM100 graph scalar guard")
        .expect_err("changed scalar must invalidate the prepared identity");

    for replay in 0..GRAPH_REPLAYS {
        output.reset(ctx, &case.initial);
        validate_sm100_graph_replay(&ctx.stream, &ctx.kernels, &prepared)
            .expect("validate SM100 graph replay");
        graph.launch().expect("launch captured SM100 graph");
        let actual = output.download(ctx, case.initial.len());
        assert_eq!(
            output_digest(&actual),
            eager_digest,
            "{:?}/{:?}/{physical:?} graph replay {replay} changed output bits",
            case.op,
            case.dtype,
        );
    }
}

fn run_graph_case(ctx: &GpuCtx, case: &CensusCase) {
    for tile in [Sm100Tile::M128N64, Sm100Tile::M128N128] {
        let maps = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(tile))
            .expect("prepare graph tensor maps");
        for stages in [Sm100Stages::S2, Sm100Stages::S3, Sm100Stages::S4] {
            for schedule in [Sm100Schedule::C4, Sm100Schedule::P8] {
                run_graph_route(
                    ctx,
                    case,
                    &maps,
                    Sm100PhysicalRoute {
                        tile,
                        stages,
                        schedule,
                    },
                );
            }
        }
    }
}

fn assert_nn_without_bias_scalars(ctx: &GpuCtx, dtype: WeightDtype) {
    let case = CensusCase::new(ctx, Sm100Op::Nn, dtype, ASYMMETRIC);
    for (tile, alpha, beta) in [
        (Sm100Tile::M128N64, 0.5, 0.0),
        (Sm100Tile::M128N128, -0.75, 0.5),
        (Sm100Tile::M128N64, 0.0, 1.0),
    ] {
        let physical = Sm100PhysicalRoute {
            tile,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        };
        let maps = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(tile))
            .expect("prepare bias-free NN maps");
        let route = forced_route(ctx, &case, physical);
        let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
        output.reset(ctx, &case.initial);
        let operands = Sm100LaunchOperands {
            output_ptr: output.ptr(),
            bias_ptr: 0,
            alpha,
            beta,
        };
        let expected = case.reference_for(operands);
        let prepared =
            prepare_sm100_tcgen_forced(&ctx.stream, &ctx.kernels, route, &maps, operands)
                .expect("prepare bias-free NN scalar path");
        launch_sm100_tcgen_prepared(&ctx.stream, &ctx.kernels, &prepared)
            .expect("launch bias-free NN scalar path");
        let actual = output.download(ctx, case.initial.len());
        case.assert_correct_for(&actual, route, &expected);
    }
}

fn assert_multi_stream_replay(ctx: &GpuCtx) {
    let case = CensusCase::new(ctx, Sm100Op::Nn, WeightDtype::Bf16, ASYMMETRIC);
    let physical = Sm100PhysicalRoute {
        tile: Sm100Tile::M128N128,
        stages: Sm100Stages::S3,
        schedule: Sm100Schedule::P8,
    };
    let route = forced_route(ctx, &case, physical);
    let maps = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical.tile))
        .expect("prepare multi-stream maps");
    let alternate = ctx.stream.fork().expect("fork second stream");
    let mut main_output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let mut alternate_output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let main = prepare_sm100_tcgen_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        &maps,
        case.operands(main_output.ptr()),
    )
    .expect("prepare main-stream route");
    let on_alternate = prepare_sm100_tcgen_forced(
        &alternate,
        &ctx.kernels,
        route,
        &maps,
        case.operands(alternate_output.ptr()),
    )
    .expect("prepare alternate-stream route");

    for repetition in 0..16 {
        main_output.reset_on(&ctx.stream, &case.initial);
        alternate_output.reset_on(&alternate, &case.initial);
        launch_sm100_tcgen_prepared(&ctx.stream, &ctx.kernels, &main)
            .expect("launch main-stream route");
        launch_sm100_tcgen_prepared(&alternate, &ctx.kernels, &on_alternate)
            .expect("launch alternate-stream route");
        let main_values = main_output.download_on(&ctx.stream, case.initial.len());
        let alternate_values = alternate_output.download_on(&alternate, case.initial.len());
        case.assert_correct(&main_values, route);
        assert_eq!(
            output_digest(&main_values),
            output_digest(&alternate_values),
            "multi-stream replay {repetition} changed output bits"
        );
    }
}

fn assert_semantic_rejections(ctx: &GpuCtx) {
    for op in [Sm100Op::Nn, Sm100Op::Tn, Sm100Op::Nt] {
        let case = CensusCase::new(ctx, op, WeightDtype::Bf16, ASYMMETRIC);
        let physical = Sm100PhysicalRoute {
            tile: Sm100Tile::M128N64,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        };
        let maps =
            prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical.tile))
                .expect("prepare rejection maps");
        let route = forced_route(ctx, &case, physical);
        let output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
        let operands = case.operands(output.ptr());
        let rejected = match op {
            Sm100Op::Nn => Sm100LaunchOperands {
                alpha: 0.75,
                ..operands
            },
            Sm100Op::Tn => Sm100LaunchOperands {
                bias_ptr: case.bias.cached_ptr(),
                ..operands
            },
            Sm100Op::Nt => Sm100LaunchOperands {
                beta: 1.0,
                ..operands
            },
        };
        assert!(
            prepare_sm100_tcgen_forced(&ctx.stream, &ctx.kernels, route, &maps, rejected).is_err()
        );
    }
}

fn assert_map_and_route_fail_closed(ctx: &GpuCtx) {
    let case = CensusCase::new(ctx, Sm100Op::Nn, WeightDtype::Bf16, ASYMMETRIC);
    let physical = Sm100PhysicalRoute {
        tile: Sm100Tile::M128N64,
        stages: Sm100Stages::S2,
        schedule: Sm100Schedule::C4,
    };
    let mut request = case.request(physical.tile);
    request.a_ptr = 0;
    assert!(validate_sm100_map_request(request).is_err());
    assert!(prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    request = case.request(physical.tile);
    request.b_ptr += 1;
    assert!(validate_sm100_map_request(request).is_err());
    assert!(prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    request = case.request(physical.tile);
    request.shape.lda += 1;
    assert!(validate_sm100_map_request(request).is_err());
    assert!(prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    let target = ctx
        .kernels
        .sm100_target_candidate()
        .expect("SM100 target identity");
    for axis in 0..3 {
        request = case.request(physical.tile);
        match axis {
            0 => request.shape.m = 0,
            1 => request.shape.k = 0,
            _ => request.shape.n = 0,
        }
        assert!(validate_sm100_map_request(request).is_err());
        assert!(prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());
        let invalid = Sm100ForcedRoute {
            op: case.op,
            dtype: case.dtype,
            physical,
            shape: request.shape,
        };
        assert!(resolve_sm100_forced(target.device_cc, Some(target), invalid).is_err());
    }

    let requested = Sm100ForcedRoute {
        op: case.op,
        dtype: case.dtype,
        physical,
        shape: case.shape,
    };
    assert_eq!(
        resolve_sm100_forced(target.device_cc, None, requested).expect("fail-closed route result"),
        None
    );
    assert!(SM100_AUTO_CELLS_CC100.is_empty());
    assert!(SM100_AUTO_CELLS_CC103.is_empty());
}

fn assert_replay_rejects_allocation_generation_change(ctx: &GpuCtx) {
    let case = CensusCase::new(ctx, Sm100Op::Nn, WeightDtype::Bf16, ASYMMETRIC);
    let physical = Sm100PhysicalRoute {
        tile: Sm100Tile::M128N64,
        stages: Sm100Stages::S2,
        schedule: Sm100Schedule::C4,
    };
    let route = forced_route(ctx, &case, physical);
    let maps = prepare_sm100_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical.tile))
        .expect("prepare allocation-generation maps");
    let mut output = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    output.reset(ctx, &case.initial);
    let prepared = prepare_sm100_tcgen_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        &maps,
        case.operands(output.ptr()),
    )
    .expect("prepare allocation-generation route");
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_sm100_tcgen_prepared(&ctx.stream, &ctx.kernels, &prepared).map(|_| ())
        })
    }
    .expect("capture allocation-generation route");
    drop(output);
    let replacement = CensusOutput::new(ctx, case.op, case.dtype, case.initial.len());
    let error = validate_sm100_graph_replay(&ctx.stream, &ctx.kernels, &prepared)
        .expect_err("freed or recycled output allocation must invalidate graph replay");
    assert!(error.contains("allocation"), "{error}");
    drop(replacement);
    drop(graph);
}

#[test]
#[ignore = "needs a real CC 10.0 or CC 10.3 Blackwell device"]
fn qualifies_all_forced_sm100_triad_routes() {
    let first_device = match GpuDevice::new(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping the SM100 triad check without a CUDA device: {error}");
            return;
        }
    };
    if !matches!(first_device.compute_capability, (10, 0) | (10, 3)) {
        eprintln!(
            "skipping the SM100 triad check on compute capability {}.{}",
            first_device.compute_capability.0, first_device.compute_capability.1
        );
        return;
    }
    assert_eq!(SM100_KERNEL_SPECS.len(), 72);
    let first_ctx = GpuCtx::new(&first_device).expect("first GPU context");
    assert!(first_ctx.kernels.has_sm100_tcgen());
    let first_case = CensusCase::new(&first_ctx, Sm100Op::Nn, WeightDtype::Bf16, (31, 32, 65));
    let first_maps = prepare_sm100_tensor_maps(
        &first_ctx.stream,
        &first_ctx.kernels,
        first_case.request(Sm100Tile::M128N64),
    )
    .expect("prepare first-context maps");
    run_route(
        &first_ctx,
        &first_case,
        &first_maps,
        Sm100PhysicalRoute {
            tile: Sm100Tile::M128N64,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        },
        1,
    );
    drop(first_case);
    drop(first_ctx);
    drop(first_device);

    let device = GpuDevice::new(0).expect("fresh CUDA device context");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(
        ctx.kernels.has_sm100_tcgen(),
        "Blackwell detected but the complete SM100 triad module failed qualification"
    );

    for op in [Sm100Op::Nn, Sm100Op::Tn, Sm100Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for dims in independent_axis_shapes() {
                let case = CensusCase::new(&ctx, op, dtype, dims);
                run_case(&ctx, &case, 0);
            }
            let subview = CensusCase::with_origins(&ctx, op, dtype, ASYMMETRIC, (1, 1), (3, 2));
            assert!(!subview.request(Sm100Tile::M128N64).a_ptr.is_multiple_of(16));
            assert!(!subview.request(Sm100Tile::M128N64).b_ptr.is_multiple_of(16));
            run_case(&ctx, &subview, 0);
            run_alternate_subview_case(&ctx, &subview);
            let asymmetric = CensusCase::new(&ctx, op, dtype, ASYMMETRIC);
            run_case(&ctx, &asymmetric, DETERMINISM_RERUNS);
            if op == Sm100Op::Nn && dtype == WeightDtype::Bf16 {
                run_graph_case(&ctx, &subview);
            } else {
                run_graph_case(&ctx, &asymmetric);
            }
        }
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        assert_nn_without_bias_scalars(&ctx, dtype);
    }
    assert_multi_stream_replay(&ctx);
    assert_semantic_rejections(&ctx);
    assert_map_and_route_fail_closed(&ctx);
    assert_replay_rejects_allocation_generation_change(&ctx);
}
