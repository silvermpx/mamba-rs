//! Real-device qualification for the deterministic generic SM120 triad.
//!
//! Automatic tables remain empty. This census forces every physical route,
//! checks the ordinary MMA16 baseline bit-for-bit, and is also the focused
//! target for memcheck, initcheck, racecheck, and synccheck.
#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::sync::Arc;

use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, SM120_KERNEL_SPECS, SM120_TENSOR_MAP_REVISION,
    SM120_TUNING_REVISION, Sm120Bk, Sm120ForcedRoute, Sm120LaunchOperands, Sm120MapRequest,
    Sm120NumericContract, Sm120Op, Sm120PhysicalRoute, Sm120PreparedLaunch,
    Sm120PreparedTensorMaps, Sm120Shape, Sm120Stages, Sm120Tile, TcFwdOperands, TcTile,
    launch_sm120_tma_prepared, prepare_sm120_tensor_maps, prepare_sm120_tma_forced,
    resolve_sm120_forced, sgemm_bi_backward_dw_tc_with_tile, sgemm_bi_backward_dx_tc_with_tile,
    sgemm_bi_forward_tc_with_tile, validate_sm120_graph_replay, validate_sm120_map_request,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{FramedSha256, ModuleKind, Sha256Digest};

const EDGES: &[usize] = &[1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129];
const ASYMMETRIC: (usize, usize, usize) = (65, 127, 129);
const ALIGNED_BASELINE: (usize, usize, usize) = (128, 128, 128);
const EAGER_REPEATS: usize = 100;
const GRAPH_REPLAYS: usize = 100;
const GUARD: usize = 64;

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

fn synth(len: usize, seed: u64, dtype: WeightDtype) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let base = ((state & 0xFFFFFF) as f32 / 16_777_216.0) * 0.4 - 0.2;
            let value = match index % 8 {
                0 => base * 8.0,
                1 => -base * 4.0,
                _ => base,
            };
            quantize(dtype, value)
        })
        .collect()
}

fn adversarial(len: usize, dtype: WeightDtype) -> Vec<f32> {
    let values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffc0_4321),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    (0..len)
        .map(|index| quantize(dtype, values[index % values.len()]))
        .collect()
}

fn digest(values: &[f32]) -> Sha256Digest {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    FramedSha256::bytes(&bytes)
}

fn physical_routes() -> Vec<Sm120PhysicalRoute> {
    let mut routes = Vec::new();
    for tile in [
        Sm120Tile::M64N64,
        Sm120Tile::M128N64,
        Sm120Tile::M64N128,
        Sm120Tile::M128N128,
    ] {
        for bk in [Sm120Bk::Bk32, Sm120Bk::Bk64] {
            for stages in [Sm120Stages::S2, Sm120Stages::S3] {
                routes.push(Sm120PhysicalRoute { tile, bk, stages });
            }
        }
    }
    routes
}

fn independent_axis_shapes() -> BTreeSet<(usize, usize, usize)> {
    let mut shapes = BTreeSet::new();
    for &edge in EDGES {
        shapes.insert((edge, ASYMMETRIC.1, ASYMMETRIC.2));
        shapes.insert((ASYMMETRIC.0, edge, ASYMMETRIC.2));
        shapes.insert((ASYMMETRIC.0, ASYMMETRIC.1, edge));
    }
    shapes
}

enum OutputStorage {
    Typed(DtypedBuf),
    F32(GpuBuffer),
}

struct GuardedOutput {
    storage: OutputStorage,
    dtype: WeightDtype,
    offset: usize,
    logical_len: usize,
    sentinel: f32,
}

impl GuardedOutput {
    fn new(
        ctx: &GpuCtx,
        op: Sm120Op,
        dtype: WeightDtype,
        logical_len: usize,
        offset: usize,
    ) -> Self {
        let output_dtype = if op == Sm120Op::Tn {
            WeightDtype::F32
        } else {
            dtype
        };
        let len = offset + logical_len + GUARD;
        let storage = if output_dtype == WeightDtype::F32 {
            OutputStorage::F32(GpuBuffer::zeros(&ctx.stream, len).expect("f32 output"))
        } else {
            OutputStorage::Typed(
                DtypedBuf::zeros(&ctx.stream, len, output_dtype).expect("typed output"),
            )
        };
        Self {
            storage,
            dtype: output_dtype,
            offset,
            logical_len,
            sentinel: quantize(output_dtype, 19.25),
        }
    }

    fn ptr(&self) -> u64 {
        let base = match &self.storage {
            OutputStorage::Typed(buffer) => buffer.cached_ptr(),
            OutputStorage::F32(buffer) => buffer.cached_ptr(),
        };
        base + (self.offset * self.dtype.size_bytes()) as u64
    }

    fn reset(&mut self, stream: &Arc<cudarc::driver::CudaStream>, initial: &[f32]) {
        assert_eq!(initial.len(), self.logical_len);
        let mut values = vec![self.sentinel; self.offset + self.logical_len + GUARD];
        values[self.offset..self.offset + self.logical_len].copy_from_slice(initial);
        match &mut self.storage {
            OutputStorage::Typed(buffer) => {
                buffer.upload_f32(stream, &values).expect("reset typed")
            }
            OutputStorage::F32(buffer) => buffer.upload(stream, &values).expect("reset f32"),
        }
    }

    fn download(&self, stream: &Arc<cudarc::driver::CudaStream>) -> Vec<f32> {
        let mut values = match &self.storage {
            OutputStorage::Typed(buffer) => {
                let mut values = vec![0.0; self.offset + self.logical_len + GUARD];
                buffer
                    .download_f32(stream, &mut values)
                    .expect("download typed output");
                values
            }
            OutputStorage::F32(buffer) => buffer.to_cpu(stream).expect("download f32 output"),
        };
        for (index, value) in values[..self.offset].iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                self.sentinel.to_bits(),
                "output prefix guard {index}"
            );
        }
        for (index, value) in values[self.offset + self.logical_len..].iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                self.sentinel.to_bits(),
                "output suffix guard {index}"
            );
        }
        values.drain(..self.offset);
        values.truncate(self.logical_len);
        values
    }
}

struct CensusCase {
    op: Sm120Op,
    dtype: WeightDtype,
    shape: Sm120Shape,
    a: DtypedBuf,
    b: DtypedBuf,
    a_offset: usize,
    b_offset: usize,
    a_storage_before: Vec<f32>,
    b_storage_before: Vec<f32>,
    a_values: Vec<f32>,
    b_values: Vec<f32>,
    bias: GpuBuffer,
    bias_values: Vec<f32>,
    initial: Vec<f32>,
    expected: Vec<f32>,
    output_rows: usize,
    output_columns: usize,
}

impl CensusCase {
    fn new(ctx: &GpuCtx, op: Sm120Op, dtype: WeightDtype, dims: (usize, usize, usize)) -> Self {
        Self::with_origins(ctx, op, dtype, dims, (0, 0), (0, 0), false)
    }

    fn compact(ctx: &GpuCtx, op: Sm120Op, dtype: WeightDtype, dims: (usize, usize, usize)) -> Self {
        Self::with_origins(ctx, op, dtype, dims, (0, 0), (0, 0), true)
    }

    fn with_origins(
        ctx: &GpuCtx,
        op: Sm120Op,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        a_origin: (usize, usize),
        b_origin: (usize, usize),
        compact: bool,
    ) -> Self {
        let (m, k, n) = dims;
        let (a_rows, a_width) = match op {
            Sm120Op::Nn | Sm120Op::Tn => (m, k),
            Sm120Op::Nt => (m, n),
        };
        let (b_rows, b_width) = match op {
            Sm120Op::Nn => (k, n),
            Sm120Op::Tn => (m, n),
            Sm120Op::Nt => (k, n),
        };
        let output_rows = if op == Sm120Op::Tn { k } else { m };
        let output_columns = if op == Sm120Op::Nt { k } else { n };
        let lda = if compact {
            assert_eq!(a_origin, (0, 0));
            assert!(a_width.is_multiple_of(8));
            a_width
        } else {
            round_up_8(a_origin.0 + a_width + 1)
        };
        let ldb = if compact {
            assert_eq!(b_origin, (0, 0));
            assert!(b_width.is_multiple_of(8));
            b_width
        } else {
            round_up_8(b_origin.0 + b_width + 1)
        };
        let shape = Sm120Shape {
            m,
            k,
            n,
            lda,
            ldb,
            ldc: if compact {
                output_columns
            } else {
                output_columns + 3
            },
        };
        let a_values = synth(
            a_rows * a_width,
            0xA11CE ^ m as u64 ^ (k as u64) << 8,
            dtype,
        );
        let b_values = synth(b_rows * b_width, 0xB0B ^ n as u64 ^ (m as u64) << 8, dtype);
        let a_offset = a_origin.1 * lda + a_origin.0;
        let b_offset = b_origin.1 * ldb + b_origin.0;
        let mut a_storage = vec![quantize(dtype, 13.0); (a_origin.1 + a_rows) * lda + GUARD];
        let mut b_storage = vec![quantize(dtype, -11.0); (b_origin.1 + b_rows) * ldb + GUARD];
        for row in 0..a_rows {
            let source = &a_values[row * a_width..(row + 1) * a_width];
            let start = a_offset + row * lda;
            a_storage[start..start + a_width].copy_from_slice(source);
        }
        for row in 0..b_rows {
            let source = &b_values[row * b_width..(row + 1) * b_width];
            let start = b_offset + row * ldb;
            b_storage[start..start + b_width].copy_from_slice(source);
        }
        let a = DtypedBuf::zeros(&ctx.stream, a_storage.len(), dtype).expect("A");
        a.upload_f32(&ctx.stream, &a_storage).expect("upload A");
        let b = DtypedBuf::zeros(&ctx.stream, b_storage.len(), dtype).expect("B");
        b.upload_f32(&ctx.stream, &b_storage).expect("upload B");
        let bias_values = synth(
            output_columns,
            0xB1A5 ^ output_columns as u64,
            WeightDtype::F32,
        );
        let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_values).expect("bias");
        let output_dtype = if op == Sm120Op::Tn {
            WeightDtype::F32
        } else {
            dtype
        };
        let initial = synth(
            output_rows * shape.ldc,
            0xC001 ^ output_rows as u64,
            output_dtype,
        );
        let mut case = Self {
            op,
            dtype,
            shape,
            a,
            b,
            a_offset,
            b_offset,
            a_storage_before: a_storage,
            b_storage_before: b_storage,
            a_values,
            b_values,
            bias,
            bias_values,
            initial,
            expected: Vec::new(),
            output_rows,
            output_columns,
        };
        case.expected = case.reference(case.operands(0, false));
        case
    }

    fn request(&self, physical: Sm120PhysicalRoute) -> Sm120MapRequest {
        Sm120MapRequest {
            op: self.op,
            dtype: self.dtype,
            tile: physical.tile,
            bk: physical.bk,
            a_ptr: self.a.cached_ptr() + (self.a_offset * 2) as u64,
            b_ptr: self.b.cached_ptr() + (self.b_offset * 2) as u64,
            shape: self.shape,
        }
    }

    fn operands(&self, output_ptr: u64, baseline_contract: bool) -> Sm120LaunchOperands {
        Sm120LaunchOperands {
            output_ptr,
            bias_ptr: if self.op == Sm120Op::Nn {
                self.bias.cached_ptr()
            } else {
                0
            },
            alpha: if baseline_contract || self.op == Sm120Op::Nn {
                1.0
            } else {
                0.75
            },
            beta: if baseline_contract {
                0.0
            } else if self.op == Sm120Op::Nn {
                -0.25
            } else if self.op == Sm120Op::Tn {
                1.0
            } else {
                0.0
            },
        }
    }

    fn reference(&self, operands: Sm120LaunchOperands) -> Vec<f32> {
        let mut output = self.initial.clone();
        for row in 0..self.output_rows {
            for column in 0..self.output_columns {
                let reduction = match self.op {
                    Sm120Op::Nn => self.shape.k,
                    Sm120Op::Tn => self.shape.m,
                    Sm120Op::Nt => self.shape.n,
                };
                let mut sum = 0.0f32;
                for inner in 0..reduction {
                    let (a, b) = match self.op {
                        Sm120Op::Nn => (
                            self.a_values[row * self.shape.k + inner],
                            self.b_values[inner * self.shape.n + column],
                        ),
                        Sm120Op::Tn => (
                            self.a_values[inner * self.shape.k + row],
                            self.b_values[inner * self.shape.n + column],
                        ),
                        Sm120Op::Nt => (
                            self.a_values[row * self.shape.n + inner],
                            self.b_values[column * self.shape.n + inner],
                        ),
                    };
                    sum = a.mul_add(b, sum);
                }
                let index = row * self.shape.ldc + column;
                let value = match self.op {
                    Sm120Op::Nn => {
                        let seeded = sum + self.bias_values[column];
                        operands
                            .beta
                            .mul_add(self.initial[index], operands.alpha * seeded)
                    }
                    Sm120Op::Tn => operands.alpha.mul_add(sum, self.initial[index]),
                    Sm120Op::Nt => operands.alpha * sum,
                };
                output[index] = if self.op == Sm120Op::Tn {
                    value
                } else {
                    quantize(self.dtype, value)
                };
            }
        }
        output
    }

    fn assert_cpu_reference(&self, actual: &[f32], route: Sm120ForcedRoute) {
        let (absolute_tolerance, relative_tolerance) = match (self.op, self.dtype) {
            (Sm120Op::Tn, _) => (8.0e-4, 8.0e-4),
            (_, WeightDtype::Bf16) => (3.0e-2, 3.0e-2),
            (_, WeightDtype::F16) => (5.0e-3, 5.0e-3),
            (_, WeightDtype::F32) => unreachable!(),
        };
        for row in 0..self.output_rows {
            for column in 0..self.output_columns {
                let index = row * self.shape.ldc + column;
                let got = actual[index];
                let want = self.expected[index];
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

    fn assert_inputs_unchanged(&self, stream: &Arc<cudarc::driver::CudaStream>) {
        let mut a = vec![0.0; self.a.len_elems()];
        let mut b = vec![0.0; self.b.len_elems()];
        self.a
            .download_f32(stream, &mut a)
            .expect("download A guard");
        self.b
            .download_f32(stream, &mut b)
            .expect("download B guard");
        assert_eq!(
            a.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            self.a_storage_before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "SM120 modified A"
        );
        assert_eq!(
            b.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            self.b_storage_before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "SM120 modified B"
        );
    }

    fn replace_with_adversarial_inputs(&mut self, stream: &Arc<cudarc::driver::CudaStream>) {
        assert_eq!(self.a_offset, 0);
        assert_eq!(self.b_offset, 0);
        self.a_values = adversarial(self.a_values.len(), self.dtype);
        self.b_values = adversarial(self.b_values.len(), self.dtype)
            .into_iter()
            .rev()
            .collect();
        self.a_storage_before[..self.a_values.len()].copy_from_slice(&self.a_values);
        self.b_storage_before[..self.b_values.len()].copy_from_slice(&self.b_values);
        self.a
            .upload_f32(stream, &self.a_storage_before)
            .expect("upload adversarial A");
        self.b
            .upload_f32(stream, &self.b_storage_before)
            .expect("upload adversarial B");
    }
}

fn forced_route(ctx: &GpuCtx, case: &CensusCase, physical: Sm120PhysicalRoute) -> Sm120ForcedRoute {
    let requested = Sm120ForcedRoute {
        op: case.op,
        dtype: case.dtype,
        physical,
        shape: case.shape,
    };
    let caps = ctx.kernels.sm120_device_caps().expect("SM120 device caps");
    let target = ctx
        .kernels
        .sm120_target_candidate()
        .expect("SM120 target identity");
    resolve_sm120_forced(caps, Some(target), requested)
        .expect("valid forced route")
        .expect("exact SM120 route loaded")
}

fn assert_identity_and_resources(prepared: &Sm120PreparedLaunch, route: Sm120ForcedRoute) {
    let identity = prepared.identity();
    assert_eq!(
        identity.numeric_contract,
        Sm120NumericContract::TmaMma16F32V1
    );
    assert_eq!(identity.op, route.op);
    assert_eq!(identity.dtype, route.dtype);
    assert_eq!(identity.physical, route.physical);
    assert_eq!(identity.shape, route.shape);
    assert_eq!(identity.symbol, route.kernel_spec().unwrap().symbol);
    assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
    assert_eq!(identity.tensor_map_revision, SM120_TENSOR_MAP_REVISION);
    assert_eq!(identity.tuning_revision, SM120_TUNING_REVISION);
    assert_eq!(
        identity.target.nvrtc_arch,
        identity.compiler.target.as_str()
    );
    assert!(matches!(identity.target.ptx_target, "sm_120" | "sm_121"));
    assert!(matches!(
        identity.device.compute_capability,
        (12, 0) | (12, 1)
    ));

    let spec = route.kernel_spec().unwrap();
    let resources = prepared.resources();
    assert_eq!(resources.threads, spec.threads);
    assert_eq!(resources.dynamic_shared_bytes, spec.dynamic_shared_bytes);
    assert!(resources.max_threads_per_block >= spec.threads);
    assert_eq!(
        resources.local_bytes, 0,
        "{} uses local memory",
        spec.symbol
    );
    assert_eq!(
        resources.spill_store_bytes, 0,
        "{} spills stores",
        spec.symbol
    );
    assert_eq!(
        resources.spill_load_bytes, 0,
        "{} spills loads",
        spec.symbol
    );
    assert!(
        resources.registers_per_thread > 0,
        "{} register census",
        spec.symbol
    );
    assert!(
        resources.active_blocks_per_sm > 0,
        "{} zero occupancy",
        spec.symbol
    );
}

fn prepared_route(
    ctx: &GpuCtx,
    case: &CensusCase,
    maps: &Sm120PreparedTensorMaps,
    physical: Sm120PhysicalRoute,
    output_ptr: u64,
    baseline_contract: bool,
) -> (Sm120ForcedRoute, Sm120PreparedLaunch) {
    let route = forced_route(ctx, case, physical);
    let prepared = prepare_sm120_tma_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        maps,
        case.operands(output_ptr, baseline_contract),
    )
    .expect("prepare forced SM120 launch");
    assert_identity_and_resources(&prepared, route);
    assert_eq!(
        prepared.identity().tensor_maps_digest,
        maps.identity_digest()
    );
    (route, prepared)
}

fn run_candidate(
    ctx: &GpuCtx,
    case: &CensusCase,
    physical: Sm120PhysicalRoute,
    reruns: usize,
    output_offset: usize,
) {
    let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("prepare SM120 tensor maps");
    let cached = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("reuse SM120 tensor maps");
    assert_eq!(maps.identity_digest(), cached.identity_digest());
    let mut output =
        GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), output_offset);
    let (route, prepared) = prepared_route(ctx, case, &maps, physical, output.ptr(), false);
    let mut baseline = Vec::new();
    for repetition in 0..=reruns {
        output.reset(&ctx.stream, &case.initial);
        let launched = launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)
            .expect("launch prepared SM120 route");
        assert_eq!(launched, prepared.identity());
        let actual = output.download(&ctx.stream);
        let bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        if repetition == 0 {
            case.assert_cpu_reference(&actual, route);
            baseline = bits;
        } else {
            assert_eq!(
                bits, baseline,
                "{:?}/{:?}/{physical:?} changed bits on repetition {repetition}",
                case.op, case.dtype,
            );
        }
    }
    case.assert_inputs_unchanged(&ctx.stream);
}

fn launch_baseline(ctx: &GpuCtx, case: &CensusCase, output: &GuardedOutput) -> Result<(), String> {
    let typed = |ptr| TypedPtr {
        ptr,
        dtype: case.dtype,
    };
    let dims = (case.shape.m, case.shape.k, case.shape.n);
    match case.op {
        Sm120Op::Nn => sgemm_bi_forward_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            &TcFwdOperands {
                y: typed(output.ptr()),
                x: typed(case.a.cached_ptr()),
                w: typed(case.b.cached_ptr()),
                bias_ptr: case.bias.cached_ptr(),
            },
            dims,
            TcTile::Tile128,
        ),
        Sm120Op::Tn => sgemm_bi_backward_dw_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            output.ptr(),
            typed(case.b.cached_ptr()),
            typed(case.a.cached_ptr()),
            dims,
            TcTile::Tile128,
        ),
        Sm120Op::Nt => sgemm_bi_backward_dx_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            typed(output.ptr()),
            typed(case.a.cached_ptr()),
            typed(case.b.cached_ptr()),
            dims,
            TcTile::Tile128,
        ),
    }
}

fn assert_exact_baseline(ctx: &GpuCtx, case: &CensusCase, physical: Sm120PhysicalRoute) {
    assert_eq!(case.a_offset, 0);
    assert_eq!(case.b_offset, 0);
    let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("prepare baseline-comparison maps");
    let mut candidate = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    let mut baseline = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    candidate.reset(&ctx.stream, &case.initial);
    baseline.reset(&ctx.stream, &case.initial);
    let (_, prepared) = prepared_route(ctx, case, &maps, physical, candidate.ptr(), true);
    launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)
        .expect("launch SM120 baseline comparison");
    launch_baseline(ctx, case, &baseline).expect("launch same-target MMA16 baseline");
    let candidate = candidate.download(&ctx.stream);
    let baseline = baseline.download(&ctx.stream);
    assert_eq!(
        candidate
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        baseline
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "{:?}/{:?}/{physical:?} differs from the same-target MMA16 baseline",
        case.op,
        case.dtype,
    );
    case.assert_inputs_unchanged(&ctx.stream);
}

fn changed_operands(op: Sm120Op, mut operands: Sm120LaunchOperands) -> Sm120LaunchOperands {
    if op == Sm120Op::Nn {
        operands.beta = f32::from_bits(operands.beta.to_bits() ^ 1);
    } else {
        operands.alpha = f32::from_bits(operands.alpha.to_bits() ^ 1);
    }
    operands
}

fn run_graph_route(ctx: &GpuCtx, case: &CensusCase, physical: Sm120PhysicalRoute) {
    let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("prepare graph tensor maps");
    let route = forced_route(ctx, case, physical);
    let mut output = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 3);
    let operands = case.operands(output.ptr(), false);
    let prepared = prepare_sm120_tma_forced(&ctx.stream, &ctx.kernels, route, &maps, operands)
        .expect("prepare graph launch");
    output.reset(&ctx.stream, &case.initial);
    let eager_identity = launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)
        .expect("eager graph baseline");
    let eager = output.download(&ctx.stream);
    case.assert_cpu_reference(&eager, route);
    let eager_digest = digest(&eager);

    output.reset(&ctx.stream, &case.initial);
    let mut captured_identity = None;
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            let prepare_error =
                prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
                    .expect_err("tensor-map preparation during capture must fail");
            if !prepare_error.contains("prepared before graph capture") {
                return Err(format!(
                    "unexpected capture-time preparation error: {prepare_error}"
                ));
            }
            captured_identity = Some(launch_sm120_tma_prepared(
                &ctx.stream,
                &ctx.kernels,
                &prepared,
            )?);
            Ok(())
        })
    }
    .expect("capture prepared SM120 route");
    let captured_identity = captured_identity.expect("captured route identity");
    eager_identity
        .ensure_current(captured_identity, "SM120 graph census")
        .expect("eager and captured identity agree");

    let changed = prepare_sm120_tma_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        &maps,
        changed_operands(case.op, operands),
    )
    .expect("prepare changed-scalar identity");
    captured_identity
        .ensure_current(changed.identity(), "SM120 graph scalar guard")
        .expect_err("changed scalar must invalidate graph identity");

    let alternate = ctx.stream.fork().expect("alternate validation stream");
    validate_sm120_graph_replay(&alternate, &ctx.kernels, &prepared)
        .expect_err("a graph route is bound to its capture stream");

    for replay in 0..GRAPH_REPLAYS {
        output.reset(&ctx.stream, &case.initial);
        validate_sm120_graph_replay(&ctx.stream, &ctx.kernels, &prepared)
            .expect("validate SM120 graph replay");
        graph.launch().expect("launch captured SM120 graph");
        let actual = output.download(&ctx.stream);
        assert_eq!(
            digest(&actual),
            eager_digest,
            "{:?}/{:?}/{physical:?} graph replay {replay} changed output bits",
            case.op,
            case.dtype,
        );
    }
}

fn assert_multi_stream_concurrency(ctx: &GpuCtx) {
    let case = CensusCase::compact(ctx, Sm120Op::Nn, WeightDtype::Bf16, ALIGNED_BASELINE);
    let physical = Sm120PhysicalRoute {
        tile: Sm120Tile::M128N128,
        bk: Sm120Bk::Bk64,
        stages: Sm120Stages::S3,
    };
    let route = forced_route(ctx, &case, physical);
    let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("prepare multi-stream maps");
    let alternate = ctx.stream.fork().expect("fork census stream");
    let mut main_output = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    let mut other_output = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    let main = prepare_sm120_tma_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        &maps,
        case.operands(main_output.ptr(), true),
    )
    .expect("prepare main-stream route");
    let other = prepare_sm120_tma_forced(
        &alternate,
        &ctx.kernels,
        route,
        &maps,
        case.operands(other_output.ptr(), true),
    )
    .expect("prepare alternate-stream route");
    for repetition in 0..16 {
        main_output.reset(&ctx.stream, &case.initial);
        other_output.reset(&alternate, &case.initial);
        launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &main)
            .expect("launch main-stream route");
        launch_sm120_tma_prepared(&alternate, &ctx.kernels, &other)
            .expect("launch alternate-stream route");
        let main_values = main_output.download(&ctx.stream);
        let other_values = other_output.download(&alternate);
        assert_eq!(
            digest(&main_values),
            digest(&other_values),
            "multi-stream repetition {repetition} changed output bits"
        );
    }
}

fn assert_semantic_rejections(ctx: &GpuCtx) {
    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        let case = CensusCase::new(ctx, op, WeightDtype::Bf16, ASYMMETRIC);
        let physical = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        };
        let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
            .expect("prepare rejection maps");
        let route = forced_route(ctx, &case, physical);
        let output = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
        let operands = case.operands(output.ptr(), false);
        let rejected = match op {
            Sm120Op::Nn => Sm120LaunchOperands {
                alpha: 0.75,
                ..operands
            },
            Sm120Op::Tn => Sm120LaunchOperands {
                bias_ptr: case.bias.cached_ptr(),
                ..operands
            },
            Sm120Op::Nt => Sm120LaunchOperands {
                beta: 1.0,
                ..operands
            },
        };
        assert!(
            prepare_sm120_tma_forced(&ctx.stream, &ctx.kernels, route, &maps, rejected).is_err()
        );
    }
}

fn assert_map_and_route_fail_closed(ctx: &GpuCtx) {
    let case = CensusCase::new(ctx, Sm120Op::Nn, WeightDtype::Bf16, ASYMMETRIC);
    let physical = Sm120PhysicalRoute {
        tile: Sm120Tile::M128N128,
        bk: Sm120Bk::Bk64,
        stages: Sm120Stages::S3,
    };
    let mut request = case.request(physical);
    request.a_ptr = 0;
    assert!(validate_sm120_map_request(request).is_err());
    assert!(prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    request = case.request(physical);
    request.b_ptr += 1;
    assert!(validate_sm120_map_request(request).is_err());
    assert!(prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    request = case.request(physical);
    request.shape.lda += 1;
    assert!(validate_sm120_map_request(request).is_err());
    assert!(prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, request).is_err());

    let caps = ctx.kernels.sm120_device_caps().expect("SM120 caps");
    let target = ctx.kernels.sm120_target_candidate().expect("SM120 target");
    for axis in 0..3 {
        let mut invalid = forced_route(ctx, &case, physical);
        match axis {
            0 => invalid.shape.m = 0,
            1 => invalid.shape.k = 0,
            _ => invalid.shape.n = 0,
        }
        assert!(resolve_sm120_forced(caps, Some(target), invalid).is_err());
    }
    let requested = forced_route(ctx, &case, physical);
    assert_eq!(
        resolve_sm120_forced(caps, None, requested).expect("fail-closed route result"),
        None
    );
    assert!(SM120_AUTO_CELLS_CC120.is_empty());
    assert!(SM120_AUTO_CELLS_CC121.is_empty());
}

fn assert_replay_rejects_allocation_generation_change(ctx: &GpuCtx) {
    let case = CensusCase::new(ctx, Sm120Op::Nn, WeightDtype::Bf16, ASYMMETRIC);
    let physical = Sm120PhysicalRoute {
        tile: Sm120Tile::M128N64,
        bk: Sm120Bk::Bk32,
        stages: Sm120Stages::S2,
    };
    let route = forced_route(ctx, &case, physical);
    let maps = prepare_sm120_tensor_maps(&ctx.stream, &ctx.kernels, case.request(physical))
        .expect("prepare allocation-generation maps");
    let mut output = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    output.reset(&ctx.stream, &case.initial);
    let prepared = prepare_sm120_tma_forced(
        &ctx.stream,
        &ctx.kernels,
        route,
        &maps,
        case.operands(output.ptr(), false),
    )
    .expect("prepare allocation-generation route");
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared).map(|_| ())
        })
    }
    .expect("capture allocation-generation route");
    drop(output);
    let replacement = GuardedOutput::new(ctx, case.op, case.dtype, case.initial.len(), 0);
    let error = validate_sm120_graph_replay(&ctx.stream, &ctx.kernels, &prepared)
        .expect_err("freed or recycled output allocation must invalidate graph replay");
    assert!(error.contains("allocation"), "{error}");
    drop(replacement);
    drop(graph);
}

fn sm120_context() -> Option<(GpuDevice, GpuCtx)> {
    let device = match GpuDevice::new(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping SM120 triad census without a CUDA device: {error}");
            return None;
        }
    };
    if !matches!(device.compute_capability, (12, 0) | (12, 1)) {
        eprintln!(
            "skipping SM120 triad census on compute capability {}.{}",
            device.compute_capability.0, device.compute_capability.1
        );
        return None;
    }
    let ctx = GpuCtx::new(&device).expect("SM120 GPU context");
    assert!(
        ctx.kernels.has_sm120_tma_mma16(),
        "SM120 detected but the complete generic TMA/MMA16 module failed qualification"
    );
    Some((device, ctx))
}

#[test]
#[ignore = "needs a real CC 12.0 or CC 12.1 device"]
fn qualifies_all_96_forced_sm120_routes() {
    let Some((_device, ctx)) = sm120_context() else {
        return;
    };
    assert_eq!(SM120_KERNEL_SPECS.len(), 96);
    assert!(SM120_AUTO_CELLS_CC120.is_empty());
    assert!(SM120_AUTO_CELLS_CC121.is_empty());

    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let baseline = CensusCase::compact(&ctx, op, dtype, ALIGNED_BASELINE);
            for physical in physical_routes() {
                assert_exact_baseline(&ctx, &baseline, physical);
            }

            let mut exceptional = CensusCase::compact(&ctx, op, dtype, (64, 64, 64));
            exceptional.replace_with_adversarial_inputs(&ctx.stream);
            for physical in physical_routes() {
                assert_exact_baseline(&ctx, &exceptional, physical);
            }

            for dims in independent_axis_shapes() {
                let case = CensusCase::new(&ctx, op, dtype, dims);
                for (index, physical) in physical_routes().into_iter().enumerate() {
                    run_candidate(&ctx, &case, physical, 0, index % 8);
                }
            }

            for logical_offset in 1..=7 {
                let case = CensusCase::with_origins(
                    &ctx,
                    op,
                    dtype,
                    ASYMMETRIC,
                    (logical_offset, 1),
                    (8 - logical_offset, 2),
                    false,
                );
                assert!(!case.request(physical_routes()[0]).a_ptr.is_multiple_of(128));
                assert!(!case.request(physical_routes()[0]).b_ptr.is_multiple_of(128));
                for physical in physical_routes() {
                    run_candidate(&ctx, &case, physical, 0, logical_offset);
                }
            }

            let repeated = CensusCase::new(&ctx, op, dtype, ASYMMETRIC);
            for physical in physical_routes() {
                run_candidate(&ctx, &repeated, physical, EAGER_REPEATS, 3);
                run_graph_route(&ctx, &repeated, physical);
            }
        }
    }

    assert_multi_stream_concurrency(&ctx);
    assert_semantic_rejections(&ctx);
    assert_map_and_route_fail_closed(&ctx);
    assert_replay_rejects_allocation_generation_change(&ctx);
}

#[test]
#[ignore = "run under compute-sanitizer on a real CC 12.0 and CC 12.1 device"]
fn sm120_stage_wrap_tail_sanitizer_target() {
    let Some((_device, ctx)) = sm120_context() else {
        return;
    };
    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for physical in physical_routes() {
                let reduction = match physical.bk {
                    Sm120Bk::Bk32 => match physical.stages {
                        Sm120Stages::S2 => 81,
                        Sm120Stages::S3 => 113,
                    },
                    Sm120Bk::Bk64 => match physical.stages {
                        Sm120Stages::S2 => 145,
                        Sm120Stages::S3 => 209,
                    },
                };
                let dims = match op {
                    Sm120Op::Nn => (65, reduction, 129),
                    Sm120Op::Tn => (reduction, 65, 129),
                    Sm120Op::Nt => (65, 129, reduction),
                };
                let case = CensusCase::with_origins(&ctx, op, dtype, dims, (3, 1), (5, 2), false);
                run_candidate(&ctx, &case, physical, 2, 7);
            }
        }
    }
}
