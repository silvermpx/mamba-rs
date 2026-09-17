use super::*;
use crate::mamba_ssm::gpu::blas::gpu_gemm_bi_forward_raw;
use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::graph_capture::{capture_into_graph, capture_into_graph_with_gemm_plan};
use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
    CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, PhysicalGemmBackend, PolicyDtype,
    RecordedGemmTrace, ResolvedInstructionFamily, ResolvedNumericContract,
    ResolvedOperandConversion, ResolvedOutputOwnership, SCHEDULE_REVISION,
};
use std::collections::HashMap;

const BIG_DYNAMIC_SHARED_BYTES: u32 = 34 * 1024;
const GUARD_FLOATS: usize = 16;
const GUARD_WORDS: [u32; 4] = [0x7fc1_2345, 0xff81_3579, 0x8000_0000, 0x5a5a_a5a5];
const EXCEPTION_WORDS: [u32; 8] = [
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc1_2345,
    0x7f81_2345,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaunchPath {
    ForcedBig,
    Production,
}

#[derive(Clone, Copy)]
struct Epilogue {
    alpha: f32,
    beta: f32,
    bias: bool,
}

#[derive(Clone, Copy)]
struct RuntimeCase {
    label: &'static str,
    op: ResolvedGemmOp,
    shape: F32TriadShape,
    epilogue: Epilogue,
    path: LaunchPath,
    parity: Option<&'static str>,
    exceptional: bool,
    poison_output: bool,
}

#[derive(Clone, Copy)]
struct MatrixLayout {
    rows: usize,
    width: usize,
    stride: usize,
    offset: usize,
}

struct HostMatrix {
    layout: MatrixLayout,
    values: Vec<f32>,
}

#[derive(Clone, Copy)]
enum ExpectedWord {
    Exact(u32),
    Nan,
}

struct HostCase {
    spec: RuntimeCase,
    a: HostMatrix,
    b: HostMatrix,
    bias: Option<HostMatrix>,
    output: HostMatrix,
    expected: Vec<ExpectedWord>,
}

struct DeviceCase {
    a: GpuBuffer,
    b: GpuBuffer,
    bias: Option<GpuBuffer>,
    output: GpuBuffer,
}

const fn shape(m: usize, k: usize, n: usize, lda: usize, ldb: usize, ldc: usize) -> F32TriadShape {
    F32TriadShape {
        m,
        k,
        n,
        lda,
        ldb,
        ldc,
    }
}

fn runtime_cases() -> [RuntimeCase; 14] {
    [
        RuntimeCase {
            label: "nn-aligned-forced-overwrite",
            op: ResolvedGemmOp::Nn,
            shape: shape(512, 16, 640, 16, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: false,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("nn-aligned-overwrite"),
            exceptional: false,
            poison_output: true,
        },
        RuntimeCase {
            label: "nn-aligned-production-overwrite",
            op: ResolvedGemmOp::Nn,
            shape: shape(512, 16, 640, 16, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: false,
            },
            path: LaunchPath::Production,
            parity: Some("nn-aligned-overwrite"),
            exceptional: false,
            poison_output: true,
        },
        RuntimeCase {
            label: "nn-aligned-forced-bias-beta",
            op: ResolvedGemmOp::Nn,
            shape: shape(512, 16, 640, 16, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: -0.5,
                bias: true,
            },
            path: LaunchPath::ForcedBig,
            parity: None,
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "nn-steady-forced-bias",
            op: ResolvedGemmOp::Nn,
            shape: shape(512, 33, 641, 33, 641, 641),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: true,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("nn-steady-bias"),
            exceptional: false,
            poison_output: true,
        },
        RuntimeCase {
            label: "nn-steady-production-bias",
            op: ResolvedGemmOp::Nn,
            shape: shape(512, 33, 641, 33, 641, 641),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: true,
            },
            path: LaunchPath::Production,
            parity: Some("nn-steady-bias"),
            exceptional: false,
            poison_output: true,
        },
        RuntimeCase {
            label: "nn-tail-forced-padded-exceptional",
            op: ResolvedGemmOp::Nn,
            shape: shape(513, 17, 513, 19, 517, 519),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: true,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("nn-tail-exceptional"),
            exceptional: true,
            poison_output: false,
        },
        RuntimeCase {
            label: "nn-tail-production-exceptional",
            op: ResolvedGemmOp::Nn,
            shape: shape(513, 17, 513, 17, 513, 513),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: true,
            },
            path: LaunchPath::Production,
            parity: Some("nn-tail-exceptional"),
            exceptional: true,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-aligned-forced-accumulate",
            op: ResolvedGemmOp::Tn,
            shape: shape(16, 512, 640, 512, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("tn-aligned-accumulate"),
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-aligned-production-accumulate",
            op: ResolvedGemmOp::Tn,
            shape: shape(16, 512, 640, 512, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::Production,
            parity: Some("tn-aligned-accumulate"),
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-aligned-forced-scaled-accumulate",
            op: ResolvedGemmOp::Tn,
            shape: shape(16, 512, 640, 512, 640, 640),
            epilogue: Epilogue {
                alpha: -0.75,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::ForcedBig,
            parity: None,
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-steady-forced-accumulate",
            op: ResolvedGemmOp::Tn,
            shape: shape(33, 512, 640, 512, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("tn-steady-accumulate"),
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-steady-production-accumulate",
            op: ResolvedGemmOp::Tn,
            shape: shape(33, 512, 640, 512, 640, 640),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::Production,
            parity: Some("tn-steady-accumulate"),
            exceptional: false,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-tail-forced-exceptional",
            op: ResolvedGemmOp::Tn,
            shape: shape(17, 513, 513, 513, 513, 513),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::ForcedBig,
            parity: Some("tn-tail-exceptional"),
            exceptional: true,
            poison_output: false,
        },
        RuntimeCase {
            label: "tn-tail-production-exceptional",
            op: ResolvedGemmOp::Tn,
            shape: shape(17, 513, 513, 513, 513, 513),
            epilogue: Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: false,
            },
            path: LaunchPath::Production,
            parity: Some("tn-tail-exceptional"),
            exceptional: true,
            poison_output: false,
        },
    ]
}

fn matrix_layouts(spec: RuntimeCase) -> [MatrixLayout; 3] {
    let shape = spec.shape;
    let forced_offset = usize::from(spec.path == LaunchPath::ForcedBig) * 4;
    match spec.op {
        ResolvedGemmOp::Nn => [
            MatrixLayout {
                rows: shape.m,
                width: shape.k,
                stride: shape.lda,
                offset: if spec.path == LaunchPath::Production {
                    0
                } else {
                    forced_offset
                },
            },
            MatrixLayout {
                rows: shape.k,
                width: shape.n,
                stride: shape.ldb,
                offset: forced_offset,
            },
            MatrixLayout {
                rows: shape.m,
                width: shape.n,
                stride: shape.ldc,
                offset: if spec.path == LaunchPath::Production {
                    0
                } else {
                    forced_offset
                },
            },
        ],
        ResolvedGemmOp::Tn => [
            MatrixLayout {
                rows: shape.m,
                width: shape.k,
                stride: shape.lda,
                offset: forced_offset,
            },
            MatrixLayout {
                rows: shape.m,
                width: shape.n,
                stride: shape.ldb,
                offset: forced_offset,
            },
            MatrixLayout {
                rows: shape.k,
                width: shape.n,
                stride: shape.ldc,
                offset: 4,
            },
        ],
        ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
    }
}

fn guard_word(index: usize) -> u32 {
    GUARD_WORDS[index % GUARD_WORDS.len()]
}

fn finite_value(index: usize, seed: u32) -> f32 {
    let mut state = seed.wrapping_add(u32::try_from(index).unwrap_or(u32::MAX));
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let signed = ((state >> 16) & 63) as i32 - 31;
    signed as f32 * (1.0 / 32.0)
}

fn host_matrix(layout: MatrixLayout, seed: u32, active_zero: bool) -> HostMatrix {
    let active_end = layout.offset + (layout.rows - 1) * layout.stride + layout.width;
    let mut values = (0..active_end + GUARD_FLOATS)
        .map(|index| f32::from_bits(guard_word(index)))
        .collect::<Vec<_>>();
    for row in 0..layout.rows {
        for column in 0..layout.width {
            let logical = row * layout.width + column;
            values[layout.offset + row * layout.stride + column] = if active_zero {
                0.0
            } else {
                finite_value(logical, seed)
            };
        }
    }
    HostMatrix { layout, values }
}

impl HostMatrix {
    fn get(&self, row: usize, column: usize) -> f32 {
        self.values[self.layout.offset + row * self.layout.stride + column]
    }

    fn set(&mut self, row: usize, column: usize, value: f32) {
        let index = self.layout.offset + row * self.layout.stride + column;
        self.values[index] = value;
    }

    fn fill_active(&mut self, value: f32) {
        for row in 0..self.layout.rows {
            for column in 0..self.layout.width {
                self.set(row, column, value);
            }
        }
    }
}

impl HostCase {
    fn new(spec: RuntimeCase) -> Self {
        spec.shape
            .validate(spec.op)
            .expect("valid runtime case shape");
        let [a_layout, b_layout, output_layout] = matrix_layouts(spec);
        let mut a = host_matrix(a_layout, 0x1234_5678, spec.exceptional);
        let mut b = host_matrix(b_layout, 0x9abc_def0, spec.exceptional);
        let mut output = host_matrix(output_layout, 0x6d2b_79f5, false);
        if spec.poison_output {
            output.fill_active(f32::from_bits(0x7fc5_a55a));
        }
        let mut bias = spec.epilogue.bias.then(|| {
            host_matrix(
                MatrixLayout {
                    rows: 1,
                    width: spec.shape.n,
                    stride: spec.shape.n,
                    offset: 4,
                },
                0x0bad_f00d,
                spec.exceptional,
            )
        });
        if spec.exceptional {
            Self::inject_exceptional_values(spec.op, &mut a, &mut b, &mut output);
            if let Some(bias) = bias.as_mut() {
                bias.fill_active(0.0);
            }
        }
        let expected = Self::oracle(spec, &a, &b, bias.as_ref(), &output);
        Self {
            spec,
            a,
            b,
            bias,
            output,
            expected,
        }
    }

    fn inject_exceptional_values(
        op: ResolvedGemmOp,
        a: &mut HostMatrix,
        b: &mut HostMatrix,
        output: &mut HostMatrix,
    ) {
        match op {
            ResolvedGemmOp::Nn => {
                for (index, bits) in EXCEPTION_WORDS.into_iter().enumerate() {
                    a.set(0, index, 1.0);
                    b.set(index, index, f32::from_bits(bits));
                    output.set(0, index, 0.0);
                }
            }
            ResolvedGemmOp::Tn => {
                for (index, bits) in EXCEPTION_WORDS.into_iter().enumerate() {
                    a.set(index, index, f32::from_bits(bits));
                    b.set(index, index, 1.0);
                    output.set(index, index, 0.0);
                }
            }
            ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
        }
    }

    fn oracle(
        spec: RuntimeCase,
        a: &HostMatrix,
        b: &HostMatrix,
        bias: Option<&HostMatrix>,
        output: &HostMatrix,
    ) -> Vec<ExpectedWord> {
        let rows = spec.shape.output_rows(spec.op);
        let columns = spec.shape.output_columns(spec.op);
        let mut expected = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = match spec.op {
                    ResolvedGemmOp::Nn => bias.map_or(0.0, |bias| bias.get(0, column)),
                    ResolvedGemmOp::Tn => 0.0,
                    ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
                };
                match spec.op {
                    ResolvedGemmOp::Nn => {
                        for reduction in 0..spec.shape.k {
                            accumulator = a
                                .get(row, reduction)
                                .mul_add(b.get(reduction, column), accumulator);
                        }
                    }
                    ResolvedGemmOp::Tn => {
                        for reduction in 0..spec.shape.m {
                            accumulator = a
                                .get(reduction, row)
                                .mul_add(b.get(reduction, column), accumulator);
                        }
                    }
                    ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
                }
                let old = output.get(row, column);
                let value = match spec.op {
                    ResolvedGemmOp::Nn => {
                        let scaled = spec.epilogue.alpha * accumulator;
                        if spec.epilogue.beta == 0.0 {
                            scaled
                        } else {
                            spec.epilogue.beta.mul_add(old, scaled)
                        }
                    }
                    ResolvedGemmOp::Tn => spec.epilogue.alpha.mul_add(accumulator, old),
                    ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
                };
                expected.push(if value.is_nan() {
                    ExpectedWord::Nan
                } else {
                    ExpectedWord::Exact(value.to_bits())
                });
            }
        }
        expected
    }
}

impl DeviceCase {
    fn new(ctx: &GpuCtx, host: &HostCase) -> Self {
        let a = GpuBuffer::from_cpu(&ctx.stream, &host.a.values).expect("allocate guarded A");
        let b = GpuBuffer::from_cpu(&ctx.stream, &host.b.values).expect("allocate guarded B");
        let bias = host
            .bias
            .as_ref()
            .map(|bias| GpuBuffer::from_cpu(&ctx.stream, &bias.values).expect("allocate bias"));
        let output =
            GpuBuffer::from_cpu(&ctx.stream, &host.output.values).expect("allocate output");
        ctx.stream
            .synchronize()
            .expect("finish NN/TN runtime allocations");
        Self { a, b, bias, output }
    }

    fn reset_output(&mut self, ctx: &GpuCtx, host: &HostCase) {
        self.output
            .upload(&ctx.stream, &host.output.values)
            .expect("restore runtime output");
    }

    fn matrix_ptr(ctx: &GpuCtx, buffer: &GpuBuffer, matrix: &HostMatrix) -> CUptr {
        buffer.raw_ptr_at(&ctx.stream, matrix.layout.offset)
    }

    fn operands(&self, ctx: &GpuCtx, host: &HostCase) -> F32TriadOperands {
        F32TriadOperands {
            output: Self::matrix_ptr(ctx, &self.output, &host.output),
            a: Self::matrix_ptr(ctx, &self.a, &host.a),
            b: Self::matrix_ptr(ctx, &self.b, &host.b),
            bias: self
                .bias
                .as_ref()
                .zip(host.bias.as_ref())
                .map(|(buffer, matrix)| Self::matrix_ptr(ctx, buffer, matrix)),
            alpha: host.spec.epilogue.alpha,
            beta: host.spec.epilogue.beta,
        }
    }
}

fn checked_i32(value: usize, name: &str) -> i32 {
    i32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds i32"))
}

fn big_grid(spec: RuntimeCase) -> u32 {
    let rows = spec.shape.output_rows(spec.op).div_ceil(128);
    let columns = spec.shape.output_columns(spec.op).div_ceil(128);
    u32::try_from(rows.checked_mul(columns).expect("Big grid overflow"))
        .expect("Big grid exceeds u32")
}

unsafe fn launch_forced_big(
    ctx: &GpuCtx,
    host: &HostCase,
    device: &mut DeviceCase,
) -> Result<(), String> {
    let operands = device.operands(ctx, host);
    let shape = host.spec.shape;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (big_grid(host.spec), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: BIG_DYNAMIC_SHARED_BYTES,
    };
    match host.spec.op {
        ResolvedGemmOp::Nn => {
            let bias = operands.bias.unwrap_or(0);
            let m = checked_i32(shape.m, "M");
            let n = checked_i32(shape.n, "N");
            let k = checked_i32(shape.k, "K");
            let lda = checked_i32(shape.lda, "lda");
            let ldb = checked_i32(shape.ldb, "ldb");
            let ldc = checked_i32(shape.ldc, "ldc");
            let mut launch = ctx
                .stream
                .launch_builder(&ctx.kernels.triad_kernels().gemm_bi_nn);
            launch.arg(&operands.output);
            launch.arg(&operands.a);
            launch.arg(&operands.b);
            launch.arg(&bias);
            launch.arg(&operands.alpha);
            launch.arg(&operands.beta);
            launch.arg(&m);
            launch.arg(&n);
            launch.arg(&k);
            launch.arg(&lda);
            launch.arg(&ldb);
            launch.arg(&ldc);
            unsafe { launch.launch(config) }
                .map_err(|error| format!("forced Big NN: {error:?}"))?;
        }
        ResolvedGemmOp::Tn => {
            assert_eq!(shape.lda, shape.k, "forced TN requires contiguous A");
            assert_eq!(shape.ldb, shape.n, "forced TN requires contiguous B");
            assert_eq!(shape.ldc, shape.n, "forced TN requires contiguous output");
            let m = checked_i32(shape.m, "M");
            let k = checked_i32(shape.k, "K");
            let n = checked_i32(shape.n, "N");
            let mut launch = ctx
                .stream
                .launch_builder(&ctx.kernels.triad_kernels().gemm_bi_tn);
            launch.arg(&operands.output);
            launch.arg(&operands.a);
            launch.arg(&operands.b);
            launch.arg(&operands.alpha);
            launch.arg(&m);
            launch.arg(&k);
            launch.arg(&n);
            unsafe { launch.launch(config) }
                .map_err(|error| format!("forced Big TN: {error:?}"))?;
        }
        ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
    }
    Ok(())
}

fn launch_production(ctx: &GpuCtx, host: &HostCase, device: &mut DeviceCase) -> Result<(), String> {
    let shape = host.spec.shape;
    assert_eq!(
        shape,
        F32TriadShape::contiguous(host.spec.op, (shape.m, shape.k, shape.n)),
        "production qualification requires contiguous operands"
    );
    match host.spec.op {
        ResolvedGemmOp::Nn => {
            assert_eq!(host.spec.epilogue.alpha, 1.0);
            assert_eq!(host.spec.epilogue.beta, 0.0);
            let bias = device
                .bias
                .as_ref()
                .zip(host.bias.as_ref())
                .map(|(buffer, matrix)| DeviceCase::matrix_ptr(ctx, buffer, matrix));
            gpu_gemm_bi_forward_raw(
                ctx,
                &mut device.output,
                &device.a,
                DeviceCase::matrix_ptr(ctx, &device.b, &host.b),
                bias,
                (shape.m, shape.k, shape.n),
            )
        }
        ResolvedGemmOp::Tn => {
            assert_eq!(host.spec.epilogue.alpha, 1.0);
            assert_eq!(host.spec.epilogue.beta, 1.0);
            launch_cached_f32_backward_dw(
                ctx,
                DeviceCase::matrix_ptr(ctx, &device.output, &host.output),
                &device.b,
                &device.a,
                (shape.m, shape.k, shape.n),
            )
        }
        ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
    }
}

fn launch_case(ctx: &GpuCtx, host: &HostCase, device: &mut DeviceCase) -> Result<(), String> {
    match host.spec.path {
        LaunchPath::ForcedBig => unsafe { launch_forced_big(ctx, host, device) },
        LaunchPath::Production => launch_production(ctx, host, device),
    }
}

fn active_bits(matrix: &HostMatrix, actual: &[u32]) -> Vec<u32> {
    let mut active = Vec::with_capacity(matrix.layout.rows * matrix.layout.width);
    for row in 0..matrix.layout.rows {
        let start = matrix.layout.offset + row * matrix.layout.stride;
        active.extend_from_slice(&actual[start..start + matrix.layout.width]);
    }
    active
}

fn assert_output(host: &HostCase, actual: &[u32]) -> Vec<u32> {
    assert_eq!(
        actual.len(),
        host.output.values.len(),
        "{}",
        host.spec.label
    );
    let active = active_bits(&host.output, actual);
    for (index, (&actual, expected)) in active.iter().zip(&host.expected).enumerate() {
        match expected {
            ExpectedWord::Exact(expected) => {
                assert_eq!(actual, *expected, "{} output {index}", host.spec.label);
            }
            ExpectedWord::Nan => {
                assert!(
                    f32::from_bits(actual).is_nan(),
                    "{} output {index} must be NaN",
                    host.spec.label
                );
            }
        }
    }
    let layout = host.output.layout;
    for (index, (&actual, initial)) in actual
        .iter()
        .zip(host.output.values.iter().map(|value| value.to_bits()))
        .enumerate()
    {
        let relative = index.checked_sub(layout.offset);
        let inside = relative.is_some_and(|relative| {
            let row = relative / layout.stride;
            let column = relative % layout.stride;
            row < layout.rows && column < layout.width
        });
        if !inside {
            assert_eq!(actual, initial, "{} output guard {index}", host.spec.label);
        }
    }
    active
}

fn assert_input_storage(ctx: &GpuCtx, host: &HostCase, device: &DeviceCase) {
    for (label, expected, actual) in [
        (
            "A",
            host.a.values.as_slice(),
            device.a.to_cpu(&ctx.stream).expect("download guarded A"),
        ),
        (
            "B",
            host.b.values.as_slice(),
            device.b.to_cpu(&ctx.stream).expect("download guarded B"),
        ),
    ] {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} {label} storage changed",
            host.spec.label
        );
    }
    if let (Some(expected), Some(actual)) = (&host.bias, &device.bias) {
        assert_eq!(
            actual
                .to_cpu(&ctx.stream)
                .expect("download guarded bias")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} bias storage changed",
            host.spec.label
        );
    }
}

fn assert_big_route(host: &HostCase, trace: &RecordedGemmTrace) {
    let routes = trace.routes();
    assert_eq!(routes.len(), 1, "{} route count", host.spec.label);
    let route = routes[0];
    let expected_symbol = match host.spec.op {
        ResolvedGemmOp::Nn => "nn_big",
        ResolvedGemmOp::Tn => "tn_aligned",
        ResolvedGemmOp::Nt => unreachable!("NN/TN qualification received NT"),
    };
    assert_eq!(route.op, host.spec.op);
    assert_eq!(route.dtype, PolicyDtype::F32);
    assert_eq!(route.backend, PhysicalGemmBackend::ScalarFma);
    assert_eq!(route.numeric_contract, ResolvedNumericContract::ScalarFma);
    assert_eq!(
        route.instruction_family,
        ResolvedInstructionFamily::ScalarFma
    );
    assert_eq!(route.operand_conversion, ResolvedOperandConversion::None);
    assert_eq!(
        route.ownership,
        ResolvedOutputOwnership::OneCtaPerOutputTile
    );
    assert_eq!(route.symbol, expected_symbol);
    assert_eq!(route.module_kind, ModuleKind::TriadScalar);
    assert_eq!(
        route.shape,
        (host.spec.shape.m, host.spec.shape.k, host.spec.shape.n)
    );
    assert_eq!(
        route.strides,
        (
            host.spec.shape.lda,
            host.spec.shape.ldb,
            host.spec.shape.ldc
        )
    );
    assert_eq!(route.tile, (128, 128));
    assert_eq!(route.bk, 16);
    assert_eq!(route.stages, 2);
    assert_eq!(route.threads, 256);
    assert_eq!(route.launch.grid_dim, (big_grid(host.spec), 1, 1));
    assert_eq!(route.launch.block_dim, (256, 1, 1));
    assert_eq!(route.launch.shared_mem_bytes, BIG_DYNAMIC_SHARED_BYTES);
    assert_eq!(
        trace
            .launches()
            .expect("production route launch set")
            .launch_count,
        1
    );
}

fn eager_runs(
    ctx: &GpuCtx,
    host: &HostCase,
    device: &mut DeviceCase,
) -> (Vec<u32>, Option<RecordedGemmTrace>) {
    let mut baseline = None;
    let mut trace = None;
    for iteration in 0..3 {
        device.reset_output(ctx, host);
        if iteration == 0 && host.spec.path == LaunchPath::Production {
            trace = Some(
                ctx.record_eager_gemm_trace(|| launch_case(ctx, host, device))
                    .expect("record production Big route"),
            );
        } else {
            launch_case(ctx, host, device).expect("launch eager NN/TN case");
        }
        ctx.stream.synchronize().expect("finish eager NN/TN case");
        let output = device
            .output
            .to_cpu(&ctx.stream)
            .expect("download eager NN/TN output")
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let active = assert_output(host, &output);
        if let Some(baseline) = &baseline {
            assert_eq!(active, *baseline, "{} eager {iteration}", host.spec.label);
        } else {
            baseline = Some(active);
        }
    }
    (baseline.expect("three eager outputs"), trace)
}

fn graph_runs(
    ctx: &GpuCtx,
    host: &HostCase,
    device: &mut DeviceCase,
    eager: &[u32],
    trace: Option<&RecordedGemmTrace>,
) {
    device.reset_output(ctx, host);
    if let Some(trace) = trace {
        assert_big_route(host, trace);
        let manifest = trace.manifest();
        let (graph, plan) = unsafe {
            capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, &manifest, || {
                launch_case(ctx, host, device)
            })
        }
        .expect("capture production NN/TN graph");
        let plan = plan.expect("production graph route plan");
        assert_eq!(plan.routes(), trace.routes());
        assert_eq!(
            plan.launches,
            trace.launches().expect("eager production launch set")
        );
        for iteration in 0..3 {
            device.reset_output(ctx, host);
            plan.with_validated_launch(ctx, host.spec.label, || {
                graph
                    .launch()
                    .map_err(|error| format!("launch production graph: {error:?}"))
            })
            .expect("validate and replay production graph");
            ctx.stream
                .synchronize()
                .expect("finish production graph replay");
            let output = device
                .output
                .to_cpu(&ctx.stream)
                .expect("download production graph output")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                assert_output(host, &output),
                eager,
                "{} graph {iteration}",
                host.spec.label
            );
        }
    } else {
        let graph = unsafe { capture_into_graph(&ctx.stream, || launch_case(ctx, host, device)) }
            .expect("capture forced Big NN/TN graph");
        for iteration in 0..3 {
            device.reset_output(ctx, host);
            graph.launch().expect("replay forced Big NN/TN graph");
            ctx.stream
                .synchronize()
                .expect("finish forced Big graph replay");
            let output = device
                .output
                .to_cpu(&ctx.stream)
                .expect("download forced Big graph output")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                assert_output(host, &output),
                eager,
                "{} graph {iteration}",
                host.spec.label
            );
        }
    }
}

fn run_runtime_case(ctx: &GpuCtx, spec: RuntimeCase) -> Vec<u32> {
    let host = HostCase::new(spec);
    let mut device = DeviceCase::new(ctx, &host);
    let (eager, trace) = eager_runs(ctx, &host, &mut device);
    graph_runs(ctx, &host, &mut device, &eager, trace.as_ref());
    assert_input_storage(ctx, &host, &device);
    eager
}

#[test]
fn runtime_production_shapes_resolve_to_single_big_routes() {
    for multiprocessor_count in [20, 56, 82, 108, 120, 128, 132, 142, 148, 170] {
        for (op, dims, expected) in [
            (
                ResolvedGemmOp::Nn,
                (512, 16, 640),
                ScalarDispatchPlan::NnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Nn,
                (512, 33, 641),
                ScalarDispatchPlan::NnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Nn,
                (513, 17, 513),
                ScalarDispatchPlan::NnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Tn,
                (16, 512, 640),
                ScalarDispatchPlan::TnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Tn,
                (33, 512, 640),
                ScalarDispatchPlan::TnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Tn,
                (17, 513, 513),
                ScalarDispatchPlan::TnFinal { slim: false },
            ),
        ] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, dims),
            };
            let plan = scalar_dispatch_plan(request, multiprocessor_count)
                .expect("resolve production runtime shape");
            assert_eq!(plan, expected, "{op:?} {dims:?} {multiprocessor_count} SMs");
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: (op == ResolvedGemmOp::Nn).then_some(0x4000),
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            let nodes = scalar_physical_nodes(request, operands, plan).expect("build Big route");
            assert_eq!(nodes.len(), 1);
            assert_eq!(
                nodes[0].symbol,
                if op == ResolvedGemmOp::Nn {
                    "nn_big"
                } else {
                    "tn_aligned"
                }
            );
            assert_eq!(
                nodes[0].launch.grid_dim.0,
                big_grid(RuntimeCase {
                    label: "route",
                    op,
                    shape: request.shape,
                    epilogue: Epilogue {
                        alpha: operands.alpha,
                        beta: operands.beta,
                        bias: operands.bias.is_some(),
                    },
                    path: LaunchPath::Production,
                    parity: None,
                    exceptional: false,
                    poison_output: false,
                })
            );
        }
    }
}

fn qualified_tn_m16n16_facts() -> ScalarLaunchFacts {
    let target = CudaTarget::new("compute_120").expect("valid scalar qualification target");
    ScalarLaunchFacts {
        scalar_artifact: ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [4; 32],
            artifact_digest: [5; 32],
        },
        scalar_compiler: CompilerIdentity {
            source_digest: [3; 32],
            invocation_digest: [4; 32],
            header_manifest_digest: [6; 32],
            target,
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [7; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        },
        fixed_artifact: ArtifactIdentity {
            module_kind: ModuleKind::Fixed,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [0; 32],
            artifact_digest: [0; 32],
        },
        fixed_compiler: CompilerIdentity {
            source_digest: [0; 32],
            invocation_digest: [0; 32],
            header_manifest_digest: [0; 32],
            target,
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [0; 32],
            nvrtc_library_known: false,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        },
        fixed_copyplan_loaded: false,
        sm89_exact_f32_artifact: None,
        sm89_exact_f32_compiler: None,
        sm89_exact_f32_symbols_loaded: [false; 3],
        sm89_exact_f32_d128_artifact: None,
        sm89_exact_f32_d128_compiler: None,
        sm89_exact_f32_d128_symbols_loaded: [false; 2],
        compute_capability: (12, 0),
        multiprocessor_count: 170,
    }
}

fn qualified_tn_m16n16_request() -> F32TriadRequest {
    F32TriadRequest {
        op: ResolvedGemmOp::Tn,
        shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (256, 512, 384)),
    }
}

fn qualified_tn_m16n16_operands() -> F32TriadOperands {
    F32TriadOperands {
        output: 0x1000,
        a: 0x2000,
        b: 0x3000,
        bias: None,
        alpha: 1.0,
        beta: 1.0,
    }
}

#[test]
fn tn_m16n16_selector_is_exact_and_fail_closed() {
    let facts = qualified_tn_m16n16_facts();
    let request = qualified_tn_m16n16_request();
    let operands = qualified_tn_m16n16_operands();
    let fallback = ScalarDispatchPlan::TnSplitM {
        m_chunk: 16,
        chunks: 16,
    };
    assert_eq!(
        scalar_launch_plan(facts, request, operands).unwrap(),
        ScalarDispatchPlan::TnM16N16SplitM16Qualified
    );

    let dimensions = [request.shape.m, request.shape.k, request.shape.n];
    for axis in 0..3 {
        for delta in [-1_isize, 1] {
            let mut neighbor = dimensions;
            neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
            let mutated = F32TriadRequest {
                op: ResolvedGemmOp::Tn,
                shape: F32TriadShape::contiguous(
                    ResolvedGemmOp::Tn,
                    (neighbor[0], neighbor[1], neighbor[2]),
                ),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
    }

    for shape in [
        F32TriadShape {
            lda: request.shape.lda + 1,
            ..request.shape
        },
        F32TriadShape {
            ldb: request.shape.ldb + 1,
            ..request.shape
        },
        F32TriadShape {
            ldc: request.shape.ldc + 1,
            ..request.shape
        },
    ] {
        let mutated = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape,
        };
        assert_ne!(
            scalar_launch_plan(facts, mutated, operands).unwrap(),
            ScalarDispatchPlan::TnM16N16SplitM16Qualified
        );
    }

    let mut old_toolkit = facts;
    old_toolkit.scalar_compiler.nvrtc_version = (13, 0);
    assert_eq!(
        scalar_launch_plan(old_toolkit, request, operands).unwrap(),
        fallback
    );

    let mut wrong_arch = facts;
    wrong_arch.compute_capability = (8, 9);
    assert_eq!(
        scalar_launch_plan(wrong_arch, request, operands).unwrap(),
        fallback
    );

    let mut wrong_sm_count = facts;
    wrong_sm_count.multiprocessor_count = 169;
    assert_ne!(
        scalar_launch_plan(wrong_sm_count, request, operands).unwrap(),
        ScalarDispatchPlan::TnM16N16SplitM16Qualified
    );

    let mut wrong_compile_key = facts;
    wrong_compile_key.scalar_artifact.compile_key[0] ^= 1;
    assert_eq!(
        scalar_launch_plan(wrong_compile_key, request, operands).unwrap(),
        fallback
    );

    let mut unknown_library = facts;
    unknown_library.scalar_compiler.nvrtc_library_known = false;
    assert_eq!(
        scalar_launch_plan(unknown_library, request, operands).unwrap(),
        fallback
    );

    for rejected in [
        F32TriadOperands {
            bias: Some(0x4000),
            ..operands
        },
        F32TriadOperands {
            alpha: -1.0,
            ..operands
        },
        F32TriadOperands {
            beta: 0.0,
            ..operands
        },
        F32TriadOperands {
            output: 0,
            ..operands
        },
        F32TriadOperands {
            a: operands.a + 4,
            ..operands
        },
        F32TriadOperands {
            b: operands.b + 8,
            ..operands
        },
    ] {
        assert_eq!(
            scalar_launch_plan(facts, request, rejected).unwrap(),
            fallback
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA GPU"]
fn scalar_big_nn_tn_runtime_matrix_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(device.compute_capability >= (8, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_gemm_mode(crate::mamba_ssm::gpu::GemmMode::Deterministic)
        .unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_bi_tensor_cores(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);

    let mut forced_outputs = HashMap::new();
    for spec in runtime_cases() {
        let output = run_runtime_case(&ctx, spec);
        if let Some(parity) = spec.parity {
            match spec.path {
                LaunchPath::ForcedBig => {
                    assert!(forced_outputs.insert(parity, output).is_none());
                }
                LaunchPath::Production => {
                    assert_eq!(
                        output,
                        forced_outputs
                            .remove(parity)
                            .expect("forced parity output must run first"),
                        "{parity} forced and production bits"
                    );
                }
            }
        }
    }
    assert!(forced_outputs.is_empty(), "unmatched forced parity outputs");
}
