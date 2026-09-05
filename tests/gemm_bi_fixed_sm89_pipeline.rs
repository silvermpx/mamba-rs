//! Actual NVRTC Fixed half pipeline admission, arithmetic and launch gates.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaGraph, sys};
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedTile, fixed_forward, fixed_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

const CANDIDATE: FixedTile = FixedTile::Tc128Sm89Pipeline;
const RUNGS: [FixedTile; 5] = [
    FixedTile::Tc16,
    FixedTile::Tc64,
    FixedTile::Tc128,
    FixedTile::TcW64,
    FixedTile::TcWn64,
];

#[test]
#[ignore = "requires exact Ada 142SM CUDA13.2 and qualified hot-cell AUTO promotion"]
fn fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (hot_m, k, n) in [
            (4621, 384, 1928),
            (4621, 768, 2304),
            (4621, 1928, 384),
            (2048, 768, 2304),
            (2048, 2304, 768),
        ] {
            let rows = hot_m + 18;
            for exceptional in [false, true] {
                let mut a_host: Vec<_> = (0..rows * k)
                    .map(|i| half_bits(((i * 7 % 19) as f32 - 9.0) * 0.03125, dtype))
                    .collect();
                let mut b_host: Vec<_> = (0..k * n)
                    .map(|i| half_bits(((i * 11 % 23) as f32 - 11.0) * 0.03125, dtype))
                    .collect();
                let mut bias_host: Vec<_> =
                    (0..n).map(|i| (i % 7) as f32 * 0.015625 - 0.0625).collect();
                if exceptional {
                    let specials = match dtype {
                        WeightDtype::Bf16 => [0, 0x8000, 0x7f80, 0xff80, 0x7f81, 0xffff],
                        WeightDtype::F16 => [0, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0xffff],
                        WeightDtype::F32 => unreachable!(),
                    };
                    for r in 0..rows {
                        a_host[r * k] = specials[r % specials.len()];
                    }
                    for column in 0..n {
                        b_host[k / 2 * n + column] = specials[column % specials.len()];
                    }
                    bias_host[0] = f32::from_bits(0x7f80_0001);
                    bias_host[1] = f32::NEG_INFINITY;
                }
                let a = upload_half(&ctx, &a_host);
                let b = upload_half(&ctx, &b_host);
                let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias upload");
                for has_bias in [false, true] {
                    let full = upload_half(&ctx, &vec![0x7fff; rows * n]);
                    let operands = FixedFwdOperands {
                        c: typed(full.cached_ptr(), dtype),
                        x: typed(a.cached_ptr(), dtype),
                        w: typed(b.cached_ptr(), dtype),
                        bias_ptr: has_bias.then_some(bias.cached_ptr()),
                    };
                    fixed_forward_with_tile(
                        &ctx,
                        operands,
                        FixedShape { m: rows, k, n },
                        FixedTile::Tc128,
                    )
                    .expect("full incumbent reference outside AUTO cells");
                    let reference = raw_half(&ctx, &full);
                    for m in [hot_m, hot_m - 1, hot_m + 1, 1, 17, 129] {
                        for (row_offset, output_offset) in [(0, 8), (17, 8), (17, 1)] {
                            let initial = vec![0x7fff; output_offset + m * n + 9];
                            let mut output = upload_half(&ctx, &initial);
                            let view = FixedFwdOperands {
                                c: typed(output.cached_ptr() + (output_offset * 2) as u64, dtype),
                                x: typed(a.cached_ptr() + (row_offset * k * 2) as u64, dtype),
                                ..operands
                            };
                            let run = || {
                                fixed_forward(
                                    &ctx,
                                    view.c,
                                    view.x,
                                    view.w,
                                    view.bias_ptr,
                                    (m, k, n),
                                )
                            };
                            let picked = run().expect("actual AUTO launch");
                            assert_eq!(
                                picked == CANDIDATE,
                                m == hot_m && output_offset == 8,
                                "AUTO promotion scope {dtype:?} M={m} K={k} N={n} row={row_offset} out={output_offset} bias={has_bias}: {picked:?}"
                            );
                            let mut expected = initial.clone();
                            expected[output_offset..output_offset + m * n]
                                .copy_from_slice(&reference[row_offset * n..(row_offset + m) * n]);
                            assert!(
                                raw_half(&ctx, &output) == expected,
                                "AUTO prefix/view/guard bits"
                            );
                            let graph =
                                unsafe { capture_into_graph(&ctx.stream, || run().map(|_| ())) }
                                    .expect("capture actual AUTO");
                            if picked == CANDIDATE {
                                assert_pipeline_graph(&graph, dtype);
                            }
                            for _ in 0..2 {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison graph output");
                                graph.launch().expect("AUTO graph replay");
                                assert!(
                                    raw_half(&ctx, &output) == expected,
                                    "AUTO graph bits/guards"
                                );
                            }
                        }
                    }
                }
                assert!(raw_half(&ctx, &a) == a_host, "A changed");
                assert!(raw_half(&ctx, &b) == b_host, "B changed");
            }
        }
    }
}

#[test]
#[ignore = "requires exact Ada CC8.9 and the admitted Fixed half pipeline"]
fn fixed_sm89_half_pipeline_rejects_unsafe_operands_and_dimensions() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    let dt = WeightDtype::Bf16;
    let a = upload_half(&ctx, &vec![0; 17 * 65]);
    let b = upload_half(&ctx, &vec![0; 65 * 131]);
    let c = upload_half(&ctx, &vec![0x7fff; 17 * 131]);
    let good = FixedFwdOperands {
        c: typed(c.cached_ptr(), dt),
        x: typed(a.cached_ptr(), dt),
        w: typed(b.cached_ptr(), dt),
        bias_ptr: None,
    };
    let shape = FixedShape {
        m: 17,
        k: 65,
        n: 131,
    };
    fixed_forward_with_tile(&ctx, good, shape, CANDIDATE).expect("odd-stride positive control");
    for bad in [
        FixedFwdOperands {
            x: typed(0, dt),
            ..good
        },
        FixedFwdOperands {
            w: typed(0, dt),
            ..good
        },
        FixedFwdOperands {
            c: typed(0, dt),
            ..good
        },
        FixedFwdOperands {
            x: typed(a.cached_ptr() + 1, dt),
            ..good
        },
        FixedFwdOperands {
            w: typed(b.cached_ptr() + 1, dt),
            ..good
        },
        FixedFwdOperands {
            c: typed(c.cached_ptr() + 1, dt),
            ..good
        },
        FixedFwdOperands {
            bias_ptr: Some(1),
            ..good
        },
        FixedFwdOperands {
            c: typed(c.cached_ptr(), WeightDtype::F32),
            ..good
        },
        FixedFwdOperands {
            w: typed(b.cached_ptr(), WeightDtype::F16),
            ..good
        },
    ] {
        assert!(
            fixed_forward_with_tile(&ctx, bad, shape, CANDIDATE).is_err(),
            "unsafe operands admitted"
        );
    }
    for bad in [
        FixedShape {
            m: i32::MAX as usize,
            ..shape
        },
        FixedShape {
            n: i32::MAX as usize,
            ..shape
        },
        FixedShape {
            k: i32::MAX as usize,
            ..shape
        },
        FixedShape {
            m: 1 << 20,
            n: 1 << 29,
            ..shape
        },
        FixedShape {
            m: i32::MAX as usize + 1,
            ..shape
        },
    ] {
        assert!(
            fixed_forward_with_tile(&ctx, good, bad, CANDIDATE).is_err(),
            "unsafe shape admitted: {bad:?}"
        );
    }
    let empty = FixedFwdOperands {
        c: typed(0, dt),
        x: typed(0, dt),
        w: typed(0, dt),
        bias_ptr: None,
    };
    for no_output in [FixedShape { m: 0, ..shape }, FixedShape { n: 0, ..shape }] {
        fixed_forward_with_tile(&ctx, empty, no_output, CANDIDATE)
            .expect("empty output is a no-op");
    }
}

fn upload_half(ctx: &GpuCtx, bits: &[u16]) -> GpuByteBuffer {
    let mut buffer = GpuByteBuffer::zeros(&ctx.stream, bits.len() * 2).expect("allocate half");
    buffer
        .upload_bytes(&ctx.stream, bytemuck::cast_slice(bits))
        .expect("upload exact half bits");
    ctx.stream
        .synchronize()
        .expect("complete half upload before host slice can drop");
    buffer
}

fn raw_half(ctx: &GpuCtx, buffer: &GpuByteBuffer) -> Vec<u16> {
    let mut bytes = vec![0; buffer.len_bytes()];
    ctx.stream
        .memcpy_dtoh(buffer.inner(), &mut bytes)
        .expect("download raw half bytes");
    // Pageable host storage does not make Async copies synchronously safe.
    ctx.stream
        .synchronize()
        .expect("complete raw half readback");
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect()
}

fn half_bits(value: f32, dtype: WeightDtype) -> u16 {
    match dtype {
        WeightDtype::Bf16 => half::bf16::from_f32(value).to_bits(),
        WeightDtype::F16 => half::f16::from_f32(value).to_bits(),
        WeightDtype::F32 => unreachable!(),
    }
}

fn typed(ptr: u64, dtype: WeightDtype) -> TypedPtr {
    TypedPtr { ptr, dtype }
}

fn assert_pipeline_graph(graph: &CudaGraph, dtype: WeightDtype) {
    let mut count = 0;
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        count, 1,
        "pipeline must capture one kernel and no workspace node"
    );
    let mut node = std::ptr::null_mut();
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut params = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut name = std::ptr::null();
    assert_eq!(
        unsafe { sys::cuFuncGetName(&mut name, params.func) },
        sys::CUresult::CUDA_SUCCESS
    );
    let expected = format!("gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_{}", dtype.as_str());
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes(),
        expected.as_bytes()
    );
    assert_eq!(
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        (256, 1, 1)
    );
    assert_eq!(params.sharedMemBytes, 71_680);
}

#[test]
#[ignore = "requires exact Ada CC8.9; cold NVRTC loaded symbol, not an NVCC experiment"]
fn fixed_sm89_half_pipeline_forced_cross_rung_prefix_view_graph_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9), "this gate requires Ada");
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    let rows = 2066;
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        // K192/N136 is aligned and reuses the asynchronous two-stage ring.
        // K65/N131 separately exercises scalar staging and masked output.
        for (k, n) in [(64, 136), (192, 136), (65, 131), (0, 131)] {
            for corpus in 0..4 {
                let mut a_bits: Vec<_> = (0..rows * k)
                    .map(|i| half_bits(((i * 7 % 19) as f32 - 9.0) * 0.03125, dtype))
                    .collect();
                let mut b_bits: Vec<_> = (0..k * n)
                    .map(|i| half_bits(((i * 11 % 23) as f32 - 11.0) * 0.03125, dtype))
                    .collect();
                let special: [u16; 8] = match dtype {
                    WeightDtype::Bf16 => {
                        [0, 0x8000, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fff, 0xffff]
                    }
                    WeightDtype::F16 => [0, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7fff, 0xffff],
                    WeightDtype::F32 => unreachable!(),
                };
                if k != 0 && corpus == 1 {
                    for row in 0..rows {
                        a_bits[row * k] = special[row % special.len()];
                    }
                }
                if k != 0 && corpus == 2 {
                    for column in 0..n {
                        b_bits[column] = special[column % special.len()];
                    }
                }
                let bias_host: Vec<_> = (0..n)
                    .map(|i| {
                        if corpus == 3 {
                            f32::from_bits(
                                [
                                    0,
                                    0x8000_0000,
                                    0x7f80_0000,
                                    0xff80_0000,
                                    0x7f80_0001,
                                    0xffff_ffff,
                                ][i % 6],
                            )
                        } else {
                            (i % 7) as f32 * 0.015625 - 0.0625
                        }
                    })
                    .collect();
                // Keep one allocated element at K0 but pass actual null inputs.
                let a = upload_half(&ctx, if a_bits.is_empty() { &[0] } else { &a_bits });
                let b = upload_half(&ctx, if b_bits.is_empty() { &[0] } else { &b_bits });
                let a_shifted_bits: Vec<_> = std::iter::once(0x7fff)
                    .chain(a_bits.iter().copied())
                    .collect();
                let b_shifted_bits: Vec<_> = std::iter::once(0x7fff)
                    .chain(b_bits.iter().copied())
                    .collect();
                let a_shifted = upload_half(&ctx, &a_shifted_bits);
                let b_shifted = upload_half(&ctx, &b_shifted_bits);
                let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias upload");
                for has_bias in [false, true] {
                    let full = upload_half(&ctx, &vec![0x7fff; rows * n]);
                    let operands = FixedFwdOperands {
                        c: typed(full.cached_ptr(), dtype),
                        x: typed(if k == 0 { 0 } else { a.cached_ptr() }, dtype),
                        w: typed(if k == 0 { 0 } else { b.cached_ptr() }, dtype),
                        bias_ptr: has_bias.then_some(bias.cached_ptr()),
                    };
                    fixed_forward_with_tile(
                        &ctx,
                        operands,
                        FixedShape { m: rows, k, n },
                        FixedTile::Tc128,
                    )
                    .expect("incumbent full reference");
                    let reference = raw_half(&ctx, &full);
                    for m in [
                        1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2047,
                        2048, 2049,
                    ] {
                        for (row_offset, output_offset, shift_a, shift_b) in [
                            (0, 8, false, false),
                            (17, 1, false, false),
                            (17, 8, true, false),
                            (17, 8, false, true),
                        ] {
                            let initial = vec![0x7fff; output_offset + m * n + 9];
                            let mut output = upload_half(&ctx, &initial);
                            let view = FixedFwdOperands {
                                c: typed(output.cached_ptr() + (output_offset * 2) as u64, dtype),
                                x: typed(
                                    if k == 0 {
                                        0
                                    } else {
                                        (if shift_a {
                                            a_shifted.cached_ptr() + 2
                                        } else {
                                            a.cached_ptr()
                                        }) + (row_offset * k * 2) as u64
                                    },
                                    dtype,
                                ),
                                w: typed(
                                    if k == 0 {
                                        0
                                    } else if shift_b {
                                        b_shifted.cached_ptr() + 2
                                    } else {
                                        b.cached_ptr()
                                    },
                                    dtype,
                                ),
                                ..operands
                            };
                            let shape = FixedShape { m, k, n };
                            let mut expected = initial.clone();
                            expected[output_offset..output_offset + m * n]
                                .copy_from_slice(&reference[row_offset * n..(row_offset + m) * n]);
                            for tile in std::iter::once(CANDIDATE).chain(RUNGS) {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison C");
                                fixed_forward_with_tile(&ctx, view, shape, tile).unwrap_or_else(
                                    |e| panic!("{tile:?} {dtype:?} M={m} K={k} N={n}: {e}"),
                                );
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    expected,
                                    "rung/prefix/view/guard bits: {tile:?} {dtype:?} M={m} K={k} N={n} row={row_offset} bias={has_bias} corpus={corpus}"
                                );
                            }
                            let run = || fixed_forward_with_tile(&ctx, view, shape, CANDIDATE);
                            for _ in 0..2 {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison repeat");
                                run().expect("eager repeat");
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    expected,
                                    "eager repeat changed bits"
                                );
                            }
                            let graph = unsafe { capture_into_graph(&ctx.stream, run) }
                                .expect("capture candidate");
                            assert_pipeline_graph(&graph, dtype);
                            for _ in 0..2 {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison replay");
                                graph.launch().expect("graph replay");
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    expected,
                                    "graph replay changed bits"
                                );
                            }
                        }
                    }
                }
                if k != 0 {
                    assert_eq!(raw_half(&ctx, &a), a_bits, "A changed");
                    assert_eq!(raw_half(&ctx, &b), b_bits, "B changed");
                }
                assert_eq!(
                    raw_half(&ctx, &a_shifted),
                    a_shifted_bits,
                    "shifted A changed"
                );
                assert_eq!(
                    raw_half(&ctx, &b_shifted),
                    b_shifted_bits,
                    "shifted B changed"
                );
                let bias_readback = bias.to_cpu(&ctx.stream).expect("bias download");
                ctx.stream.synchronize().expect("complete bias readback");
                assert_eq!(
                    bias_readback
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    bias_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    "bias changed"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires exact Ada CC8.9; bounded actual-NVRTC smoke for all four sanitizer tools"]
fn fixed_sm89_half_pipeline_sanitizer_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9), "this gate requires Ada");
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    // 16 incumbent launches + 128 executed candidate launches = 144 GEMMs.
    // Another 32 candidate calls only enqueue capture nodes, not executions.
    // Module admission probes are outside this test-body count.
    const HALF_GUARD: usize = 8;
    const BIAS_GUARD: usize = 4;
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n, exceptional) in [
            (129, 192, 136, false), // aligned cp.async, three slabs, M/N tails
            (17, 65, 131, false),   // scalar staging and masked scalar output
            (17, 0, 131, false),    // actual null A/B, with and without bias
            (17, 192, 136, true),   // one exceptional A/B/bias corpus
        ] {
            let rows = m + 1;
            let mut a_host = vec![0x7fff; HALF_GUARD + rows * k + HALF_GUARD];
            let mut b_host = vec![0x7fff; HALF_GUARD + k * n + HALF_GUARD];
            let mut bias_host = vec![f32::from_bits(0x7fc0_1234); BIAS_GUARD + n + BIAS_GUARD];
            for i in 0..rows * k {
                a_host[HALF_GUARD + i] = half_bits(((i * 7 % 19) as f32 - 9.0) * 0.03125, dtype);
            }
            for i in 0..k * n {
                b_host[HALF_GUARD + i] = half_bits(((i * 11 % 23) as f32 - 11.0) * 0.03125, dtype);
            }
            for i in 0..n {
                bias_host[BIAS_GUARD + i] = (i % 7) as f32 * 0.015625 - 0.0625;
            }
            if exceptional {
                let special = match dtype {
                    WeightDtype::Bf16 => {
                        [0, 0x8000, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fff, 0xffff]
                    }
                    WeightDtype::F16 => [0, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7fff, 0xffff],
                    WeightDtype::F32 => unreachable!(),
                };
                for row in 0..rows {
                    a_host[HALF_GUARD + row * k] = special[row % special.len()];
                }
                for column in 0..n {
                    b_host[HALF_GUARD + k / 2 * n + column] = special[column % special.len()];
                }
                bias_host[BIAS_GUARD] = f32::from_bits(0x7f80_0001);
                bias_host[BIAS_GUARD + 1] = f32::NEG_INFINITY;
            }
            let a = upload_half(&ctx, &a_host);
            let b = upload_half(&ctx, &b_host);
            let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("guarded bias upload");
            for has_bias in [false, true] {
                let full_initial = vec![0x7fff; HALF_GUARD + rows * n + HALF_GUARD];
                let full = upload_half(&ctx, &full_initial);
                let operands = FixedFwdOperands {
                    c: typed(full.cached_ptr() + (HALF_GUARD * 2) as u64, dtype),
                    x: typed(
                        if k == 0 {
                            0
                        } else {
                            a.cached_ptr() + (HALF_GUARD * 2) as u64
                        },
                        dtype,
                    ),
                    w: typed(
                        if k == 0 {
                            0
                        } else {
                            b.cached_ptr() + (HALF_GUARD * 2) as u64
                        },
                        dtype,
                    ),
                    bias_ptr: has_bias.then_some(bias.cached_ptr() + (BIAS_GUARD * 4) as u64),
                };
                fixed_forward_with_tile(
                    &ctx,
                    operands,
                    FixedShape { m: rows, k, n },
                    FixedTile::Tc128,
                )
                .expect("bounded incumbent full reference");
                let reference = raw_half(&ctx, &full);
                assert_eq!(&reference[..HALF_GUARD], &full_initial[..HALF_GUARD]);
                assert_eq!(
                    &reference[HALF_GUARD + rows * n..],
                    &full_initial[HALF_GUARD + rows * n..],
                    "incumbent output suffix guard"
                );
                for (row_offset, output_offset) in [(0, HALF_GUARD), (1, 1)] {
                    let initial = vec![0x7fff; output_offset + m * n + HALF_GUARD];
                    let mut output = upload_half(&ctx, &initial);
                    let view = FixedFwdOperands {
                        c: typed(output.cached_ptr() + (output_offset * 2) as u64, dtype),
                        x: typed(
                            if k == 0 {
                                0
                            } else {
                                operands.x.ptr + (row_offset * k * 2) as u64
                            },
                            dtype,
                        ),
                        ..operands
                    };
                    let shape = FixedShape { m, k, n };
                    let mut expected = initial.clone();
                    let reference_start = HALF_GUARD + row_offset * n;
                    expected[output_offset..output_offset + m * n]
                        .copy_from_slice(&reference[reference_start..reference_start + m * n]);
                    let run = || fixed_forward_with_tile(&ctx, view, shape, CANDIDATE);
                    for _ in 0..2 {
                        output
                            .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                            .expect("poison sanitizer eager output");
                        run().expect("forced NVRTC pipeline eager");
                        assert_eq!(
                            raw_half(&ctx, &output),
                            expected,
                            "sanitizer eager bits/guards {dtype:?} {shape:?} row={row_offset} C_offset={output_offset} bias={has_bias} exceptional={exceptional}"
                        );
                    }
                    let graph = unsafe { capture_into_graph(&ctx.stream, run) }
                        .expect("capture actual forced NVRTC pipeline");
                    assert_pipeline_graph(&graph, dtype);
                    for _ in 0..2 {
                        output
                            .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                            .expect("poison sanitizer graph output");
                        graph.launch().expect("sanitizer graph replay");
                        assert_eq!(
                            raw_half(&ctx, &output),
                            expected,
                            "sanitizer graph bits/guards {dtype:?} {shape:?} row={row_offset} C_offset={output_offset} bias={has_bias} exceptional={exceptional}"
                        );
                    }
                    assert_eq!(raw_half(&ctx, &a), a_host, "A bits/guards changed");
                    assert_eq!(raw_half(&ctx, &b), b_host, "B bits/guards changed");
                    let bias_readback = bias.to_cpu(&ctx.stream).expect("guarded bias download");
                    ctx.stream
                        .synchronize()
                        .expect("complete guarded bias readback");
                    assert_eq!(
                        bias_readback
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        bias_host
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        "bias bits/guards changed"
                    );
                }
            }
        }
    }
}
