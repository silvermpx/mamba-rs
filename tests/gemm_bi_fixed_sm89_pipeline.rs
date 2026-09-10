//! Actual NVRTC Fixed half pipeline admission, arithmetic and launch gates.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaGraph, sys};
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceTile, inference_forward,
    inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::TUNING_TABLE_REVISION;

const CANDIDATE: InferenceTile = InferenceTile::Tc128Sm89Pipeline;
const SWIZZLE_CANDIDATE: InferenceTile = InferenceTile::Tc128Sm89Swizzle;
const S3_CANDIDATE: InferenceTile = InferenceTile::Tc128Sm89S3;
const D_FINALIST: InferenceTile = InferenceTile::TcM64N64Sm89S3;
const E_FINALIST: InferenceTile = InferenceTile::TcM128N64Sm89S2;
const RUNGS: [InferenceTile; 5] = [
    InferenceTile::Tc16,
    InferenceTile::Tc64,
    InferenceTile::Tc128,
    InferenceTile::TcW64,
    InferenceTile::TcWn64,
];

fn expected_ada_half_auto_v45(
    nvrtc: (i32, i32),
    dtype: WeightDtype,
    shape: InferenceShape,
    has_bias: bool,
) -> Option<InferenceTile> {
    use InferenceTile::{
        Tc128Sm89Pipeline as Pipeline, Tc128Sm89S3 as S3, Tc128Sm89Swizzle as Swizzle,
    };

    match (nvrtc, dtype, (shape.m, shape.k, shape.n), has_bias) {
        ((13, 2), WeightDtype::F16, (2048, 768, 2304), false) => Some(D_FINALIST),
        ((13, 2), WeightDtype::F16, (2048, 2304, 768), false) => Some(E_FINALIST),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), false) => Some(Pipeline),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), true) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 1928, 384), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (2048, 2304, 768), _) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), _) => Some(Pipeline),
        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false) => Some(S3),
        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), true) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Pipeline),
        ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
        ((13, 2), WeightDtype::F16, (4621, 768, 2304), false) => Some(S3),
        ((13, 2), WeightDtype::F16, (4621, 768, 2304), true) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (4621, 1928, 384), _) => Some(Pipeline),
        ((13, 2), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (2048, 2304, 768), _) => Some(Pipeline),
        _ => None,
    }
}

#[test]
fn fixed_sm89_half_swizzle_has_a_distinct_forced_route() {
    assert_ne!(SWIZZLE_CANDIDATE, CANDIDATE);
    assert_eq!(format!("{SWIZZLE_CANDIDATE:?}"), "Tc128Sm89Swizzle");
}

#[test]
fn fixed_sm89_half_s3_has_a_distinct_forced_route() {
    assert_ne!(S3_CANDIDATE, CANDIDATE);
    assert_ne!(S3_CANDIDATE, SWIZZLE_CANDIDATE);
    assert_eq!(format!("{S3_CANDIDATE:?}"), "Tc128Sm89S3");
}

#[test]
#[ignore = "requires exact Ada CC8.9 and live independent half holders"]
fn fixed_sm89_half_swizzle_and_pipeline_holders_are_independently_live() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    println!(
        "S3_MODULE_IDENTITIES compiler={:?} artifacts={:?}",
        ctx.kernels.compiler_identity(),
        ctx.kernels.artifact_set_identity()
    );
    assert!(ctx.kernels.fixed_sm89_half_pipeline.is_some());
    assert!(ctx.kernels.fixed_sm89_half_pipeline_rejection.is_none());
    let swizzle = ctx
        .kernels
        .fixed_sm89_half_swizzle
        .as_ref()
        .unwrap_or_else(|| {
            panic!(
                "swizzle holder rejected: {:?}",
                ctx.kernels.fixed_sm89_half_swizzle_rejection
            )
        });
    assert!(ctx.kernels.fixed_sm89_half_swizzle_rejection.is_none());
    for (dtype, function) in [
        (WeightDtype::Bf16, &swizzle.bf16),
        (WeightDtype::F16, &swizzle.f16),
    ] {
        let local = function.local_size_bytes().expect("local bytes");
        let registers = function.num_regs().expect("registers");
        let static_shared = function.shared_size_bytes().expect("static shared");
        let max_threads = function.max_threads_per_block().expect("max threads");
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(256, 69_632, None)
            .expect("occupancy");
        assert_eq!(local, 0);
        assert_eq!(static_shared, 0);
        assert!((1..=224).contains(&registers));
        assert!(max_threads >= 256);
        assert!(occupancy >= 1);
        println!(
            "SWIZZLE_RESOURCE dtype={} local={} registers={} static_shared={} max_threads={} active_blocks={} dynamic_shared=69632",
            dtype.as_str(),
            local,
            registers,
            static_shared,
            max_threads,
            occupancy
        );
    }
    let s3 = ctx.kernels.fixed_sm89_half_s3.as_ref().unwrap_or_else(|| {
        panic!(
            "s3 holder rejected: {:?}",
            ctx.kernels.fixed_sm89_half_s3_rejection
        )
    });
    assert!(ctx.kernels.fixed_sm89_half_s3_rejection.is_none());
    for (dtype, function) in [(WeightDtype::Bf16, &s3.bf16), (WeightDtype::F16, &s3.f16)] {
        let local = function.local_size_bytes().expect("local bytes");
        let registers = function.num_regs().expect("registers");
        let static_shared = function.shared_size_bytes().expect("static shared");
        let max_threads = function.max_threads_per_block().expect("max threads");
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(256, 98_304, None)
            .expect("occupancy");
        assert_eq!(local, 0);
        assert_eq!(static_shared, 0);
        assert!((1..=188).contains(&registers));
        assert!(max_threads >= 256);
        assert!(occupancy >= 1);
        println!(
            "S3_RESOURCE dtype={} local={} registers={} static_shared={} max_threads={} active_blocks={} dynamic_shared=98304",
            dtype.as_str(),
            local,
            registers,
            static_shared,
            max_threads,
            occupancy
        );
    }
    for (label, function, rejection, register_cap, threads, shared, active_blocks) in [
        (
            "rna_n96",
            ctx.kernels.fixed_sm89_tf32_rna_n96.as_ref(),
            &ctx.kernels.fixed_sm89_tf32_rna_n96_rejection,
            136,
            256,
            86_016,
            1,
        ),
        (
            "half_m64n64_s3_f16",
            ctx.kernels.fixed_sm89_half_m64n64_s3_f16.as_ref(),
            &ctx.kernels.fixed_sm89_half_m64n64_s3_f16_rejection,
            110,
            128,
            49_152,
            2,
        ),
        (
            "half_m128n64_s2_f16",
            ctx.kernels.fixed_sm89_half_m128n64_s2_f16.as_ref(),
            &ctx.kernels.fixed_sm89_half_m128n64_s2_f16_rejection,
            132,
            128,
            49_152,
            2,
        ),
    ] {
        let function = function.unwrap_or_else(|| panic!("{label} rejected: {rejection:?}"));
        assert!(rejection.is_none());
        let local = function.local_size_bytes().expect("local bytes");
        let registers = function.num_regs().expect("registers");
        let static_shared = function.shared_size_bytes().expect("static shared");
        let max_threads = function.max_threads_per_block().expect("max threads");
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)
            .expect("occupancy");
        assert_eq!(local, 0);
        assert_eq!(static_shared, 0);
        assert!((1..=register_cap).contains(&registers));
        assert!(max_threads >= threads as i32);
        assert_eq!(occupancy, active_blocks);
        println!(
            "FINALIST_RESOURCE label={label} local={local} registers={registers} static_shared={static_shared} max_threads={max_threads} active_blocks={occupancy} dynamic_shared={shared}"
        );
    }
}

#[test]
#[ignore = "requires exact Ada 142SM CUDA12.8/13.0/13.2 and qualified hot-cell AUTO promotion"]
fn fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits() {
    fixed_sm89_half_hot_cell_prefix_view_graph_bits(None, None);
}

#[test]
#[ignore = "requires exact Ada CC8.9 and the admitted forced half swizzle"]
fn fixed_sm89_half_swizzle_forced_hot_a_e_prefix_view_graph_bits() {
    fixed_sm89_half_hot_cell_prefix_view_graph_bits(Some(SWIZZLE_CANDIDATE), None);
}

#[test]
#[ignore = "requires exact Ada CC8.9 and the admitted forced half s3"]
fn fixed_sm89_half_s3_forced_hot_a_e_prefix_view_graph_bits() {
    fixed_sm89_half_hot_cell_prefix_view_graph_bits(Some(S3_CANDIDATE), None);
}

#[test]
#[ignore = "requires exact Ada CC8.9 and the admitted forced F16 D finalist"]
fn fixed_sm89_half_d_finalist_prefix_view_graph_bits() {
    fixed_sm89_half_hot_cell_prefix_view_graph_bits(Some(D_FINALIST), Some((768, 2304)));
}

#[test]
#[ignore = "requires exact Ada CC8.9 and the admitted forced F16 E finalist"]
fn fixed_sm89_half_e_finalist_prefix_view_graph_bits() {
    fixed_sm89_half_hot_cell_prefix_view_graph_bits(Some(E_FINALIST), Some((2304, 768)));
}

#[test]
#[ignore = "requires exact Ada CC8.9; finalist launchers must reject before touching tiny buffers"]
fn fixed_sm89_half_finalists_fail_closed_outside_exact_forced_contract() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    let a = upload_half(&ctx, &[0; 8]);
    let b = upload_half(&ctx, &[0; 8]);
    let c = upload_half(&ctx, &[0x7fff; 8]);
    let good = InferenceFwdOperands {
        c: typed(c.cached_ptr(), WeightDtype::F16),
        x: typed(a.cached_ptr(), WeightDtype::F16),
        w: typed(b.cached_ptr(), WeightDtype::F16),
        bias_ptr: None,
    };
    for (tile, shape) in [
        (
            D_FINALIST,
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
        ),
        (
            E_FINALIST,
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ),
    ] {
        for bad in [
            InferenceFwdOperands {
                c: typed(c.cached_ptr() + 2, WeightDtype::F16),
                ..good
            },
            InferenceFwdOperands {
                x: typed(a.cached_ptr() + 2, WeightDtype::F16),
                ..good
            },
            InferenceFwdOperands {
                w: typed(b.cached_ptr() + 2, WeightDtype::F16),
                ..good
            },
            InferenceFwdOperands {
                bias_ptr: Some(4),
                ..good
            },
            InferenceFwdOperands {
                c: typed(c.cached_ptr(), WeightDtype::Bf16),
                x: typed(a.cached_ptr(), WeightDtype::Bf16),
                w: typed(b.cached_ptr(), WeightDtype::Bf16),
                ..good
            },
        ] {
            assert!(
                inference_forward_with_tile(&ctx, bad, shape, tile).is_err(),
                "{tile:?} admitted an unsafe dtype/pointer/bias contract"
            );
        }
        for bad_shape in [
            InferenceShape { m: 0, ..shape },
            InferenceShape { m: 2049, ..shape },
            InferenceShape {
                k: shape.k - 64,
                ..shape
            },
            InferenceShape {
                n: shape.n - 64,
                ..shape
            },
        ] {
            assert!(
                inference_forward_with_tile(&ctx, good, bad_shape, tile).is_err(),
                "{tile:?} admitted out-of-contract shape {bad_shape:?}"
            );
        }
    }
}

fn fixed_sm89_half_hot_cell_prefix_view_graph_bits(
    forced: Option<InferenceTile>,
    finalist: Option<(usize, usize)>,
) {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    if forced.is_none() {
        let compiler = ctx.kernels.compiler_identity();
        assert_eq!(TUNING_TABLE_REVISION, 45);
        assert!(compiler.nvrtc_library_known);
        assert!(matches!(
            compiler.nvrtc_version,
            (12, 8) | (13, 0) | (13, 2)
        ));
        assert!(ctx.kernels.fixed_sm89_half_pipeline.is_some());
        assert!(ctx.kernels.fixed_sm89_half_pipeline_rejection.is_none());
        assert!(ctx.kernels.fixed_sm89_half_swizzle.is_some());
        assert!(ctx.kernels.fixed_sm89_half_swizzle_rejection.is_none());
    }
    let dtypes = if finalist.is_some() {
        vec![WeightDtype::F16]
    } else {
        vec![WeightDtype::Bf16, WeightDtype::F16]
    };
    let shapes = finalist.map_or_else(
        || {
            vec![
                (4621, 384, 1928),
                (4621, 768, 2304),
                (4621, 1928, 384),
                (2048, 768, 2304),
                (2048, 2304, 768),
            ]
        },
        |(k, n)| vec![(2048, k, n)],
    );
    for dtype in dtypes {
        for &(hot_m, k, n) in &shapes {
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
                let bias_states: &[bool] = if finalist.is_some() {
                    &[false]
                } else {
                    &[false, true]
                };
                for &has_bias in bias_states {
                    let full = upload_half(&ctx, &vec![0x7fff; rows * n]);
                    let operands = InferenceFwdOperands {
                        c: typed(full.cached_ptr(), dtype),
                        x: typed(a.cached_ptr(), dtype),
                        w: typed(b.cached_ptr(), dtype),
                        bias_ptr: has_bias.then_some(bias.cached_ptr()),
                    };
                    inference_forward_with_tile(
                        &ctx,
                        operands,
                        InferenceShape { m: rows, k, n },
                        InferenceTile::Tc128,
                    )
                    .expect("full incumbent reference outside AUTO cells");
                    let reference = raw_half(&ctx, &full);
                    let prefixes = if finalist.is_some() {
                        vec![hot_m, hot_m - 1, 1, 17, 129]
                    } else {
                        vec![hot_m, hot_m - 1, hot_m + 1, 1, 17, 129]
                    };
                    let views: &[(usize, usize)] = if finalist.is_some() {
                        &[(0, 8), (17, 8)]
                    } else {
                        &[(0, 8), (17, 8), (17, 1)]
                    };
                    for m in prefixes {
                        for &(row_offset, output_offset) in views {
                            let guards = vec![0x7fff; output_offset + m * n + 9];
                            let mut expected = guards.clone();
                            expected[output_offset..output_offset + m * n]
                                .copy_from_slice(&reference[row_offset * n..(row_offset + m) * n]);
                            let mut initial = guards;
                            for (poison, gold) in initial[output_offset..output_offset + m * n]
                                .iter_mut()
                                .zip(&expected[output_offset..output_offset + m * n])
                            {
                                *poison = *gold ^ 0xffff;
                                assert_ne!(*poison, *gold);
                            }
                            let mut output = upload_half(&ctx, &initial);
                            assert_eq!(raw_half(&ctx, &output), initial, "initial poison readback");
                            let view = InferenceFwdOperands {
                                c: typed(output.cached_ptr() + (output_offset * 2) as u64, dtype),
                                x: typed(a.cached_ptr() + (row_offset * k * 2) as u64, dtype),
                                ..operands
                            };
                            let run = || -> Result<InferenceTile, String> {
                                if let Some(tile) = forced {
                                    inference_forward_with_tile(
                                        &ctx,
                                        view,
                                        InferenceShape { m, k, n },
                                        tile,
                                    )?;
                                    Ok(tile)
                                } else {
                                    inference_forward(
                                        &ctx,
                                        view.c,
                                        view.x,
                                        view.w,
                                        view.bias_ptr,
                                        (m, k, n),
                                    )
                                }
                            };
                            let picked = run().expect("hot-cell launch");
                            if forced.is_none() {
                                if m == hot_m && output_offset == 8 {
                                    let expected = expected_ada_half_auto_v45(
                                        ctx.kernels.compiler_identity().nvrtc_version,
                                        dtype,
                                        InferenceShape { m, k, n },
                                        has_bias,
                                    )
                                    .expect("literal revision-45 hot-cell expectation");
                                    assert_eq!(
                                        picked, expected,
                                        "AUTO promotion scope {dtype:?} M={m} K={k} N={n} row={row_offset} out={output_offset} bias={has_bias}"
                                    );
                                } else {
                                    assert!(
                                        !matches!(
                                            picked,
                                            CANDIDATE
                                                | SWIZZLE_CANDIDATE
                                                | S3_CANDIDATE
                                                | D_FINALIST
                                                | E_FINALIST
                                        ),
                                        "AUTO candidate escaped scope {dtype:?} M={m} K={k} N={n} row={row_offset} out={output_offset} bias={has_bias}: {picked:?}"
                                    );
                                }
                            } else {
                                assert_eq!(picked, forced.unwrap());
                            }
                            assert!(
                                raw_half(&ctx, &output) == expected,
                                "AUTO prefix/view/guard bits"
                            );
                            let graph =
                                unsafe { capture_into_graph(&ctx.stream, || run().map(|_| ())) }
                                    .expect("capture actual AUTO");
                            if matches!(
                                picked,
                                CANDIDATE
                                    | SWIZZLE_CANDIDATE
                                    | S3_CANDIDATE
                                    | D_FINALIST
                                    | E_FINALIST
                            ) {
                                assert_half_graph(
                                    &graph,
                                    picked,
                                    dtype,
                                    view,
                                    InferenceShape { m, k, n },
                                );
                            }
                            for _ in 0..2 {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison graph output");
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    initial,
                                    "poison upload readback"
                                );
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
    let good = InferenceFwdOperands {
        c: typed(c.cached_ptr(), dt),
        x: typed(a.cached_ptr(), dt),
        w: typed(b.cached_ptr(), dt),
        bias_ptr: None,
    };
    let shape = InferenceShape {
        m: 17,
        k: 65,
        n: 131,
    };
    for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
        inference_forward_with_tile(&ctx, good, shape, candidate)
            .expect("odd-stride positive control");
    }
    for bad in [
        InferenceFwdOperands {
            x: typed(0, dt),
            ..good
        },
        InferenceFwdOperands {
            w: typed(0, dt),
            ..good
        },
        InferenceFwdOperands {
            c: typed(0, dt),
            ..good
        },
        InferenceFwdOperands {
            x: typed(a.cached_ptr() + 1, dt),
            ..good
        },
        InferenceFwdOperands {
            w: typed(b.cached_ptr() + 1, dt),
            ..good
        },
        InferenceFwdOperands {
            c: typed(c.cached_ptr() + 1, dt),
            ..good
        },
        InferenceFwdOperands {
            bias_ptr: Some(1),
            ..good
        },
        InferenceFwdOperands {
            c: typed(c.cached_ptr(), WeightDtype::F32),
            ..good
        },
        InferenceFwdOperands {
            w: typed(b.cached_ptr(), WeightDtype::F16),
            ..good
        },
    ] {
        for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
            assert!(
                inference_forward_with_tile(&ctx, bad, shape, candidate).is_err(),
                "unsafe operands admitted by {candidate:?}"
            );
        }
    }
    for bad in [
        InferenceShape {
            m: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            n: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            k: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            m: 1 << 20,
            n: 1 << 29,
            ..shape
        },
        InferenceShape {
            m: i32::MAX as usize + 1,
            ..shape
        },
    ] {
        for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
            assert!(
                inference_forward_with_tile(&ctx, good, bad, candidate).is_err(),
                "unsafe shape admitted by {candidate:?}: {bad:?}"
            );
        }
    }
    let empty = InferenceFwdOperands {
        c: typed(0, dt),
        x: typed(0, dt),
        w: typed(0, dt),
        bias_ptr: None,
    };
    for k in [2_147_483_521, 2_147_483_584] {
        let error =
            inference_forward_with_tile(&ctx, good, InferenceShape { m: 1, k, n: 1 }, S3_CANDIDATE)
                .expect_err("S3 lookahead overflow must reject before issuing any GPU work");
        assert!(
            error.contains("S3 padded K"),
            "unexpected boundary rejection: {error}"
        );
    }
    for no_output in [
        InferenceShape { m: 0, ..shape },
        InferenceShape { n: 0, ..shape },
    ] {
        for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
            inference_forward_with_tile(&ctx, empty, no_output, candidate)
                .expect("empty output is a no-op");
        }
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

fn assert_half_graph(
    graph: &CudaGraph,
    tile: InferenceTile,
    dtype: WeightDtype,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) {
    assert_half_graph_params(
        graph,
        tile,
        dtype,
        operands,
        shape,
        [
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            shape.m as u32,
            shape.n as u32,
            shape.k as u32,
            shape.k as u32,
            shape.n as u32,
            shape.n as u32,
        ],
    );
}

fn assert_half_graph_params(
    graph: &CudaGraph,
    tile: InferenceTile,
    dtype: WeightDtype,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    bundle: [u32; 8],
) {
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
    let (expected, threads, bm, bn, shared) = match tile {
        InferenceTile::Tc128Sm89Pipeline => (
            format!("gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_{}", dtype.as_str()),
            256,
            128,
            128,
            71_680,
        ),
        InferenceTile::Tc128Sm89Swizzle => (
            format!("gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_{}", dtype.as_str()),
            256,
            128,
            128,
            69_632,
        ),
        InferenceTile::Tc128Sm89S3 => (
            format!("gemm_bi_nn_fixed_sm89_tc128_s3_v1_{}", dtype.as_str()),
            256,
            128,
            128,
            98_304,
        ),
        InferenceTile::TcM64N64Sm89S3 => {
            assert_eq!(dtype, WeightDtype::F16);
            (
                "gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16".into(),
                128,
                64,
                64,
                49_152,
            )
        }
        InferenceTile::TcM128N64Sm89S2 => {
            assert_eq!(dtype, WeightDtype::F16);
            (
                "gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16".into(),
                128,
                128,
                64,
                49_152,
            )
        }
        _ => panic!("not an Ada half physical route: {tile:?}"),
    };
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes(),
        expected.as_bytes()
    );
    assert_eq!(
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        (threads, 1, 1)
    );
    assert_eq!(
        (params.gridDimX, params.gridDimY, params.gridDimZ),
        (
            (shape.m as u32).div_ceil(bm) * (shape.n as u32).div_ceil(bn),
            1,
            1,
        )
    );
    assert_eq!(params.sharedMemBytes, shared);
    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut offset = 0;
        let mut size = 0;
        assert_eq!(
            unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!((offset, size), expected);
    }
    let mut offset = 0;
    let mut size = 0;
    assert_eq!(
        unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) },
        sys::CUresult::CUDA_ERROR_INVALID_VALUE
    );
    assert!(!params.kernelParams.is_null());
    for (index, expected) in [
        operands.c.ptr,
        operands.x.ptr,
        operands.w.ptr,
        operands.bias_ptr.unwrap_or(0),
    ]
    .into_iter()
    .enumerate()
    {
        let pointer = unsafe { *params.kernelParams.add(index) };
        assert!(!pointer.is_null());
        assert_eq!(unsafe { pointer.cast::<u64>().read_unaligned() }, expected);
    }
    let bundle_pointer = unsafe { *params.kernelParams.add(4) };
    assert!(!bundle_pointer.is_null());
    assert_eq!(
        unsafe { bundle_pointer.cast::<[u32; 8]>().read_unaligned() },
        bundle
    );
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawHalfParams([u32; 8]);
unsafe impl cudarc::driver::DeviceRepr for RawHalfParams {}

fn launch_raw_half(
    ctx: &GpuCtx,
    tile: InferenceTile,
    dtype: WeightDtype,
    ops: InferenceFwdOperands,
    params: RawHalfParams,
) {
    use cudarc::driver::PushKernelArg;
    let (holder, shared) = match tile {
        SWIZZLE_CANDIDATE => (
            ctx.kernels.fixed_sm89_half_swizzle.as_ref().unwrap(),
            69_632,
        ),
        S3_CANDIDATE => (ctx.kernels.fixed_sm89_half_s3.as_ref().unwrap(), 98_304),
        _ => panic!("unsupported raw half route"),
    };
    let bias = ops.bias_ptr.unwrap_or(0);
    let mut launch = ctx.stream.launch_builder(holder.get(dtype));
    launch
        .arg(&ops.c.ptr)
        .arg(&ops.x.ptr)
        .arg(&ops.w.ptr)
        .arg(&bias)
        .arg(&params);
    unsafe {
        launch.launch(cudarc::driver::LaunchConfig {
            grid_dim: (params.0[2].div_ceil(128) * params.0[3].div_ceil(128), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared,
        })
    }
    .expect("raw five-argument half launch");
}

#[test]
#[ignore = "requires Ada production NVRTC holders; nonunit scalars and independent physical strides"]
fn fixed_sm89_half_s3_raw_alpha_beta_strides_graph_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n, lda, ldb, ldc, offset) in [
            (129, 256, 136, 264, 144, 144, 8),
            (17, 65, 131, 69, 137, 139, 1),
            (17, 0, 131, 3, 137, 139, 1),
        ] {
            let a_bits: Vec<_> = (0..offset + m * lda + 9)
                .map(|i| half_bits(((i % 7) as f32 - 3.0) / 32.0, dtype))
                .collect();
            let b_bits: Vec<_> = (0..offset + k * ldb + 9)
                .map(|i| half_bits(((i % 11) as f32 - 5.0) / 32.0, dtype))
                .collect();
            let bias_bits: Vec<_> = (0..n)
                .map(|i| f32::from_bits([0x3dcccccd, 0xbeaaaaab][i % 2]))
                .collect();
            let a = upload_half(&ctx, &a_bits);
            let b = upload_half(&ctx, &b_bits);
            let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_bits).unwrap();
            for (alpha, beta, has_bias) in [
                (0.75f32, 0.0f32, false),
                (-0.5, 0.25, false),
                (1.0, -0.25, true),
            ] {
                let initial = vec![half_bits(3.0, dtype); offset + m * ldc + 9];
                let oracle = upload_half(&ctx, &initial);
                let mut output = upload_half(&ctx, &initial);
                let ops = InferenceFwdOperands {
                    c: typed(output.cached_ptr() + 2 * offset as u64, dtype),
                    x: typed(
                        if k == 0 {
                            0
                        } else {
                            a.cached_ptr() + 2 * offset as u64
                        },
                        dtype,
                    ),
                    w: typed(
                        if k == 0 {
                            0
                        } else {
                            b.cached_ptr() + 2 * offset as u64
                        },
                        dtype,
                    ),
                    bias_ptr: has_bias.then_some(bias.cached_ptr()),
                };
                let params = RawHalfParams([
                    alpha.to_bits(),
                    beta.to_bits(),
                    m as u32,
                    n as u32,
                    k as u32,
                    lda as u32,
                    ldb as u32,
                    ldc as u32,
                ]);
                launch_raw_half(
                    &ctx,
                    SWIZZLE_CANDIDATE,
                    dtype,
                    InferenceFwdOperands {
                        c: typed(oracle.cached_ptr() + 2 * offset as u64, dtype),
                        ..ops
                    },
                    params,
                );
                let expected = raw_half(&ctx, &oracle);
                let mut poison = initial.clone();
                for row in 0..m {
                    for column in 0..n {
                        let i = offset + row * ldc + column;
                        if beta == 0.0 {
                            poison[i] = expected[i] ^ 0xffff;
                        }
                        assert_ne!(
                            poison[i], expected[i],
                            "each independently reset output must differ from gold"
                        );
                    }
                }
                for _ in 0..2 {
                    output
                        .upload_bytes(&ctx.stream, bytemuck::cast_slice(&poison))
                        .unwrap();
                    assert_eq!(raw_half(&ctx, &output), poison);
                    launch_raw_half(&ctx, S3_CANDIDATE, dtype, ops, params);
                    assert_eq!(
                        raw_half(&ctx, &output),
                        expected,
                        "raw S3 scalar/stride/guard bits"
                    );
                }
                let graph = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        launch_raw_half(&ctx, S3_CANDIDATE, dtype, ops, params);
                        Ok(())
                    })
                }
                .unwrap();
                assert_half_graph_params(
                    &graph,
                    S3_CANDIDATE,
                    dtype,
                    ops,
                    InferenceShape { m, k, n },
                    params.0,
                );
                for _ in 0..2 {
                    output
                        .upload_bytes(&ctx.stream, bytemuck::cast_slice(&poison))
                        .unwrap();
                    assert_eq!(raw_half(&ctx, &output), poison);
                    graph.launch().unwrap();
                    assert_eq!(
                        raw_half(&ctx, &output),
                        expected,
                        "raw S3 independently reset graph bits"
                    );
                }
            }
            assert_eq!(raw_half(&ctx, &a), a_bits);
            assert_eq!(raw_half(&ctx, &b), b_bits);
            assert_eq!(
                bias.to_cpu(&ctx.stream)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                bias_bits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
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
        for (k, n) in [
            (64, 136),
            (128, 136),
            (192, 136),
            (256, 136),
            (65, 131),
            (0, 131),
        ] {
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
                    let operands = InferenceFwdOperands {
                        c: typed(full.cached_ptr(), dtype),
                        x: typed(if k == 0 { 0 } else { a.cached_ptr() }, dtype),
                        w: typed(if k == 0 { 0 } else { b.cached_ptr() }, dtype),
                        bias_ptr: has_bias.then_some(bias.cached_ptr()),
                    };
                    inference_forward_with_tile(
                        &ctx,
                        operands,
                        InferenceShape { m: rows, k, n },
                        InferenceTile::Tc128,
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
                            let guards = vec![0x7fff; output_offset + m * n + 9];
                            let mut expected = guards.clone();
                            expected[output_offset..output_offset + m * n]
                                .copy_from_slice(&reference[row_offset * n..(row_offset + m) * n]);
                            let mut initial = guards;
                            for (poison, gold) in initial[output_offset..output_offset + m * n]
                                .iter_mut()
                                .zip(&expected[output_offset..output_offset + m * n])
                            {
                                *poison = *gold ^ 0xffff;
                                assert_ne!(*poison, *gold);
                            }
                            let mut output = upload_half(&ctx, &initial);
                            assert_eq!(raw_half(&ctx, &output), initial, "initial poison readback");
                            let view = InferenceFwdOperands {
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
                            let shape = InferenceShape { m, k, n };
                            for tile in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE]
                                .into_iter()
                                .chain(RUNGS)
                            {
                                output
                                    .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                    .expect("poison C");
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    initial,
                                    "poison upload readback"
                                );
                                inference_forward_with_tile(&ctx, view, shape, tile)
                                    .unwrap_or_else(|e| {
                                        panic!("{tile:?} {dtype:?} M={m} K={k} N={n}: {e}")
                                    });
                                assert_eq!(
                                    raw_half(&ctx, &output),
                                    expected,
                                    "rung/prefix/view/guard bits: {tile:?} {dtype:?} M={m} K={k} N={n} row={row_offset} bias={has_bias} corpus={corpus}"
                                );
                            }
                            for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
                                let run =
                                    || inference_forward_with_tile(&ctx, view, shape, candidate);
                                for _ in 0..2 {
                                    output
                                        .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                        .expect("poison repeat");
                                    assert_eq!(
                                        raw_half(&ctx, &output),
                                        initial,
                                        "poison upload readback"
                                    );
                                    run().expect("eager repeat");
                                    assert_eq!(
                                        raw_half(&ctx, &output),
                                        expected,
                                        "eager repeat changed bits"
                                    );
                                }
                                let graph = unsafe { capture_into_graph(&ctx.stream, run) }
                                    .expect("capture candidate");
                                assert_half_graph(&graph, candidate, dtype, view, shape);
                                for _ in 0..2 {
                                    output
                                        .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                        .expect("poison replay");
                                    assert_eq!(
                                        raw_half(&ctx, &output),
                                        initial,
                                        "poison upload readback"
                                    );
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
#[ignore = "requires exact Ada CC8.9; bounded half rounding-edge corpus"]
fn fixed_sm89_half_swizzle_rounding_edges_match_incumbent_across_store_paths() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let edges: [u16; 10] = match dtype {
            WeightDtype::Bf16 => [
                0x0001, 0x007f, 0x0080, 0x7f7f, 0x8001, 0x807f, 0x8080, 0xff7f, 0x3f80, 0xbf80,
            ],
            WeightDtype::F16 => [
                0x0001, 0x03ff, 0x0400, 0x7bff, 0x8001, 0x83ff, 0x8400, 0xfbff, 0x3c00, 0xbc00,
            ],
            WeightDtype::F32 => unreachable!(),
        };
        for (m, k, n) in [(129, 64, 136), (17, 65, 131)] {
            let mut a_host: Vec<_> = (0..m * k).map(|i| edges[i % edges.len()]).collect();
            // Keep one finite-only row beside the dense max-finite/cancellation rows,
            // so this corpus checks RNE output stores as well as NaN/overflow behavior.
            let finite_edges = [edges[0], edges[1], edges[2], edges[4], edges[5], edges[6]];
            for column in 0..k {
                a_host[(m - 1) * k + column] = finite_edges[column % finite_edges.len()];
            }
            let b_host: Vec<_> = (0..k * n)
                .map(|i| edges[(i * 7 + i / n) % edges.len()])
                .collect();
            let bias_host: Vec<_> = (0..n)
                .map(|i| f32::from_bits([0x3dcccccd, 0xbdcccccd, 0x3eaaaaab, 0xbeaaaaab][i % 4]))
                .collect();
            let a = upload_half(&ctx, &a_host);
            let b = upload_half(&ctx, &b_host);
            let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias upload");
            let shape = InferenceShape { m, k, n };
            let reference = upload_half(&ctx, &vec![0x7fff; m * n]);
            let reference_ops = InferenceFwdOperands {
                c: typed(reference.cached_ptr(), dtype),
                x: typed(a.cached_ptr(), dtype),
                w: typed(b.cached_ptr(), dtype),
                bias_ptr: Some(bias.cached_ptr()),
            };
            inference_forward_with_tile(&ctx, reference_ops, shape, InferenceTile::Tc128)
                .expect("portable incumbent rounding oracle");
            let expected = raw_half(&ctx, &reference);
            let exponent_mask = if dtype == WeightDtype::Bf16 {
                0x7f80
            } else {
                0x7c00
            };
            assert!(
                expected
                    .iter()
                    .any(|bits| bits & exponent_mask != exponent_mask),
                "rounding-edge corpus must retain finite outputs"
            );
            for output_offset in [1, 8] {
                let initial = vec![0x7fff; output_offset + m * n + 9];
                let mut poison = initial.clone();
                for (actual, expected) in poison[output_offset..output_offset + m * n]
                    .iter_mut()
                    .zip(&expected)
                {
                    *actual = !*expected;
                }
                let mut output = upload_half(&ctx, &initial);
                let operands = InferenceFwdOperands {
                    c: typed(output.cached_ptr() + (output_offset * 2) as u64, dtype),
                    ..reference_ops
                };
                for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
                    output
                        .upload_bytes(&ctx.stream, bytemuck::cast_slice(&poison))
                        .expect("poison rounding-edge output");
                    assert_eq!(raw_half(&ctx, &output), poison, "poison upload readback");
                    inference_forward_with_tile(&ctx, operands, shape, candidate)
                        .unwrap_or_else(|error| panic!("{candidate:?}: {error}"));
                    let actual = raw_half(&ctx, &output);
                    assert_eq!(&actual[..output_offset], &initial[..output_offset]);
                    assert_eq!(
                        &actual[output_offset..output_offset + m * n],
                        expected.as_slice(),
                        "rounding-edge bits differ for {candidate:?}/{dtype:?}/{shape:?}/offset={output_offset}"
                    );
                    assert_eq!(
                        &actual[output_offset + m * n..],
                        &initial[output_offset + m * n..]
                    );
                }
            }
            assert_eq!(raw_half(&ctx, &a), a_host);
            assert_eq!(raw_half(&ctx, &b), b_host);
        }
    }
}

#[test]
#[ignore = "requires exact Ada CC8.9; bounded actual-NVRTC smoke for all four sanitizer tools"]
fn fixed_sm89_half_pipeline_sanitizer_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9), "this gate requires Ada");
    let ctx = GpuCtx::new(&device).expect("NVRTC context");
    // All three independently admitted half routes execute eager and captured
    // cases; module-admission probes precede the guarded test body.
    const HALF_GUARD: usize = 8;
    const BIAS_GUARD: usize = 4;
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n, exceptional) in [
            (129, 256, 136, false), // aligned cp.async, three slabs, M/N tails
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
                let operands = InferenceFwdOperands {
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
                inference_forward_with_tile(
                    &ctx,
                    operands,
                    InferenceShape { m: rows, k, n },
                    InferenceTile::Tc128,
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
                    let mut initial = vec![0x7fff; output_offset + m * n + HALF_GUARD];
                    let mut output = upload_half(&ctx, &initial);
                    let view = InferenceFwdOperands {
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
                    let shape = InferenceShape { m, k, n };
                    let mut expected = initial.clone();
                    let reference_start = HALF_GUARD + row_offset * n;
                    expected[output_offset..output_offset + m * n]
                        .copy_from_slice(&reference[reference_start..reference_start + m * n]);
                    for (poison, gold) in initial[output_offset..output_offset + m * n]
                        .iter_mut()
                        .zip(&expected[output_offset..output_offset + m * n])
                    {
                        *poison = *gold ^ 0xffff;
                        assert_ne!(*poison, *gold);
                    }
                    for candidate in [CANDIDATE, SWIZZLE_CANDIDATE, S3_CANDIDATE] {
                        let run = || inference_forward_with_tile(&ctx, view, shape, candidate);
                        for _ in 0..2 {
                            output
                                .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                .expect("poison sanitizer eager output");
                            assert_eq!(raw_half(&ctx, &output), initial, "poison upload readback");
                            run().expect("forced NVRTC Ada half eager");
                            assert_eq!(
                                raw_half(&ctx, &output),
                                expected,
                                "sanitizer eager bits/guards {candidate:?} {dtype:?} {shape:?} row={row_offset} C_offset={output_offset} bias={has_bias} exceptional={exceptional}"
                            );
                        }
                        let graph = unsafe { capture_into_graph(&ctx.stream, run) }
                            .expect("capture actual forced NVRTC Ada half");
                        assert_half_graph(&graph, candidate, dtype, view, shape);
                        for _ in 0..2 {
                            output
                                .upload_bytes(&ctx.stream, bytemuck::cast_slice(&initial))
                                .expect("poison sanitizer graph output");
                            assert_eq!(raw_half(&ctx, &output), initial, "poison upload readback");
                            graph.launch().expect("sanitizer graph replay");
                            assert_eq!(
                                raw_half(&ctx, &output),
                                expected,
                                "sanitizer graph bits/guards {candidate:?} {dtype:?} {shape:?} row={row_offset} C_offset={output_offset} bias={has_bias} exceptional={exceptional}"
                            );
                        }
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
