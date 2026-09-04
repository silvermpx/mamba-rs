#![cfg(feature = "cuda")]

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::CudaGraph;
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedSm120HalfTile, FixedTile, fixed_forward,
    fixed_forward_f32_legacy_baseline, fixed_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, Sm120Bk, Sm120ForcedRoute,
    Sm120LaunchOperands, Sm120MapRequest, Sm120Op, Sm120PhysicalRoute, Sm120Shape, Sm120Stages,
    Sm120Tile, TcTile, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
    Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile, launch_sm120_tma_prepared,
    prepare_sm120_tensor_maps, prepare_sm120_tma_forced, presize_physical_qualification_suite,
    qualify_physical_launch, resolve_sm120_forced, tf32_route_specs,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ModuleKind, ResolvedGemmOp, TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest, Sha256};

const WARMUPS: usize = 10;
const ITERS: usize = 200;

fn typed(buffer: &DtypedBuf, dtype: WeightDtype) -> TypedPtr {
    TypedPtr {
        ptr: buffer.cached_ptr(),
        dtype,
    }
}

fn average_us(ctx: &GpuCtx, mut run: impl FnMut()) -> f64 {
    for _ in 0..WARMUPS {
        run();
    }
    ctx.stream.synchronize().expect("warmup sync");
    let started = Instant::now();
    for _ in 0..ITERS {
        run();
    }
    ctx.stream.synchronize().expect("timing sync");
    started.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn fixed_tile_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
    iterations: usize,
) -> f64 {
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed window start");
    for _ in 0..iterations {
        fixed_forward_with_tile(ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed window end");
    f64::from(start.elapsed_ms(&end).expect("measure Fixed window")) * 1000.0 / iterations as f64
}

fn fixed_tile_window_iterations(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> usize {
    let pilot_iterations = 16;
    let pilot_us = fixed_tile_window_us(ctx, operands, shape, tile, pilot_iterations);
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state & 0xffff) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn f32_bits(ctx: &GpuCtx, buffer: &DtypedBuf, len: usize) -> Vec<u32> {
    let mut host = vec![0.0f32; len];
    buffer
        .download_f32(&ctx.stream, &mut host)
        .expect("f32 result download");
    host.into_iter().map(f32::to_bits).collect()
}

fn single_graph_kernel_name(graph: &CudaGraph, label: &str) -> String {
    let mut node_count = 0;
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuGraphGetNodes(
                graph.cu_graph(),
                std::ptr::null_mut(),
                &mut node_count,
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph node count"
    );
    assert_eq!(node_count, 1, "{label} physical graph node inventory");
    let mut nodes = vec![std::ptr::null_mut(); node_count];
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuGraphGetNodes(
                graph.cu_graph(),
                nodes.as_mut_ptr(),
                &mut node_count,
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph nodes"
    );
    let mut params = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph kernel params"
    );
    let mut function_name = std::ptr::null();
    assert_eq!(
        unsafe { cudarc::driver::sys::cuFuncGetName(&mut function_name, params.func) },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph kernel name"
    );
    unsafe { CStr::from_ptr(function_name) }
        .to_str()
        .expect("UTF-8 CUDA function name")
        .to_owned()
}

#[test]
fn forced_fixed_launch_uses_structured_arguments() {
    let _: fn(&GpuCtx, FixedFwdOperands, FixedShape, FixedTile) -> Result<(), String> =
        fixed_forward_with_tile;
}

#[test]
#[ignore = "requires a CC12.0 170-SM CUDA device"]
fn fixed_sm120_tf32_production_routes_match_forced_bits_and_graphs() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let m64n128 = &ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .expect("SM120 TF32 kernels")
        .m64n128_s2;
    assert_eq!(
        m64n128
            .local_size_bytes()
            .expect("M64N128 local-memory footprint"),
        0,
        "SM120 TF32 M64N128 spills to local memory"
    );
    assert_eq!(
        m64n128
            .occupancy_max_active_blocks_per_multiprocessor(128, 49_280, None)
            .expect("M64N128 occupancy"),
        2,
        "SM120 TF32 M64N128 must retain two resident CTAs"
    );
    let m128n64 = &ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .expect("SM120 TF32 kernels")
        .m128n64_s2;
    assert_eq!(
        m128n64
            .local_size_bytes()
            .expect("M128N64 local-memory footprint"),
        0,
        "SM120 TF32 M128N64 spills to local memory"
    );
    assert_eq!(
        m128n64
            .occupancy_max_active_blocks_per_multiprocessor(128, 49_280, None)
            .expect("M128N64 occupancy"),
        2,
        "SM120 TF32 M128N64 must retain two resident CTAs"
    );
    let (nvrtc_major, nvrtc_minor) = ctx.kernels.compiler_identity().nvrtc_version;
    configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
    for (cell, shape, output_offset, has_bias) in [
        (
            "A",
            FixedShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            0,
            false,
        ),
        (
            "B",
            FixedShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            0,
            false,
        ),
        (
            "C",
            FixedShape {
                m: 4621,
                k: 1928,
                n: 384,
            },
            0,
            false,
        ),
        (
            "D",
            FixedShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            0,
            false,
        ),
        (
            "E",
            FixedShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
            0,
            false,
        ),
        (
            "A_bias",
            FixedShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            0,
            true,
        ),
        (
            "B_bias",
            FixedShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            0,
            true,
        ),
        (
            "D_bias",
            FixedShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            0,
            true,
        ),
        (
            "D_misaligned",
            FixedShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            1,
            false,
        ),
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("production A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("production B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)
            .expect("production bias allocation");
        let output_len = output_offset + shape.m * shape.n;
        let forced = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
            .expect("forced output allocation");
        let production = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
            .expect("production output allocation");
        a.upload_f32(
            &ctx.stream,
            &synth(shape.m * shape.k, 0xa170_5052_4f44_0001),
        )
        .expect("production A upload");
        b.upload_f32(
            &ctx.stream,
            &synth(shape.k * shape.n, 0xb170_5052_4f44_0002),
        )
        .expect("production B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5_5052_4f44_0003))
            .expect("production bias upload");
        let bias_ptr = has_bias.then(|| bias.cached_ptr());
        let forced_operands = FixedFwdOperands {
            c: TypedPtr {
                ptr: forced.cached_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64,
                dtype: WeightDtype::F32,
            },
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr,
        };
        let production_operands = FixedFwdOperands {
            c: TypedPtr {
                ptr: production.cached_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64,
                dtype: WeightDtype::F32,
            },
            ..forced_operands
        };
        fixed_forward_with_tile(&ctx, forced_operands, shape, FixedTile::Tf32Sm120M64S2)
            .expect("forced incumbent launch");
        let selected = fixed_forward(
            &ctx,
            production_operands.c,
            production_operands.x,
            production_operands.w,
            bias_ptr,
            (shape.m, shape.k, shape.n),
        )
        .expect("production pair-store launch");
        let dims = (shape.m, shape.k, shape.n);
        let uses_wide_tile = (nvrtc_major, nvrtc_minor) == (13, 2) && dims == (4621, 768, 2304);
        let uses_producer_warp = (nvrtc_major, nvrtc_minor) == (13, 2) && dims == (2048, 768, 2304);
        let expected_tile = if uses_wide_tile {
            FixedTile::Tf32Sm120M128S2
        } else if uses_producer_warp {
            FixedTile::Tf32Sm120M64S2ProducerWarp
        } else {
            FixedTile::Tf32Sm120M64S2
        };
        assert_eq!(selected, expected_tile, "{cell} route identity");
        ctx.stream.synchronize().expect("production eager sync");
        let reference = f32_bits(&ctx, &forced, output_len);
        assert_eq!(
            f32_bits(&ctx, &production, output_len),
            reference,
            "{cell} production bits"
        );
        let graph = unsafe {
            capture_into_graph(&ctx.stream, || {
                fixed_forward(
                    &ctx,
                    production_operands.c,
                    production_operands.x,
                    production_operands.w,
                    bias_ptr,
                    (shape.m, shape.k, shape.n),
                )
                .map(|_| ())
            })
        }
        .expect("capture production pair-store graph");
        let mut node_count = 0;
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuGraphGetNodes(
                    graph.cu_graph(),
                    std::ptr::null_mut(),
                    &mut node_count,
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph node count"
        );
        assert_eq!(node_count, 1, "{cell} physical graph node inventory");
        let mut nodes = vec![std::ptr::null_mut(); node_count];
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuGraphGetNodes(
                    graph.cu_graph(),
                    nodes.as_mut_ptr(),
                    &mut node_count,
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph nodes"
        );
        let mut params = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph kernel params"
        );
        let mut function_name = std::ptr::null();
        assert_eq!(
            unsafe { cudarc::driver::sys::cuFuncGetName(&mut function_name, params.func) },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph kernel name"
        );
        let function_name = unsafe { CStr::from_ptr(function_name) }
            .to_str()
            .expect("UTF-8 CUDA function name");
        let uses_pair_store = output_offset == 0
            && ((nvrtc_major, nvrtc_minor) == (13, 0)
                || ((nvrtc_major, nvrtc_minor) == (12, 8)
                    && matches!(dims, (2048, 768, 2304) | (2048, 2304, 768))));
        let expected_function = if uses_wide_tile {
            "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2"
        } else if uses_producer_warp {
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp"
        } else if uses_pair_store {
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store"
        } else {
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2"
        };
        assert_eq!(function_name, expected_function, "{cell} physical route");
        if cell == "D" {
            let forced_graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    fixed_forward_with_tile(&ctx, forced_operands, shape, FixedTile::Tf32Sm120M64S2)
                })
            }
            .expect("capture forced incumbent graph");
            let mut forced_node_count = 0;
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphGetNodes(
                        forced_graph.cu_graph(),
                        std::ptr::null_mut(),
                        &mut forced_node_count,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(forced_node_count, 1, "forced physical graph node inventory");
            let mut forced_nodes = vec![std::ptr::null_mut(); forced_node_count];
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphGetNodes(
                        forced_graph.cu_graph(),
                        forced_nodes.as_mut_ptr(),
                        &mut forced_node_count,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            let mut forced_params = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(
                        forced_nodes[0],
                        &mut forced_params,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            let mut forced_function_name = std::ptr::null();
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuFuncGetName(
                        &mut forced_function_name,
                        forced_params.func,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                unsafe { CStr::from_ptr(forced_function_name) }
                    .to_str()
                    .expect("forced UTF-8 CUDA function name"),
                "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2"
            );
        }
        for replay in 0..10 {
            graph.launch().expect("production graph launch");
            ctx.stream.synchronize().expect("production graph sync");
            assert_eq!(
                f32_bits(&ctx, &production, output_len),
                reference,
                "{cell} production graph replay {replay}"
            );
        }
    }
}

#[test]
fn f32_legacy_baseline_launch_uses_structured_arguments() {
    let _: fn(&GpuCtx, FixedFwdOperands, FixedShape) -> Result<(), String> =
        fixed_forward_f32_legacy_baseline;
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn f32_s2_production_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for shape in [
        FixedShape { m: 1, k: 0, n: 1 },
        FixedShape {
            m: 63,
            k: 31,
            n: 63,
        },
        FixedShape {
            m: 64,
            k: 32,
            n: 64,
        },
        FixedShape {
            m: 65,
            k: 33,
            n: 65,
        },
        FixedShape {
            m: 65,
            k: 36,
            n: 68,
        },
        FixedShape {
            m: 129,
            k: 97,
            n: 193,
        },
        FixedShape {
            m: 4621,
            k: 384,
            n: 384,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("baseline allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let baseline_operands = FixedFwdOperands {
                c: typed(&baseline, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            let candidate_operands = FixedFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..baseline_operands
            };
            fixed_forward_f32_legacy_baseline(&ctx, baseline_operands, shape)
                .expect("legacy Fixed launch");
            let tile = fixed_forward(
                &ctx,
                candidate_operands.c,
                candidate_operands.x,
                candidate_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .expect("production S2 launch");
            assert_eq!(tile, FixedTile::Legacy);
            ctx.stream.synchronize().expect("S2 sync");
            assert_eq!(
                f32_bits(&ctx, &candidate, shape.m * shape.n),
                f32_bits(&ctx, &baseline, shape.m * shape.n),
                "F32 production S2 changed legacy bits for M{} K{} N{} bias={}",
                shape.m,
                shape.k,
                shape.n,
                bias_ptr.is_some(),
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_tf32_forward_is_repeatable_and_tile_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let tiles = [
        FixedTile::Tf32M128S2,
        FixedTile::Tf32M128S3,
        FixedTile::Tf32M64S2,
        FixedTile::Tf32M64S3,
        FixedTile::Tf32M16S4,
    ];
    for shape in [
        FixedShape { m: 1, k: 0, n: 1 },
        FixedShape {
            m: 15,
            k: 31,
            n: 31,
        },
        FixedShape {
            m: 65,
            k: 36,
            n: 68,
        },
        FixedShape {
            m: 129,
            k: 97,
            n: 193,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x7f32a))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x7f32b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x7f32c))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let mut reference = None;
            for tile in tiles {
                let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("output allocation");
                let operands = FixedFwdOperands {
                    c: typed(&output, WeightDtype::F32),
                    x: typed(&a, WeightDtype::F32),
                    w: typed(&b, WeightDtype::F32),
                    bias_ptr,
                };
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                ctx.stream.synchronize().expect("first TF32 sync");
                let first = f32_bits(&ctx, &output, shape.m * shape.n);
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("repeated {tile:?}: {error}"));
                ctx.stream.synchronize().expect("second TF32 sync");
                let second = f32_bits(&ctx, &output, shape.m * shape.n);
                assert_eq!(first, second, "TF32 repeat drift for {shape:?} {tile:?}");
                if let Some(expected) = &reference {
                    assert_eq!(
                        &first,
                        expected,
                        "TF32 tile drift for {shape:?} {tile:?} bias={}",
                        bias_ptr.is_some()
                    );
                } else {
                    reference = Some(first);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_tf32_is_portable_bit_exact_and_selected() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if !matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        return;
    }
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let specialized = [
        FixedTile::Tf32Sm120M128S2,
        FixedTile::Tf32Sm120M128S3,
        FixedTile::Tf32Sm120M64N128S2,
        FixedTile::Tf32Sm120M64N128S3,
        FixedTile::Tf32Sm120M64S2ProducerWarp,
        FixedTile::Tf32Sm120M64S2,
    ];
    for shape in [
        FixedShape {
            m: 65,
            k: 36,
            n: 68,
        },
        FixedShape {
            m: 129,
            k: 96,
            n: 196,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let portable = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("portable output");
            let portable_operands = FixedFwdOperands {
                c: typed(&portable, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            fixed_forward_with_tile(&ctx, portable_operands, shape, FixedTile::Tf32M64S2)
                .expect("portable TF32 launch");
            ctx.stream.synchronize().expect("portable sync");
            let expected = f32_bits(&ctx, &portable, shape.m * shape.n);
            for tile in specialized {
                let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("SM120 output");
                let operands = FixedFwdOperands {
                    c: typed(&output, WeightDtype::F32),
                    ..portable_operands
                };
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                ctx.stream.synchronize().expect("first SM120 TF32 sync");
                let first = f32_bits(&ctx, &output, shape.m * shape.n);
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("repeated {tile:?}: {error}"));
                ctx.stream.synchronize().expect("second SM120 TF32 sync");
                assert_eq!(
                    first,
                    f32_bits(&ctx, &output, shape.m * shape.n),
                    "SM120 repeat drift for {shape:?} {tile:?}"
                );
                assert_eq!(
                    first,
                    expected,
                    "SM120/portable drift for {shape:?} {tile:?} bias={}",
                    bias_ptr.is_some()
                );
            }
            let selected = fixed_forward(
                &ctx,
                portable_operands.c,
                portable_operands.x,
                portable_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .expect("automatic SM120 TF32 launch");
            assert_eq!(selected, FixedTile::Tf32Sm120M64S2);
        }
    }
}

#[test]
#[ignore = "requires a quiet RTX 5090 and screens the SM120 TF32 producer-warp candidate"]
fn fixed_sm120_tf32_producer_warp_candidate_screen() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0),
        "producer-warp candidate screen is qualified on SM120",
    );
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);

    let incumbent = FixedTile::Tf32Sm120M64S2;
    let candidate = FixedTile::Tf32Sm120M64S2ProducerWarp;
    for (label, shape) in [
        (
            "A",
            FixedShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
        ),
        (
            "B",
            FixedShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
        ),
        (
            "D",
            FixedShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
        ),
    ] {
        let output_len = shape.m * shape.n;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        let incumbent_output =
            DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32).expect("incumbent output");
        let candidate_output =
            DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32).expect("candidate output");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x6496_a000))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x6496_b000))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x6496_c000))
            .expect("bias upload");

        for (bias_label, bias_ptr) in [("none", None), ("bias", Some(bias.cached_ptr()))] {
            let incumbent_operands = FixedFwdOperands {
                c: typed(&incumbent_output, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            let candidate_operands = FixedFwdOperands {
                c: typed(&candidate_output, WeightDtype::F32),
                ..incumbent_operands
            };
            fixed_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                .expect("incumbent exact-bit launch");
            fixed_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                .expect("candidate exact-bit launch");
            ctx.stream.synchronize().expect("exact-bit synchronization");
            let expected = f32_bits(&ctx, &incumbent_output, output_len);
            assert_eq!(
                f32_bits(&ctx, &candidate_output, output_len),
                expected,
                "candidate bit drift for {label}/{bias_label}"
            );

            let graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    fixed_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                })
            }
            .expect("capture producer-warp graph");
            assert_eq!(
                single_graph_kernel_name(&graph, "producer-warp candidate"),
                "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp"
            );
            for replay in 0..10 {
                graph.launch().expect("producer-warp graph launch");
                ctx.stream.synchronize().expect("producer-warp graph sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate_output, output_len),
                    expected,
                    "candidate graph drift for {label}/{bias_label} replay {replay}"
                );
            }

            for _ in 0..128 {
                fixed_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                    .expect("incumbent warmup");
                fixed_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                    .expect("candidate warmup");
            }
            ctx.stream.synchronize().expect("candidate warmup sync");
            let incumbent_iterations =
                fixed_tile_window_iterations(&ctx, incumbent_operands, shape, incumbent);
            let candidate_iterations =
                fixed_tile_window_iterations(&ctx, candidate_operands, shape, candidate);
            for (order, candidate_first) in [
                ("candidate_then_incumbent", true),
                ("incumbent_then_candidate", false),
            ] {
                let mut ratios = Vec::with_capacity(101);
                for _ in 0..101 {
                    let (candidate_us, incumbent_us) = if candidate_first {
                        (
                            fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                candidate,
                                candidate_iterations,
                            ),
                            fixed_tile_window_us(
                                &ctx,
                                incumbent_operands,
                                shape,
                                incumbent,
                                incumbent_iterations,
                            ),
                        )
                    } else {
                        let incumbent_us = fixed_tile_window_us(
                            &ctx,
                            incumbent_operands,
                            shape,
                            incumbent,
                            incumbent_iterations,
                        );
                        let candidate_us = fixed_tile_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            candidate,
                            candidate_iterations,
                        );
                        (candidate_us, incumbent_us)
                    };
                    ratios.push(candidate_us / incumbent_us);
                }
                ratios.sort_by(f64::total_cmp);
                println!(
                    "TF32 producer-warp label={label} bias={bias_label} order={order} candidate_over_incumbent_p50={:.9} p95={:.9}",
                    percentile(&ratios, 0.50),
                    percentile(&ratios, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_tf32_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if !matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        return;
    }
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let shape = FixedShape {
        m: 65,
        k: 36,
        n: 68,
    };
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32).expect("B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("output allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let run = || {
        fixed_forward(
            &ctx,
            typed(&output, WeightDtype::F32),
            typed(&a, WeightDtype::F32),
            typed(&b, WeightDtype::F32),
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| {
            if tile == FixedTile::Tf32Sm120M64S2 {
                Ok(())
            } else {
                Err(format!("unexpected production tile {tile:?}"))
            }
        })
    };
    run().expect("warm tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    let eager = f32_bits(&ctx, &output, shape.m * shape.n);
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 TF32");
    for replay in 0..10 {
        graph.launch().expect("launch Fixed SM120 graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "Fixed SM120 graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
        .expect("cold A allocation");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
        .expect("cold B allocation");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            fixed_forward(
                &ctx,
                typed(&output, WeightDtype::F32),
                typed(&cold_a, WeightDtype::F32),
                typed(&cold_b, WeightDtype::F32),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_tf32_forward_is_batch_prefix_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let (large_m, k, n) = (129usize, 96usize, 196usize);
    let large_a_host = synth(large_m * k, 0xba7c4);
    let b_host = synth(k * n, 0xb32);
    let bias_host = synth(n, 0xb1a5);
    let small_a = DtypedBuf::zeros(&ctx.stream, k, WeightDtype::F32).expect("small A");
    let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, WeightDtype::F32).expect("large A");
    let b = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::F32).expect("B");
    let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
    let small_c = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("small C");
    let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, WeightDtype::F32).expect("large C");
    small_a
        .upload_f32(&ctx.stream, &large_a_host[..k])
        .expect("small A upload");
    large_a
        .upload_f32(&ctx.stream, &large_a_host)
        .expect("large A upload");
    b.upload_f32(&ctx.stream, &b_host).expect("B upload");
    bias.upload_f32(&ctx.stream, &bias_host)
        .expect("bias upload");
    let small_tile = fixed_forward(
        &ctx,
        typed(&small_c, WeightDtype::F32),
        typed(&small_a, WeightDtype::F32),
        typed(&b, WeightDtype::F32),
        Some(bias.cached_ptr()),
        (1, k, n),
    )
    .expect("small TF32 launch");
    let large_tile = fixed_forward(
        &ctx,
        typed(&large_c, WeightDtype::F32),
        typed(&large_a, WeightDtype::F32),
        typed(&b, WeightDtype::F32),
        Some(bias.cached_ptr()),
        (large_m, k, n),
    )
    .expect("large TF32 launch");
    ctx.stream.synchronize().expect("TF32 prefix sync");
    assert_ne!(small_tile, large_tile, "test must cross selector rungs");
    assert_eq!(
        f32_bits(&ctx, &small_c, n),
        f32_bits(&ctx, &large_c, large_m * n)[..n],
        "TF32 first-row bits changed with batch size"
    );
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits TF32 performance data"]
fn fixed_tf32_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let mut tiles = vec![
        FixedTile::Tf32M128S2,
        FixedTile::Tf32M128S3,
        FixedTile::Tf32M64S2,
        FixedTile::Tf32M64S3,
        FixedTile::Tf32M16S4,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        tiles.extend([
            FixedTile::Tf32Sm120M128S2,
            FixedTile::Tf32Sm120M128S3,
            FixedTile::Tf32Sm120M64N128S2,
            FixedTile::Tf32Sm120M64N128S3,
            FixedTile::Tf32Sm120M64S2,
        ]);
    }
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    for shape in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("C allocation");
        let operands = FixedFwdOperands {
            c: typed(&c, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        for &tile in &tiles {
            let elapsed_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
            });
            println!(
                "m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n
            );
        }
        let mut selected = FixedTile::Legacy;
        let auto_us = average_us(&ctx, || {
            selected = fixed_forward(
                &ctx,
                operands.c,
                operands.x,
                operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("automatic Fixed TF32 launch");
        });
        ctx.set_batch_invariant(false);
        ctx.set_fast_gemm(true);
        let cublas_us = average_us(&ctx, || {
            gpu_gemm_typed_forward_raw(
                &ctx,
                operands.c,
                operands.x,
                operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("cuBLAS TF32 launch");
        });
        ctx.set_batch_invariant(true);
        println!(
            "m={} k={} n={} auto={selected:?} auto_us={auto_us:.3} cublas_tf32_us={cublas_us:.3} auto_over_cublas={:.5}",
            shape.m,
            shape.k,
            shape.n,
            auto_us / cublas_us
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits specialized TF32 reference data"]
fn triad_sm120_tf32_nn_reference_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0), "SM120 required");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for shape in shapes {
        for spec in tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.op == ResolvedGemmOp::Nn)
        {
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                (shape.m, shape.k, shape.n),
                PhysicalQualificationRoute::Tf32Forced(spec.route),
            );
            let mut qualified = qualify_physical_launch(&ctx, request)
                .unwrap_or_else(|error| panic!("qualify {}: {error}", spec.symbol));
            let iterations = 200;
            let elapsed_us = qualified
                .measure_eager_window_ms(&ctx, iterations)
                .unwrap_or_else(|error| panic!("measure {}: {error}", spec.symbol))
                * 1000.0
                / iterations as f64;
            println!(
                "m={} k={} n={} specialized_symbol={} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n, spec.symbol
            );
        }
    }
}

/// One body, two families: the Fixed inference family and the Triad
/// family carry kernels of the same tile geometry. This census times each
/// pair on the same NN shapes so the slower body can be retired on evidence.
#[test]
#[ignore = "requires a quiet SM120 CUDA device and emits the Fixed/Triad pairwise census"]
fn fixed_vs_triad_pairwise_census() {
    fixed_sm120_tf32_bd_environment_preflight("pairwise census")
        .expect("pairwise census preflight");
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 3072,
        },
        FixedShape {
            m: 2048,
            k: 1536,
            n: 768,
        },
        FixedShape {
            m: 10400,
            k: 768,
            n: 384,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let sm120 = matches!(device.compute_capability, (12, 0) | (12, 1));
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let iterations = 200;
    // The half routes stage an upcast scratch that cannot grow once a
    // qualification has captured a graph: size it for every request first.
    let half_requests = shapes
        .iter()
        .flat_map(|shape| {
            [WeightDtype::Bf16, WeightDtype::F16]
                .into_iter()
                .flat_map(move |dtype| {
                    [TcTile::Tile64, TcTile::Thin16]
                        .into_iter()
                        .map(move |tile| {
                            PhysicalQualificationRequest::contiguous(
                                ResolvedGemmOp::Nn,
                                (shape.m, shape.k, shape.n),
                                PhysicalQualificationRoute::HalfForced { dtype, tile },
                            )
                        })
                })
        })
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &half_requests)
        .expect("pre-size the half qualification scratch");
    let triad_us = |request: PhysicalQualificationRequest, label: &str| -> Option<f64> {
        match qualify_physical_launch(&ctx, request) {
            Ok(mut qualified) => Some(
                qualified
                    .measure_eager_window_ms(&ctx, iterations)
                    .unwrap_or_else(|error| panic!("measure {label}: {error}"))
                    * 1000.0
                    / iterations as f64,
            ),
            Err(error) => {
                println!("triad {label}: unavailable ({error})");
                None
            }
        }
    };
    let fixed_us =
        |operands: FixedFwdOperands, shape: FixedShape, tile: FixedTile| -> Option<f64> {
            if let Err(error) = fixed_forward_with_tile(&ctx, operands, shape, tile) {
                println!("fixed {tile:?}: unavailable ({error})");
                return None;
            }
            Some(average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
            }))
        };
    let report = |shape: FixedShape, pair: &str, fixed: Option<f64>, triad: Option<f64>| {
        let verdict = match (fixed, triad) {
            (Some(fixed), Some(triad)) if fixed < triad => {
                format!("fixed faster by {:.3}x", triad / fixed)
            }
            (Some(fixed), Some(triad)) => format!("triad faster by {:.3}x", fixed / triad),
            _ => "one side unavailable".to_string(),
        };
        let show =
            |value: Option<f64>| value.map_or("n/a".to_string(), |value| format!("{value:.3}"));
        println!(
            "m={} k={} n={} pair={pair} fixed_us={} triad_us={} verdict={verdict}",
            shape.m,
            shape.k,
            shape.n,
            show(fixed),
            show(triad)
        );
    };
    for shape in shapes {
        let dims = (shape.m, shape.k, shape.n);
        // TF32: the five portable bodies and, on SM120, the five TMA bodies.
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32).expect("A");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32).expect("B");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32).expect("C");
        let f32_operands = FixedFwdOperands {
            c: typed(&c, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let portable = [
            (
                "tf32_m128n64_s2",
                FixedTile::Tf32M128S2,
                Tf32PortableTile::M128N64,
                Tf32PortableStages::S2,
            ),
            (
                "tf32_m128n64_s3",
                FixedTile::Tf32M128S3,
                Tf32PortableTile::M128N64,
                Tf32PortableStages::S3,
            ),
            (
                "tf32_m64n64_s2",
                FixedTile::Tf32M64S2,
                Tf32PortableTile::M64N64,
                Tf32PortableStages::S2,
            ),
            (
                "tf32_m64n64_s3",
                FixedTile::Tf32M64S3,
                Tf32PortableTile::M64N64,
                Tf32PortableStages::S3,
            ),
            (
                "tf32_m16n32_s4",
                FixedTile::Tf32M16S4,
                Tf32PortableTile::M16N32,
                Tf32PortableStages::S4,
            ),
        ];
        for (pair, fixed_tile, tile, stages) in portable {
            let route = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute { tile, stages });
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                dims,
                PhysicalQualificationRoute::Tf32Forced(route),
            );
            report(
                shape,
                pair,
                fixed_us(f32_operands, shape, fixed_tile),
                triad_us(request, pair),
            );
        }
        if sm120 {
            let tma = [
                (
                    "tf32_sm120_m128n64_s2",
                    FixedTile::Tf32Sm120M128S2,
                    Tf32Sm120Tile::M128N64,
                    Tf32Sm120Stages::S2,
                ),
                (
                    "tf32_sm120_m128n64_s3",
                    FixedTile::Tf32Sm120M128S3,
                    Tf32Sm120Tile::M128N64,
                    Tf32Sm120Stages::S3,
                ),
                (
                    "tf32_sm120_m64n128_s2",
                    FixedTile::Tf32Sm120M64N128S2,
                    Tf32Sm120Tile::M64N128,
                    Tf32Sm120Stages::S2,
                ),
                (
                    "tf32_sm120_m64n128_s3",
                    FixedTile::Tf32Sm120M64N128S3,
                    Tf32Sm120Tile::M64N128,
                    Tf32Sm120Stages::S3,
                ),
                (
                    "tf32_sm120_m64n64_s2",
                    FixedTile::Tf32Sm120M64S2,
                    Tf32Sm120Tile::M64N64,
                    Tf32Sm120Stages::S2,
                ),
            ];
            for (pair, fixed_tile, tile, stages) in tma {
                let route =
                    Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route { tile, stages });
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Nn,
                    dims,
                    PhysicalQualificationRoute::Tf32Forced(route),
                );
                report(
                    shape,
                    pair,
                    fixed_us(f32_operands, shape, fixed_tile),
                    triad_us(request, pair),
                );
            }
        }
        // Half: the portable 64x64 and 16x32 bodies, and on SM120 the five TMA tiles.
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C");
            let half_operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for (pair, fixed_tile, tile) in [
                ("half_tc64", FixedTile::Tc64, TcTile::Tile64),
                ("half_tc16", FixedTile::Tc16, TcTile::Thin16),
            ] {
                let pair = format!("{pair}_{dtype:?}");
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Nn,
                    dims,
                    PhysicalQualificationRoute::HalfForced { dtype, tile },
                );
                report(
                    shape,
                    &pair,
                    fixed_us(half_operands, shape, fixed_tile),
                    triad_us(request, &pair),
                );
            }
            if !sm120 {
                continue;
            }
            let caps = ctx.kernels.sm120_device_caps().expect("SM120 device caps");
            let target = ctx.kernels.sm120_target_candidate().expect("SM120 target");
            let sm120_shape = Sm120Shape::contiguous(Sm120Op::Nn, dims);
            let tiles = [
                (
                    "half_sm120_m64n64_bk64_s2",
                    FixedSm120HalfTile::M64N64Bk64S2,
                    Sm120Tile::M64N64,
                    Sm120Bk::Bk64,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m64n128_bk64_s2",
                    FixedSm120HalfTile::M64N128Bk64S2,
                    Sm120Tile::M64N128,
                    Sm120Bk::Bk64,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m128n64_bk32_s3",
                    FixedSm120HalfTile::M128N64Bk32S3,
                    Sm120Tile::M128N64,
                    Sm120Bk::Bk32,
                    Sm120Stages::S3,
                ),
                (
                    "half_sm120_m128n128_bk32_s2",
                    FixedSm120HalfTile::M128N128Bk32S2,
                    Sm120Tile::M128N128,
                    Sm120Bk::Bk32,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m128n128_bk32_s3",
                    FixedSm120HalfTile::M128N128Bk32S3,
                    Sm120Tile::M128N128,
                    Sm120Bk::Bk32,
                    Sm120Stages::S3,
                ),
            ];
            for (pair, fixed_tile, tile, bk, stages) in tiles {
                let pair = format!("{pair}_{dtype:?}");
                let physical = Sm120PhysicalRoute { tile, bk, stages };
                let triad = (|| -> Result<f64, String> {
                    let route = resolve_sm120_forced(
                        caps,
                        Some(target),
                        Sm120ForcedRoute {
                            op: Sm120Op::Nn,
                            dtype,
                            physical,
                            shape: sm120_shape,
                        },
                    )?
                    .ok_or_else(|| "route declined".to_string())?;
                    let maps = prepare_sm120_tensor_maps(
                        &ctx.stream,
                        &ctx.kernels,
                        Sm120MapRequest {
                            op: Sm120Op::Nn,
                            dtype,
                            tile,
                            bk,
                            a_ptr: half_operands.x.ptr,
                            b_ptr: half_operands.w.ptr,
                            shape: sm120_shape,
                        },
                    )?;
                    let prepared = prepare_sm120_tma_forced(
                        &ctx.stream,
                        &ctx.kernels,
                        route,
                        &maps,
                        Sm120LaunchOperands {
                            output_ptr: half_operands.c.ptr,
                            bias_ptr: 0,
                            alpha: 1.0,
                            beta: 0.0,
                        },
                    )?;
                    for _ in 0..WARMUPS {
                        launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)?;
                    }
                    let start = ctx
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(|error| format!("record start: {error:?}"))?;
                    for _ in 0..iterations {
                        launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)?;
                    }
                    let end = ctx
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(|error| format!("record end: {error:?}"))?;
                    Ok(f64::from(
                        start
                            .elapsed_ms(&end)
                            .map_err(|error| format!("measure: {error:?}"))?,
                    ) * 1000.0
                        / iterations as f64)
                })();
                let triad = match triad {
                    Ok(value) => Some(value),
                    Err(error) => {
                        println!("triad {pair}: unavailable ({error})");
                        None
                    }
                };
                report(
                    shape,
                    &pair,
                    fixed_us(half_operands, shape, FixedTile::Sm120Half(fixed_tile)),
                    triad,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn half_to_f32_portable_ladder_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            FixedShape { m: 9, k: 33, n: 17 },
            FixedShape {
                m: 65,
                k: 63,
                n: 127,
            },
            FixedShape {
                m: 128,
                k: 64,
                n: 128,
            },
            FixedShape {
                m: 4621,
                k: 384,
                n: 384,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("baseline allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("candidate allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
                .expect("B upload");
            let baseline_operands = FixedFwdOperands {
                c: typed(&baseline, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_operands = FixedFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..baseline_operands
            };
            fixed_forward_with_tile(&ctx, baseline_operands, shape, FixedTile::Legacy)
                .expect("legacy mixed-output launch");
            let baseline_bits = f32_bits(&ctx, &baseline, shape.m * shape.n);
            for tile in [FixedTile::Tc16, FixedTile::Tc64, FixedTile::Tc128] {
                fixed_forward_with_tile(&ctx, candidate_operands, shape, tile)
                    .unwrap_or_else(|error| panic!("mixed-output {tile:?} launch: {error}"));
                ctx.stream.synchronize().expect("mixed-output sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate, shape.m * shape.n),
                    baseline_bits,
                    "{dtype:?} to F32 {tile:?} changed bits for M{} K{} N{}",
                    shape.m,
                    shape.k,
                    shape.n,
                );
            }
            let selected = fixed_forward(
                &ctx,
                candidate_operands.c,
                candidate_operands.x,
                candidate_operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production mixed-output launch");
            assert!(
                matches!(
                    selected,
                    FixedTile::Tc16 | FixedTile::Tc64 | FixedTile::Tc128 | FixedTile::Sm120Half(_)
                ),
                "production mixed-output route retained {selected:?}"
            );
            ctx.stream
                .synchronize()
                .expect("production mixed-output sync");
            assert_eq!(
                f32_bits(&ctx, &candidate, shape.m * shape.n),
                baseline_bits,
                "{dtype:?} to F32 production changed bits for M{} K{} N{}",
                shape.m,
                shape.k,
                shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_to_f32_sm120_tma_ladder_matches_portable_bits() {
    const GUARD: usize = 8;
    const SENTINEL: f32 = 19.25;
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for (tile, shape) in [
        (
            FixedSm120HalfTile::M64N64Bk64S2,
            FixedShape {
                m: 65,
                k: 136,
                n: 72,
            },
        ),
        (
            FixedSm120HalfTile::M64N128Bk64S2,
            FixedShape {
                m: 65,
                k: 136,
                n: 136,
            },
        ),
        (
            FixedSm120HalfTile::M128N64Bk32S3,
            FixedShape {
                m: 129,
                k: 104,
                n: 72,
            },
        ),
        (
            FixedSm120HalfTile::M128N128Bk32S2,
            FixedShape {
                m: 129,
                k: 72,
                n: 136,
            },
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
                .expect("bias upload");

            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let output_len = shape.m * shape.n;
                let portable = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
                    .expect("portable allocation");
                let portable_operands = FixedFwdOperands {
                    c: typed(&portable, WeightDtype::F32),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                fixed_forward_with_tile(&ctx, portable_operands, shape, FixedTile::Tc128)
                    .expect("portable mixed-output launch");
                ctx.stream.synchronize().expect("portable sync");
                let expected = f32_bits(&ctx, &portable, output_len);

                for output_offset in [0usize, 1] {
                    let output_start = GUARD + output_offset;
                    let storage_len = output_start + output_len + GUARD;
                    let candidate = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
                        .expect("candidate allocation");
                    let candidate_operands = FixedFwdOperands {
                        c: TypedPtr {
                            ptr: candidate.cached_ptr()
                                + (output_start * WeightDtype::F32.size_bytes()) as u64,
                            dtype: WeightDtype::F32,
                        },
                        ..portable_operands
                    };
                    for poison in [-7.0f32, 9.0] {
                        let mut reset = vec![SENTINEL; storage_len];
                        reset[output_start..output_start + output_len].fill(poison);
                        candidate
                            .upload_f32(&ctx.stream, &reset)
                            .expect("candidate reset");
                        fixed_forward_with_tile(
                            &ctx,
                            candidate_operands,
                            shape,
                            FixedTile::Sm120Half(tile),
                        )
                        .unwrap_or_else(|error| {
                            panic!("SM120 mixed-output {dtype:?} {tile:?}: {error}")
                        });
                        ctx.stream.synchronize().expect("SM120 mixed-output sync");
                        let observed = f32_bits(&ctx, &candidate, storage_len);
                        assert!(
                            observed[..output_start]
                                .iter()
                                .all(|&bits| bits == SENTINEL.to_bits())
                        );
                        assert!(
                            observed[output_start + output_len..]
                                .iter()
                                .all(|&bits| bits == SENTINEL.to_bits())
                        );
                        assert_eq!(
                            observed[output_start..output_start + output_len],
                            expected,
                            "{dtype:?} SM120 mixed-output {tile:?} changed portable bits; bias={} offset={output_offset}",
                            bias_ptr.is_some(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_to_f32_sm120_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let dtype = WeightDtype::Bf16;
    let shape = FixedShape {
        m: 128,
        k: 96,
        n: 1536,
    };
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let reference = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("reference allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("output allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let base_operands = FixedFwdOperands {
        c: typed(&reference, WeightDtype::F32),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    fixed_forward_with_tile(&ctx, base_operands, shape, FixedTile::Tc128)
        .expect("portable reference launch");
    ctx.stream.synchronize().expect("portable reference sync");
    let expected = f32_bits(&ctx, &reference, shape.m * shape.n);

    let run = || {
        fixed_forward(
            &ctx,
            typed(&output, WeightDtype::F32),
            base_operands.x,
            base_operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| match tile {
            FixedTile::Sm120Half(_) => Ok(()),
            _ => Err(format!("unexpected production tile {tile:?}")),
        })
    };
    run().expect("warm mixed-output tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    assert_eq!(f32_bits(&ctx, &output, shape.m * shape.n), expected);
    let graph =
        unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 mixed-output");
    assert!(
        single_graph_kernel_name(&graph, "SM120 mixed-output").contains("_f32out_bf16"),
        "captured graph must retain the physical F32-output kernel"
    );
    for replay in 0..10 {
        graph
            .launch()
            .expect("launch Fixed SM120 mixed-output graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            expected,
            "Fixed SM120 mixed-output graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("cold A");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("cold B");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            fixed_forward(
                &ctx,
                typed(&output, WeightDtype::F32),
                typed(&cold_a, dtype),
                typed(&cold_b, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold mixed-output tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_sm120_exceptional_values_match_portable_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = FixedShape { m: 65, k: 8, n: 8 };
    let exceptional = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0x7f81_2345),
        65_504.0,
        -65_504.0,
        2.0f32.powi(-14),
        -2.0f32.powi(-14),
        1.0,
        -1.0,
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a_host = (0..shape.m * shape.k)
            .map(|index| exceptional[index % exceptional.len()])
            .collect::<Vec<_>>();
        let b_host = (0..shape.k * shape.n)
            .map(|index| exceptional[(index * 5 + 3) % exceptional.len()])
            .collect::<Vec<_>>();
        let bias_host = (0..shape.n)
            .map(|index| exceptional[(index * 7 + 4) % exceptional.len()])
            .collect::<Vec<_>>();
        let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), dtype).expect("B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, bias_host.len(), WeightDtype::F32)
            .expect("bias allocation");
        a.upload_f32(&ctx.stream, &a_host).expect("A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");
        for output_dtype in [dtype, WeightDtype::F32] {
            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let reference = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, output_dtype)
                    .expect("reference allocation");
                let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, output_dtype)
                    .expect("candidate allocation");
                let reference_operands = FixedFwdOperands {
                    c: typed(&reference, output_dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                fixed_forward_with_tile(&ctx, reference_operands, shape, FixedTile::Tc128)
                    .expect("portable exceptional reference");
                ctx.stream.synchronize().expect("portable exceptional sync");
                let expected = f32_bits(&ctx, &reference, shape.m * shape.n);
                for tile in FixedSm120HalfTile::ALL {
                    fixed_forward_with_tile(
                        &ctx,
                        FixedFwdOperands {
                            c: typed(&candidate, output_dtype),
                            ..reference_operands
                        },
                        shape,
                        FixedTile::Sm120Half(tile),
                    )
                    .unwrap_or_else(|error| panic!("exceptional {dtype:?} {tile:?}: {error}"));
                    ctx.stream
                        .synchronize()
                        .expect("exceptional candidate sync");
                    assert_eq!(
                        f32_bits(&ctx, &candidate, shape.m * shape.n),
                        expected,
                        "exceptional {dtype:?}->{output_dtype:?} {tile:?} bias={} changed portable bits",
                        bias_ptr.is_some(),
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM120 CUDA device and emits mixed-output TMA data"]
fn half_to_f32_sm120_tma_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        // The projection shapes the performance matrix measures: this is
        // where the automatic selector still falls back to the older tensor
        // core tiles instead of the TMA ones.
        FixedShape {
            m: 2048,
            k: 768,
            n: 3072,
        },
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 2048,
            k: 1536,
            n: 768,
        },
        FixedShape {
            m: 4096,
            k: 3072,
            n: 1536,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("output allocation");
            let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("cuBLAS allocation");
            let operands = FixedFwdOperands {
                c: typed(&output, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let mut auto_tile = FixedTile::Legacy;
            let auto_us = average_us(&ctx, || {
                auto_tile = fixed_forward(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("automatic mixed-output launch");
            });
            let tma_us = FixedSm120HalfTile::ALL.map(|tile| {
                average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Sm120Half(tile))
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                })
            });
            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(true);
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    typed(&cublas, WeightDtype::F32),
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS mixed-output launch");
            });
            ctx.set_batch_invariant(true);
            ctx.set_fast_gemm(false);
            println!(
                "dtype={dtype:?} m={} k={} n={} auto={auto_tile:?} auto_us={auto_us:.3} tma_m64n64_us={:.3} tma_m64n128_us={:.3} tma_m128n64_us={:.3} tma_m128n128_s2_us={:.3} tma_m128n128_s3_us={:.3} cublas_us={cublas_us:.3} auto_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                tma_us[0],
                tma_us[1],
                tma_us[2],
                tma_us[3],
                tma_us[4],
                auto_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM SM120 CUDA device and screens mixed-output hot B"]
fn half_to_f32_sm120_hot_b_paired_tile_screen() {
    fixed_sm120_tf32_bd_environment_preflight("mixed-output hot B")
        .expect("mixed-output hot B preflight");
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = FixedShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let vendor = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("vendor allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
            .expect("B upload");
        let candidate_operands = FixedFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        let vendor_operands = FixedFwdOperands {
            c: typed(&vendor, WeightDtype::F32),
            ..candidate_operands
        };
        for tile in FixedSm120HalfTile::ALL {
            for _ in 0..128 {
                fixed_forward_with_tile(
                    &ctx,
                    candidate_operands,
                    shape,
                    FixedTile::Sm120Half(tile),
                )
                .unwrap_or_else(|error| panic!("warm {tile:?}: {error}"));
            }
            ctx.stream.synchronize().expect("candidate warmup sync");
            configure_fixed_auto_vendor_vendor(&ctx, F32TriadPolicy::ExactScalarFmaV1);
            for _ in 0..128 {
                launch_fixed_auto_vendor_vendor(&ctx, vendor_operands, shape);
            }
            ctx.stream.synchronize().expect("vendor warmup sync");
            let candidate_iterations = fixed_tile_window_iterations(
                &ctx,
                candidate_operands,
                shape,
                FixedTile::Sm120Half(tile),
            );
            let vendor_iterations =
                fixed_auto_vendor_iterations(fixed_auto_vendor_vendor_window_us(
                    &ctx,
                    vendor_operands,
                    shape,
                    F32TriadPolicy::ExactScalarFmaV1,
                    16,
                ));
            for candidate_first in [true, false] {
                let mut candidate_us = Vec::with_capacity(101);
                let mut vendor_us = Vec::with_capacity(101);
                let mut ratios = Vec::with_capacity(101);
                for _ in 0..101 {
                    let (candidate_elapsed, vendor_elapsed) = if candidate_first {
                        (
                            fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                FixedTile::Sm120Half(tile),
                                candidate_iterations,
                            ),
                            fixed_auto_vendor_vendor_window_us(
                                &ctx,
                                vendor_operands,
                                shape,
                                F32TriadPolicy::ExactScalarFmaV1,
                                vendor_iterations,
                            ),
                        )
                    } else {
                        let vendor_elapsed = fixed_auto_vendor_vendor_window_us(
                            &ctx,
                            vendor_operands,
                            shape,
                            F32TriadPolicy::ExactScalarFmaV1,
                            vendor_iterations,
                        );
                        let candidate_elapsed = fixed_tile_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            FixedTile::Sm120Half(tile),
                            candidate_iterations,
                        );
                        (candidate_elapsed, vendor_elapsed)
                    };
                    candidate_us.push(candidate_elapsed);
                    vendor_us.push(vendor_elapsed);
                    ratios.push(candidate_elapsed / vendor_elapsed);
                }
                candidate_us.sort_by(f64::total_cmp);
                vendor_us.sort_by(f64::total_cmp);
                ratios.sort_by(f64::total_cmp);
                println!(
                    "dtype={dtype:?} tile={tile:?} order={} candidate_p50_us={:.6} vendor_p50_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
                    if candidate_first {
                        "candidate_then_vendor"
                    } else {
                        "vendor_then_candidate"
                    },
                    percentile(&candidate_us, 0.50),
                    percentile(&vendor_us, 0.50),
                    percentile(&ratios, 0.50),
                    percentile(&ratios, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM SM120 CUDA device and diagnoses AUTO launch overhead"]
fn half_to_f32_sm120_hot_b_auto_vs_forced() {
    fixed_sm120_tf32_bd_environment_preflight("mixed-output hot B AUTO versus forced")
        .expect("mixed-output hot B AUTO/forced preflight");
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = FixedShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    let tile = FixedTile::Sm120Half(FixedSm120HalfTile::M64N128Bk64S2);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let auto_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("AUTO output allocation");
        let forced_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("forced output allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
            .expect("B upload");
        let auto_operands = FixedFwdOperands {
            c: typed(&auto_output, WeightDtype::F32),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        let forced_operands = FixedFwdOperands {
            c: typed(&forced_output, WeightDtype::F32),
            ..auto_operands
        };
        for _ in 0..128 {
            launch_fixed_auto_vendor_custom(&ctx, auto_operands, shape);
            fixed_forward_with_tile(&ctx, forced_operands, shape, tile)
                .expect("forced warmup launch");
        }
        ctx.stream.synchronize().expect("AUTO/forced warmup sync");
        let auto_iterations = fixed_auto_vendor_iterations(fixed_auto_vendor_custom_window_us(
            &ctx,
            auto_operands,
            shape,
            F32TriadPolicy::ExactScalarFmaV1,
            16,
        ));
        let forced_iterations = fixed_tile_window_iterations(&ctx, forced_operands, shape, tile);
        for auto_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            let mut auto_us = Vec::with_capacity(101);
            let mut forced_us = Vec::with_capacity(101);
            for _ in 0..101 {
                let (auto_elapsed, forced_elapsed) = if auto_first {
                    (
                        fixed_auto_vendor_custom_window_us(
                            &ctx,
                            auto_operands,
                            shape,
                            F32TriadPolicy::ExactScalarFmaV1,
                            auto_iterations,
                        ),
                        fixed_tile_window_us(&ctx, forced_operands, shape, tile, forced_iterations),
                    )
                } else {
                    let forced_elapsed =
                        fixed_tile_window_us(&ctx, forced_operands, shape, tile, forced_iterations);
                    let auto_elapsed = fixed_auto_vendor_custom_window_us(
                        &ctx,
                        auto_operands,
                        shape,
                        F32TriadPolicy::ExactScalarFmaV1,
                        auto_iterations,
                    );
                    (auto_elapsed, forced_elapsed)
                };
                auto_us.push(auto_elapsed);
                forced_us.push(forced_elapsed);
                ratios.push(auto_elapsed / forced_elapsed);
            }
            auto_us.sort_by(f64::total_cmp);
            forced_us.sort_by(f64::total_cmp);
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} order={} auto_p50_us={:.6} forced_p50_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
                if auto_first {
                    "auto_then_forced"
                } else {
                    "forced_then_auto"
                },
                percentile(&auto_us, 0.50),
                percentile(&forced_us, 0.50),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output ladder data"]
fn half_to_f32_portable_ladder_smoke() {
    let shapes = [
        FixedShape {
            m: 1,
            k: 768,
            n: 17,
        },
        FixedShape { m: 9, k: 33, n: 17 },
        FixedShape {
            m: 64,
            k: 768,
            n: 24,
        },
        FixedShape {
            m: 65,
            k: 63,
            n: 127,
        },
        FixedShape {
            m: 128,
            k: 64,
            n: 128,
        },
        FixedShape {
            m: 512,
            k: 768,
            n: 512,
        },
        FixedShape {
            m: 4621,
            k: 384,
            n: 384,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for tile in [
                FixedTile::Legacy,
                FixedTile::Tc16,
                FixedTile::Tc64,
                FixedTile::Tc128,
            ] {
                let elapsed_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    shape.m, shape.k, shape.n,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output selector data"]
fn half_to_f32_portable_selector_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [
            1usize, 16, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4621,
        ] {
            for n in [17usize, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2304] {
                let shape = FixedShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("C allocation");
                let operands = FixedFwdOperands {
                    c: typed(&c, WeightDtype::F32),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                for tile in [FixedTile::Tc16, FixedTile::Tc64, FixedTile::Tc128] {
                    let elapsed_us = average_us(&ctx, || {
                        fixed_forward_with_tile(&ctx, operands, shape, tile)
                            .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                    });
                    println!(
                        "dtype={dtype:?} m={m} k=768 n={n} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output K-axis data"]
fn half_to_f32_portable_k_axis_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [2048usize, 4621] {
            for n in [384usize, 768, 1928, 2304] {
                for k in [384usize, 768, 1928, 2304] {
                    let shape = FixedShape { m, k, n };
                    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype)
                        .expect("A allocation");
                    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype)
                        .expect("B allocation");
                    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                        .expect("C allocation");
                    let operands = FixedFwdOperands {
                        c: typed(&c, WeightDtype::F32),
                        x: typed(&a, dtype),
                        w: typed(&b, dtype),
                        bias_ptr: None,
                    };
                    for tile in [FixedTile::Tc64, FixedTile::Tc128] {
                        let elapsed_us = average_us(&ctx, || {
                            fixed_forward_with_tile(&ctx, operands, shape, tile)
                                .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                        });
                        println!(
                            "dtype={dtype:?} m={m} k={k} n={n} tile={tile:?} elapsed_us={elapsed_us:.3}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn tcw64_matches_tc128_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            FixedShape {
                m: 65,
                k: 63,
                n: 127,
            },
            FixedShape {
                m: 128,
                k: 64,
                n: 128,
            },
            FixedShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let square =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("Tc128 allocation");
            let reuse =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("TcW64 allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
                .expect("B upload");
            let operands = FixedFwdOperands {
                c: typed(&square, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc128).expect("Tc128 launch");
            fixed_forward_with_tile(
                &ctx,
                FixedFwdOperands {
                    c: typed(&reuse, dtype),
                    ..operands
                },
                shape,
                FixedTile::TcW64,
            )
            .expect("TcW64 launch");
            ctx.stream.synchronize().expect("W64 comparison sync");
            assert_eq!(
                f32_bits(&ctx, &reuse, shape.m * shape.n),
                f32_bits(&ctx, &square, shape.m * shape.n),
                "{dtype:?} TcW64 changed bits for M{} K{} N{}",
                shape.m,
                shape.k,
                shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_portable_half_epilogues_are_tile_and_alignment_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            FixedShape {
                m: 15,
                k: 63,
                n: 31,
            },
            FixedShape {
                m: 17,
                k: 64,
                n: 32,
            },
            FixedShape {
                m: 65,
                k: 96,
                n: 65,
            },
            FixedShape {
                m: 129,
                k: 97,
                n: 132,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xe91a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xe91b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0xe91c))
                .expect("bias upload");

            for bias_ptr in [None, Some(bias.cached_ptr())] {
                for output_offset in [0usize, 1] {
                    let elements = shape.m * shape.n;
                    let reference = DtypedBuf::zeros(&ctx.stream, elements + 1, dtype)
                        .expect("reference allocation");
                    let reference_ptr = TypedPtr {
                        ptr: reference.cached_ptr() + (output_offset * dtype.size_bytes()) as u64,
                        dtype,
                    };
                    let base_operands = FixedFwdOperands {
                        c: reference_ptr,
                        x: typed(&a, dtype),
                        w: typed(&b, dtype),
                        bias_ptr,
                    };
                    fixed_forward_with_tile(&ctx, base_operands, shape, FixedTile::Tc128)
                        .expect("Tc128 reference");
                    ctx.stream.synchronize().expect("reference sync");
                    let expected = f32_bits(&ctx, &reference, elements + 1)[output_offset..]
                        [..elements]
                        .to_vec();

                    for tile in [FixedTile::Tc16, FixedTile::Tc64] {
                        let output = DtypedBuf::zeros(&ctx.stream, elements + 1, dtype)
                            .expect("candidate allocation");
                        let operands = FixedFwdOperands {
                            c: TypedPtr {
                                ptr: output.cached_ptr()
                                    + (output_offset * dtype.size_bytes()) as u64,
                                dtype,
                            },
                            ..base_operands
                        };
                        fixed_forward_with_tile(&ctx, operands, shape, tile)
                            .unwrap_or_else(|error| panic!("{tile:?} launch: {error}"));
                        ctx.stream.synchronize().expect("candidate sync");
                        assert_eq!(
                            f32_bits(&ctx, &output, elements + 1)[output_offset..][..elements],
                            expected,
                            "{dtype:?} {tile:?} epilogue drift for {shape:?} bias={} offset={output_offset}",
                            bias_ptr.is_some()
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_tma_matches_portable_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            FixedShape {
                m: 65,
                k: 40,
                n: 72,
            },
            FixedShape {
                m: 129,
                k: 96,
                n: 136,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
                .expect("bias upload");
            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let reference =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("reference");
                let reference_operands = FixedFwdOperands {
                    c: typed(&reference, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                fixed_forward_with_tile(&ctx, reference_operands, shape, FixedTile::Tc128)
                    .expect("portable reference launch");
                ctx.stream.synchronize().expect("portable reference sync");
                let expected = f32_bits(&ctx, &reference, shape.m * shape.n);
                for candidate in FixedSm120HalfTile::ALL {
                    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                        .expect("candidate output");
                    let operands = FixedFwdOperands {
                        c: typed(&output, dtype),
                        ..reference_operands
                    };
                    let tile = FixedTile::Sm120Half(candidate);
                    fixed_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("first {candidate:?}: {error}"));
                    ctx.stream.synchronize().expect("first candidate sync");
                    let first = f32_bits(&ctx, &output, shape.m * shape.n);
                    fixed_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("repeat {candidate:?}: {error}"));
                    ctx.stream.synchronize().expect("repeat candidate sync");
                    let second = f32_bits(&ctx, &output, shape.m * shape.n);
                    assert_eq!(first, second, "repeat drift for {dtype:?} {candidate:?}");
                    assert_eq!(
                        first,
                        expected,
                        "portable drift for {dtype:?} {shape:?} {candidate:?} bias={}",
                        bias_ptr.is_some(),
                    );
                }
            }
        }
    }
}

fn fixed_sm120_half_symbol(tile: FixedSm120HalfTile, dtype: WeightDtype) -> &'static str {
    match (tile, dtype) {
        (FixedSm120HalfTile::M64N64Bk64S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_bf16"
        }
        (FixedSm120HalfTile::M64N64Bk64S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_f16"
        }
        (FixedSm120HalfTile::M64N128Bk64S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_bf16"
        }
        (FixedSm120HalfTile::M64N128Bk64S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_f16"
        }
        (FixedSm120HalfTile::M128N64Bk32S3, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_bf16"
        }
        (FixedSm120HalfTile::M128N64Bk32S3, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_f16"
        }
        (FixedSm120HalfTile::M128N128Bk32S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_bf16"
        }
        (FixedSm120HalfTile::M128N128Bk32S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_f16"
        }
        (FixedSm120HalfTile::M128N128Bk32S3, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_bf16"
        }
        (FixedSm120HalfTile::M128N128Bk32S3, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_f16"
        }
        (_, WeightDtype::F32) => panic!("SM120 half symbol requested for F32"),
    }
}

fn run_fixed_sm120_half_exact_overlay_route_case(
    ctx: &GpuCtx,
    label: &str,
    dtype: WeightDtype,
    shape: FixedShape,
    has_bias: bool,
    promoted: FixedSm120HalfTile,
    incumbent: FixedSm120HalfTile,
) {
    let elements = shape.m * shape.n;
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
    a.upload_f32(
        &ctx.stream,
        &synth(shape.m * shape.k, 0xa132 ^ shape.m as u64),
    )
    .expect("A upload");
    b.upload_f32(
        &ctx.stream,
        &synth(shape.k * shape.n, 0xb132 ^ shape.n as u64),
    )
    .expect("B upload");
    bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5 ^ shape.k as u64))
        .expect("bias upload");
    let bias_ptr = has_bias.then(|| bias.cached_ptr());

    let forced = DtypedBuf::zeros(&ctx.stream, elements, dtype).expect("forced output");
    let production = DtypedBuf::zeros(&ctx.stream, elements, dtype).expect("AUTO output");
    let forced_operands = FixedFwdOperands {
        c: typed(&forced, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr,
    };
    let production_operands = FixedFwdOperands {
        c: typed(&production, dtype),
        ..forced_operands
    };
    fixed_forward_with_tile(ctx, forced_operands, shape, FixedTile::Sm120Half(incumbent))
        .expect("forced pre-overlay incumbent");
    let selected = fixed_forward(
        ctx,
        production_operands.c,
        production_operands.x,
        production_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production AUTO launch");
    assert_eq!(
        selected,
        FixedTile::Sm120Half(promoted),
        "{label} AUTO tile"
    );
    ctx.stream.synchronize().expect("eager synchronization");
    let expected = f32_bits(ctx, &forced, elements);
    assert_eq!(
        f32_bits(ctx, &production, elements),
        expected,
        "{label} eager bits"
    );

    let repeated = fixed_forward(
        ctx,
        production_operands.c,
        production_operands.x,
        production_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production AUTO repeat");
    assert_eq!(
        repeated,
        FixedTile::Sm120Half(promoted),
        "{label} repeat tile"
    );
    ctx.stream.synchronize().expect("repeat synchronization");
    assert_eq!(
        f32_bits(ctx, &production, elements),
        expected,
        "{label} repeat bits"
    );

    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            fixed_forward(
                ctx,
                production_operands.c,
                production_operands.x,
                production_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .and_then(|tile| {
                (tile == FixedTile::Sm120Half(promoted))
                    .then_some(())
                    .ok_or_else(|| format!("{label} graph selected {tile:?}"))
            })
        })
    }
    .expect("capture production AUTO graph");
    assert_eq!(
        single_graph_kernel_name(&graph, label),
        fixed_sm120_half_symbol(promoted, dtype),
        "{label} promoted physical route"
    );
    for replay in 0..10 {
        graph.launch().expect("production graph replay");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(ctx, &production, elements),
            expected,
            "{label} graph replay {replay} bits"
        );
    }

    let misaligned =
        DtypedBuf::zeros(&ctx.stream, elements + 1, dtype).expect("misaligned AUTO output");
    let misaligned_operands = FixedFwdOperands {
        c: TypedPtr {
            ptr: misaligned.cached_ptr() + dtype.size_bytes() as u64,
            dtype,
        },
        ..forced_operands
    };
    assert_eq!(misaligned_operands.c.ptr & 3, 2, "{label} C+2 setup");
    let misaligned_tile = fixed_forward(
        ctx,
        misaligned_operands.c,
        misaligned_operands.x,
        misaligned_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("misaligned AUTO launch");
    assert_eq!(
        misaligned_tile,
        FixedTile::Sm120Half(incumbent),
        "{label} C+2 fallback tile"
    );
    ctx.stream
        .synchronize()
        .expect("misaligned eager synchronization");
    let misaligned_bits = f32_bits(ctx, &misaligned, elements + 1);
    assert_eq!(&misaligned_bits[1..], expected, "{label} C+2 fallback bits");
    let misaligned_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            fixed_forward(
                ctx,
                misaligned_operands.c,
                misaligned_operands.x,
                misaligned_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    }
    .expect("capture C+2 fallback graph");
    assert_eq!(
        single_graph_kernel_name(&misaligned_graph, label),
        fixed_sm120_half_symbol(incumbent, dtype),
        "{label} C+2 physical fallback"
    );
}

#[test]
#[ignore = "requires an RTX5090 with the Fixed module loaded by NVRTC 13.2"]
fn fixed_sm120_half_exact_selector_routes_match_incumbent_bits_and_graphs() {
    use FixedSm120HalfTile::{
        M64N64Bk64S2 as C, M64N128Bk64S2 as D, M128N64Bk32S3 as A, M128N128Bk32S2 as B,
    };

    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);

    for (label, dtype, shape, has_bias, expected) in [
        (
            "A1_bf16_none_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A1_bf16_bias_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "A1_f16_none_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A1_f16_bias_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "A2_bf16_bias_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            true,
            C,
        ),
        (
            "A2_f16_bias_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            true,
            C,
        ),
        (
            "A3_f16_none_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A3_f16_bias_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "B_bf16_none_promoted",
            WeightDtype::Bf16,
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            false,
            D,
        ),
        (
            "B_bf16_bias_promoted",
            WeightDtype::Bf16,
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            true,
            D,
        ),
        (
            "B_f16_none_promoted",
            WeightDtype::F16,
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            false,
            D,
        ),
        (
            "B_f16_bias_promoted",
            WeightDtype::F16,
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            true,
            D,
        ),
        (
            "A2_bf16_none_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            false,
            C,
        ),
        (
            "A2_f16_none_rejected",
            WeightDtype::F16,
            FixedShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            false,
            C,
        ),
        (
            "A3_bf16_none_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A3_bf16_bias_rejected",
            WeightDtype::Bf16,
            FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
    ] {
        run_fixed_sm120_half_exact_overlay_route_case(
            &ctx, label, dtype, shape, has_bias, expected, C,
        );
    }
    for (shape_name, shape, incumbent) in [
        (
            "m1024_k1928_n1928",
            FixedShape {
                m: 1024,
                k: 1928,
                n: 1928,
            },
            B,
        ),
        (
            "m1024_k1928_n2304",
            FixedShape {
                m: 1024,
                k: 1928,
                n: 2304,
            },
            B,
        ),
        (
            "m1536_k1032_n1536",
            FixedShape {
                m: 1536,
                k: 1032,
                n: 1536,
            },
            B,
        ),
        (
            "m1536_k1928_n1536",
            FixedShape {
                m: 1536,
                k: 1928,
                n: 1536,
            },
            B,
        ),
        (
            "m4621_k1928_n1928",
            FixedShape {
                m: 4621,
                k: 1928,
                n: 1928,
            },
            B,
        ),
        (
            "m1536_k768_n1536",
            FixedShape {
                m: 1536,
                k: 768,
                n: 1536,
            },
            A,
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for has_bias in [false, true] {
                let label = format!(
                    "{shape_name}_{}_{}",
                    dtype.as_str(),
                    if has_bias { "bias" } else { "none" }
                );
                run_fixed_sm120_half_exact_overlay_route_case(
                    &ctx, &label, dtype, shape, has_bias, D, incumbent,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_retained_tiles_guarded_tails_are_bit_exact() {
    const GUARD: usize = 64;
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required",
    );
    assert!(
        ctx.kernels.gemm_bi_nn_half_sm120.is_some(),
        "Fixed SM120 half module must be loaded",
    );
    assert_eq!(
        FixedSm120HalfTile::ALL,
        [
            FixedSm120HalfTile::M64N64Bk64S2,
            FixedSm120HalfTile::M64N128Bk64S2,
            FixedSm120HalfTile::M128N64Bk32S3,
            FixedSm120HalfTile::M128N128Bk32S2,
            FixedSm120HalfTile::M128N128Bk32S3,
        ],
    );
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);

    for (tile, shape, geometry) in [
        (
            FixedSm120HalfTile::M64N64Bk64S2,
            FixedShape {
                m: 65,
                k: 136,
                n: 72,
            },
            (64usize, 64usize, 64usize, 2usize),
        ),
        (
            FixedSm120HalfTile::M128N64Bk32S3,
            FixedShape {
                m: 129,
                k: 104,
                n: 72,
            },
            (128, 64, 32, 3),
        ),
        (
            FixedSm120HalfTile::M64N128Bk64S2,
            FixedShape {
                m: 65,
                k: 136,
                n: 136,
            },
            (64, 128, 64, 2),
        ),
        (
            FixedSm120HalfTile::M128N128Bk32S2,
            FixedShape {
                m: 129,
                k: 72,
                n: 136,
            },
            (128, 128, 32, 2),
        ),
        (
            FixedSm120HalfTile::M128N128Bk32S3,
            FixedShape {
                m: 129,
                k: 104,
                n: 136,
            },
            (128, 128, 32, 3),
        ),
    ] {
        let (bm, bn, bk, stages) = geometry;
        assert_ne!(shape.m % bm, 0);
        assert_ne!(shape.k % bk, 0);
        assert_ne!(shape.n % bn, 0);
        assert!(shape.k.div_ceil(bk) > stages);

        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a_len = shape.m * shape.k;
            let b_len = shape.k * shape.n;
            let output_len = shape.m * shape.n;
            let mut a_host = vec![13.0f32; GUARD + a_len + GUARD];
            let mut b_host = vec![-11.0f32; GUARD + b_len + GUARD];
            a_host[GUARD..GUARD + a_len].copy_from_slice(&synth(a_len, 0xa11ce ^ shape.m as u64));
            b_host[GUARD..GUARD + b_len].copy_from_slice(&synth(b_len, 0xb0b ^ shape.n as u64));
            let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), dtype).expect("guarded A");
            let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), dtype).expect("guarded B");
            a.upload_f32(&ctx.stream, &a_host).expect("A upload");
            b.upload_f32(&ctx.stream, &b_host).expect("B upload");
            let a_ptr = a.cached_ptr() + (GUARD * dtype.size_bytes()) as u64;
            let b_ptr = b.cached_ptr() + (GUARD * dtype.size_bytes()) as u64;
            assert_eq!(a_ptr & 15, 0);
            assert_eq!(b_ptr & 15, 0);
            let mut a_before = vec![0.0f32; a_host.len()];
            let mut b_before = vec![0.0f32; b_host.len()];
            a.download_f32(&ctx.stream, &mut a_before)
                .expect("A snapshot");
            b.download_f32(&ctx.stream, &mut b_before)
                .expect("B snapshot");

            let mut bias_host = vec![23.5f32; GUARD + shape.n + GUARD];
            bias_host[GUARD..GUARD + shape.n]
                .copy_from_slice(&synth(shape.n, 0xb1a5 ^ shape.k as u64));
            let bias = DtypedBuf::zeros(&ctx.stream, bias_host.len(), WeightDtype::F32)
                .expect("guarded bias");
            bias.upload_f32(&ctx.stream, &bias_host)
                .expect("bias upload");
            let bias_ptr = bias.cached_ptr() + (GUARD * WeightDtype::F32.size_bytes()) as u64;
            let mut bias_before = vec![0.0f32; bias_host.len()];
            bias.download_f32(&ctx.stream, &mut bias_before)
                .expect("bias snapshot");
            let mut unbiased_reference: Option<Vec<u32>> = None;

            for candidate_bias in [None, Some(bias_ptr)] {
                let reference =
                    DtypedBuf::zeros(&ctx.stream, output_len, dtype).expect("portable reference");
                let reference_operands = FixedFwdOperands {
                    c: typed(&reference, dtype),
                    x: TypedPtr { ptr: a_ptr, dtype },
                    w: TypedPtr { ptr: b_ptr, dtype },
                    bias_ptr: candidate_bias,
                };
                fixed_forward_with_tile(&ctx, reference_operands, shape, FixedTile::Tc128)
                    .expect("portable reference launch");
                ctx.stream.synchronize().expect("portable reference sync");
                let expected = f32_bits(&ctx, &reference, output_len);
                assert!(expected.iter().any(|&bits| bits != 0));
                if let Some(unbiased) = &unbiased_reference {
                    assert!(
                        expected
                            .iter()
                            .zip(unbiased)
                            .any(|(left, right)| left != right),
                        "bias must change at least one output for {dtype:?} {tile:?}",
                    );
                } else {
                    unbiased_reference = Some(expected.clone());
                }

                for offset in [0usize, GUARD + 1] {
                    let storage_len = offset + output_len + GUARD;
                    let candidate = DtypedBuf::zeros(&ctx.stream, storage_len, dtype)
                        .expect("guarded candidate output");
                    let candidate_ptr =
                        candidate.cached_ptr() + (offset * dtype.size_bytes()) as u64;
                    if offset == 0 {
                        assert_eq!(candidate_ptr & 3, 0);
                    } else {
                        assert_eq!(candidate_ptr & 3, 2);
                    }
                    let candidate_operands = FixedFwdOperands {
                        c: TypedPtr {
                            ptr: candidate_ptr,
                            dtype,
                        },
                        ..reference_operands
                    };
                    let mut first_bits = None;
                    for poison in [-7.0f32, 9.0f32] {
                        let mut reset = vec![19.25f32; storage_len];
                        reset[offset..offset + output_len].fill(poison);
                        candidate
                            .upload_f32(&ctx.stream, &reset)
                            .expect("candidate reset");
                        fixed_forward_with_tile(
                            &ctx,
                            candidate_operands,
                            shape,
                            FixedTile::Sm120Half(tile),
                        )
                        .expect("SM120 guarded launch");
                        ctx.stream.synchronize().expect("SM120 guarded sync");
                        let mut observed = vec![0.0f32; storage_len];
                        candidate
                            .download_f32(&ctx.stream, &mut observed)
                            .expect("candidate download");
                        assert!(
                            observed[..offset]
                                .iter()
                                .all(|value| value.to_bits() == 19.25f32.to_bits())
                        );
                        assert!(
                            observed[offset + output_len..]
                                .iter()
                                .all(|value| value.to_bits() == 19.25f32.to_bits())
                        );
                        let logical_bits = observed[offset..offset + output_len]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>();
                        assert_eq!(logical_bits, expected, "{dtype:?} {tile:?} offset={offset}");
                        if let Some(first) = &first_bits {
                            assert_eq!(&logical_bits, first, "repeat drift for {dtype:?} {tile:?}");
                        } else {
                            first_bits = Some(logical_bits);
                        }
                        let mut a_after = vec![0.0f32; a_before.len()];
                        let mut b_after = vec![0.0f32; b_before.len()];
                        let mut bias_after = vec![0.0f32; bias_before.len()];
                        a.download_f32(&ctx.stream, &mut a_after).expect("A verify");
                        b.download_f32(&ctx.stream, &mut b_after).expect("B verify");
                        bias.download_f32(&ctx.stream, &mut bias_after)
                            .expect("bias verify");
                        assert_eq!(a_after, a_before, "A mutated for {dtype:?} {tile:?}");
                        assert_eq!(b_after, b_before, "B mutated for {dtype:?} {tile:?}");
                        assert_eq!(
                            bias_after, bias_before,
                            "bias mutated for {dtype:?} {tile:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let dtype = WeightDtype::Bf16;
    let shape = FixedShape {
        m: 128,
        k: 96,
        n: 1536,
    };
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("output");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let run = || {
        fixed_forward(
            &ctx,
            typed(&output, dtype),
            typed(&a, dtype),
            typed(&b, dtype),
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| match tile {
            FixedTile::Sm120Half(_) => Ok(()),
            _ => Err(format!("unexpected production tile {tile:?}")),
        })
    };
    run().expect("warm half tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    let eager = f32_bits(&ctx, &output, shape.m * shape.n);
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 half");
    for replay in 0..10 {
        graph.launch().expect("launch Fixed SM120 half graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "Fixed SM120 half graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("cold A");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("cold B");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            fixed_forward(
                &ctx,
                typed(&output, dtype),
                typed(&cold_a, dtype),
                typed(&cold_b, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold half tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_graph_warmup_retains_more_than_32_live_maps() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let shape = FixedShape {
        m: 128,
        k: 96,
        n: 1536,
    };

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let output =
            DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("output allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xcac4e))
            .expect("A upload");
        let mut weights = Vec::with_capacity(33);
        for index in 0..33_u64 {
            let weight =
                DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("weight allocation");
            weight
                .upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120_000 + index))
                .expect("weight upload");
            weights.push(weight);
        }

        for weight in &weights {
            let tile = fixed_forward(
                &ctx,
                typed(&output, dtype),
                typed(&a, dtype),
                typed(weight, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("warm tensor map");
            assert!(matches!(tile, FixedTile::Sm120Half(_)));
        }
        ctx.stream.synchronize().expect("warmup synchronization");
        let eager = f32_bits(&ctx, &output, shape.m * shape.n);

        let graph = unsafe {
            capture_into_graph(&ctx.stream, || {
                for weight in &weights {
                    let tile = fixed_forward(
                        &ctx,
                        typed(&output, dtype),
                        typed(&a, dtype),
                        typed(weight, dtype),
                        None,
                        (shape.m, shape.k, shape.n),
                    )?;
                    if !matches!(tile, FixedTile::Sm120Half(_)) {
                        return Err(format!("unexpected production tile {tile:?}"));
                    }
                }
                Ok(())
            })
        }
        .expect("capture more than 32 live tensor maps");
        graph.launch().expect("launch multi-map graph");
        ctx.stream.synchronize().expect("multi-map graph sync");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "{dtype:?} graph replay changed bits after 33 live map keys"
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_is_batch_prefix_invariant_across_selectors() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let (small_m, large_m, k, n) = (64usize, 512usize, 384usize, 1536usize);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let large_a_host = synth(large_m * k, 0xba7c4);
        let b_host = synth(k * n, 0xb32);
        let bias_host = synth(n, 0xb1a5);
        let small_a = DtypedBuf::zeros(&ctx.stream, small_m * k, dtype).expect("small A");
        let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, dtype).expect("large A");
        let b = DtypedBuf::zeros(&ctx.stream, k * n, dtype).expect("B");
        let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
        let small_c = DtypedBuf::zeros(&ctx.stream, small_m * n, dtype).expect("small C");
        let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, dtype).expect("large C");
        small_a
            .upload_f32(&ctx.stream, &large_a_host[..small_m * k])
            .expect("small A upload");
        large_a
            .upload_f32(&ctx.stream, &large_a_host)
            .expect("large A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");
        let small_tile = fixed_forward(
            &ctx,
            typed(&small_c, dtype),
            typed(&small_a, dtype),
            typed(&b, dtype),
            Some(bias.cached_ptr()),
            (small_m, k, n),
        )
        .expect("small launch");
        let large_tile = fixed_forward(
            &ctx,
            typed(&large_c, dtype),
            typed(&large_a, dtype),
            typed(&b, dtype),
            Some(bias.cached_ptr()),
            (large_m, k, n),
        )
        .expect("large launch");
        ctx.stream.synchronize().expect("prefix sync");
        assert_ne!(small_tile, large_tile, "test must cross selector rungs");
        assert!(
            matches!(large_tile, FixedTile::Sm120Half(_)),
            "large shape did not select SM120 TMA: {large_tile:?}"
        );
        assert_eq!(
            f32_bits(&ctx, &small_c, small_m * n),
            f32_bits(&ctx, &large_c, large_m * n)[..small_m * n],
            "{dtype:?} prefix bits changed across selectors"
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_half_narrow_n_uses_portable_numeric_family() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let (small_m, large_m, k, n) = (16usize, 8192usize, 768usize, 24usize);

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let large_a_host = synth(large_m * k, 0xba7c4);
        let b_host = synth(k * n, 0xb32);
        let bias_host = synth(n, 0xb1a5);
        let small_a = DtypedBuf::zeros(&ctx.stream, small_m * k, dtype).expect("small A");
        let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, dtype).expect("large A");
        let b = DtypedBuf::zeros(&ctx.stream, k * n, dtype).expect("B");
        let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
        small_a
            .upload_f32(&ctx.stream, &large_a_host[..small_m * k])
            .expect("small A upload");
        large_a
            .upload_f32(&ctx.stream, &large_a_host)
            .expect("large A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");

        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let small_c = DtypedBuf::zeros(&ctx.stream, small_m * n, dtype).expect("small C");
            let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, dtype).expect("large C");
            let small_tile = fixed_forward(
                &ctx,
                typed(&small_c, dtype),
                typed(&small_a, dtype),
                typed(&b, dtype),
                bias_ptr,
                (small_m, k, n),
            )
            .expect("small narrow-N launch");
            let large_tile = fixed_forward(
                &ctx,
                typed(&large_c, dtype),
                typed(&large_a, dtype),
                typed(&b, dtype),
                bias_ptr,
                (large_m, k, n),
            )
            .expect("large narrow-N launch");
            ctx.stream.synchronize().expect("narrow-N prefix sync");
            assert_eq!(small_tile, FixedTile::Tc16);
            assert_eq!(large_tile, FixedTile::Tc16);
            assert_eq!(
                f32_bits(&ctx, &small_c, small_m * n),
                f32_bits(&ctx, &large_c, large_m * n)[..small_m * n],
                "{dtype:?} narrow-N prefix bits changed with bias={}",
                bias_ptr.is_some()
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits narrow-N performance data"]
fn fixed_half_narrow_n_performance_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            FixedShape {
                m: 1,
                k: 768,
                n: 17,
            },
            FixedShape {
                m: 16,
                k: 768,
                n: 24,
            },
            FixedShape {
                m: 64,
                k: 768,
                n: 24,
            },
            FixedShape {
                m: 512,
                k: 768,
                n: 24,
            },
            FixedShape {
                m: 8192,
                k: 768,
                n: 24,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let legacy =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("legacy allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                .expect("candidate allocation");
            let vendor =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("vendor allocation");
            let legacy_ops = FixedFwdOperands {
                c: typed(&legacy, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_ops = FixedFwdOperands {
                c: typed(&candidate, dtype),
                ..legacy_ops
            };
            let vendor_ops = FixedFwdOperands {
                c: typed(&vendor, dtype),
                ..legacy_ops
            };
            let legacy_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, legacy_ops, shape, FixedTile::Legacy)
                    .expect("legacy launch");
            });
            let tc16_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, candidate_ops, shape, FixedTile::Tc16)
                    .expect("Tc16 launch");
            });
            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(true);
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    vendor_ops.c,
                    vendor_ops.x,
                    vendor_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS launch");
            });
            ctx.set_batch_invariant(true);
            ctx.set_fast_gemm(false);
            println!(
                "dtype={dtype:?} m={} k={} n={} legacy_us={legacy_us:.3} tc16_us={tc16_us:.3} cublas_us={cublas_us:.3} tc16_over_legacy={:.5} tc16_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                tc16_us / legacy_us,
                tc16_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits performance data"]
fn fixed_sm120_half_hot_census() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let portable_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc128)
                    .expect("portable launch");
            });
            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(true);
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("cuBLAS launch");
            });
            ctx.set_batch_invariant(true);
            ctx.set_fast_gemm(false);
            for candidate in FixedSm120HalfTile::ALL {
                let elapsed_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Sm120Half(candidate))
                        .unwrap_or_else(|error| panic!("forced {candidate:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={candidate:?} elapsed_us={elapsed_us:.3} portable_us={portable_us:.3} cublas_us={cublas_us:.3} over_portable={:.5} over_cublas={:.5}",
                    shape.m,
                    shape.k,
                    shape.n,
                    elapsed_us / portable_us,
                    elapsed_us / cublas_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits selector data"]
fn fixed_sm120_half_selector_census() {
    let candidates = [
        FixedSm120HalfTile::M64N64Bk64S2,
        FixedSm120HalfTile::M128N64Bk32S3,
        FixedSm120HalfTile::M128N128Bk32S2,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let dtype = WeightDtype::Bf16;
    for m in [16usize, 64, 128] {
        for k in [64usize, 384, 768, 1928] {
            for n in [384usize, 1536] {
                run_fixed_sm120_half_selector_cell(&ctx, dtype, FixedShape { m, k, n }, candidates);
            }
        }
    }
    for m in [256usize, 512, 1024] {
        for k in [384usize, 768, 1928] {
            for n in [384usize, 1536, 2304] {
                run_fixed_sm120_half_selector_cell(&ctx, dtype, FixedShape { m, k, n }, candidates);
            }
        }
    }
    for m in [1536usize, 2048, 3072, 4096, 4621] {
        for k in [384usize, 768, 1928] {
            for n in [1536usize, 1928, 2304] {
                run_fixed_sm120_half_selector_cell(&ctx, dtype, FixedShape { m, k, n }, candidates);
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired selector evidence"]
fn fixed_sm120_half_selector_paired_gaps() {
    use FixedSm120HalfTile::M128N128Bk32S2 as B;
    use FixedSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            FixedShape {
                m: 512,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            FixedShape {
                m: 1536,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 1536,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 1536,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1536,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            FixedShape {
                m: 2048,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 2048,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 2048,
                k: 1928,
                n: 2304,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 4096,
                k: 768,
                n: 2304,
            },
            B,
            A,
        ),
        (
            FixedShape {
                m: 4621,
                k: 1928,
                n: 2304,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1024,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1024,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            FixedShape {
                m: 2048,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 3072,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let incumbent = FixedTile::Sm120Half(incumbent);
            let challenger = FixedTile::Sm120Half(challenger);
            for _ in 0..128 {
                fixed_forward_with_tile(&ctx, operands, shape, incumbent)
                    .expect("incumbent warmup");
                fixed_forward_with_tile(&ctx, operands, shape, challenger)
                    .expect("challenger warmup");
            }
            ctx.stream.synchronize().expect("paired warmup sync");
            let incumbent_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, incumbent);
            let challenger_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, challenger);
            let mut ratios = Vec::with_capacity(101);
            for round in 0..101 {
                let (challenger_us, incumbent_us) = if round & 1 == 0 {
                    (
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            challenger,
                            challenger_iterations,
                        ),
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            incumbent,
                            incumbent_iterations,
                        ),
                    )
                } else {
                    let incumbent_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        incumbent,
                        incumbent_iterations,
                    );
                    let challenger_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        challenger,
                        challenger_iterations,
                    );
                    (challenger_us, incumbent_us)
                };
                ratios.push(challenger_us / incumbent_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} m={} k={} n={} incumbent={incumbent:?} challenger={challenger:?} incumbent_iterations={incumbent_iterations} challenger_iterations={challenger_iterations} ratio_p05={:.6} p50={:.6} p95={:.6}",
                shape.m,
                shape.k,
                shape.n,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and screens every SM120 half tile"]
fn fixed_sm120_half_remaining_gap_all_tile_screen() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 512,
            k: 1928,
            n: 2304,
        },
        FixedShape {
            m: 1024,
            k: 1928,
            n: 1928,
        },
        FixedShape {
            m: 1024,
            k: 1928,
            n: 2304,
        },
        FixedShape {
            m: 1536,
            k: 1032,
            n: 1536,
        },
        FixedShape {
            m: 1536,
            k: 1928,
            n: 1536,
        },
        FixedShape {
            m: 1536,
            k: 1928,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 1928,
            n: 1536,
        },
        FixedShape {
            m: 2048,
            k: 1928,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 1928,
        },
        FixedShape {
            m: 1536,
            k: 384,
            n: 1536,
        },
        FixedShape {
            m: 1536,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 3072,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 1536,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let candidate_filter = std::env::var("MAMBA_FIXED_HALF_TILE_CANDIDATE").ok();
    if let Some(filter) = candidate_filter.as_deref() {
        assert!(
            FixedSm120HalfTile::ALL
                .iter()
                .any(|tile| format!("{tile:?}") == filter),
            "unknown Fixed SM120 half candidate filter: {filter}"
        );
    }

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let incumbent_output =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("incumbent output");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_5010))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_5010))
                .expect("B upload");
            let incumbent_operands = FixedFwdOperands {
                c: typed(&incumbent_output, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let selected = fixed_forward(
                &ctx,
                incumbent_operands.c,
                incumbent_operands.x,
                incumbent_operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production AUTO launch");
            let FixedTile::Sm120Half(incumbent) = selected else {
                panic!("production AUTO selected {selected:?} for {shape:?}");
            };
            ctx.stream.synchronize().expect("incumbent result sync");
            let expected = f32_bits(&ctx, &incumbent_output, shape.m * shape.n);
            let incumbent = FixedTile::Sm120Half(incumbent);

            for candidate in FixedSm120HalfTile::ALL {
                if candidate_filter
                    .as_deref()
                    .is_some_and(|filter| format!("{candidate:?}") != filter)
                {
                    continue;
                }
                let candidate = FixedTile::Sm120Half(candidate);
                if candidate == incumbent {
                    continue;
                }
                let candidate_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                    .expect("candidate output");
                let candidate_operands = FixedFwdOperands {
                    c: typed(&candidate_output, dtype),
                    ..incumbent_operands
                };
                fixed_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                    .expect("candidate bit-gate launch");
                ctx.stream.synchronize().expect("candidate bit-gate sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate_output, shape.m * shape.n),
                    expected,
                    "candidate bits for {dtype:?} {shape:?} {candidate:?}"
                );
                for _ in 0..128 {
                    fixed_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                        .expect("incumbent warmup");
                    fixed_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                        .expect("candidate warmup");
                }
                ctx.stream.synchronize().expect("paired warmup sync");
                let incumbent_iterations =
                    fixed_tile_window_iterations(&ctx, incumbent_operands, shape, incumbent);
                let candidate_iterations =
                    fixed_tile_window_iterations(&ctx, candidate_operands, shape, candidate);
                let mut candidate_first = Vec::with_capacity(101);
                let mut incumbent_first = Vec::with_capacity(101);
                for order in 0..2 {
                    let ratios = if order == 0 {
                        &mut candidate_first
                    } else {
                        &mut incumbent_first
                    };
                    for _ in 0..101 {
                        let (candidate_us, incumbent_us) = if order == 0 {
                            (
                                fixed_tile_window_us(
                                    &ctx,
                                    candidate_operands,
                                    shape,
                                    candidate,
                                    candidate_iterations,
                                ),
                                fixed_tile_window_us(
                                    &ctx,
                                    incumbent_operands,
                                    shape,
                                    incumbent,
                                    incumbent_iterations,
                                ),
                            )
                        } else {
                            let incumbent_us = fixed_tile_window_us(
                                &ctx,
                                incumbent_operands,
                                shape,
                                incumbent,
                                incumbent_iterations,
                            );
                            let candidate_us = fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                candidate,
                                candidate_iterations,
                            );
                            (candidate_us, incumbent_us)
                        };
                        ratios.push(candidate_us / incumbent_us);
                    }
                    ratios.sort_by(f64::total_cmp);
                }
                println!(
                    concat!(
                        "dtype={:?} m={} k={} n={} incumbent={:?} ",
                        "candidate={:?} candidate_first_p50={:.6} ",
                        "candidate_first_p95={:.6} incumbent_first_p50={:.6} ",
                        "incumbent_first_p95={:.6}"
                    ),
                    dtype,
                    shape.m,
                    shape.k,
                    shape.n,
                    incumbent,
                    candidate,
                    percentile(&candidate_first, 0.50),
                    percentile(&candidate_first, 0.95),
                    percentile(&incumbent_first, 0.50),
                    percentile(&incumbent_first, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired selector boundary evidence"]
fn fixed_sm120_half_selector_paired_boundaries() {
    use FixedSm120HalfTile::M128N128Bk32S2 as B;
    use FixedSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            FixedShape {
                m: 1024,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1536,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 2048,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 4621,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1536,
                k: 1032,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 2048,
                k: 1032,
                n: 1536,
            },
            C,
            B,
        ),
        (
            FixedShape {
                m: 1536,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 2048,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 2048,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            for challenger_first in [true, false] {
                run_fixed_sm120_half_selector_boundary_cohort(
                    &ctx,
                    dtype,
                    shape,
                    incumbent,
                    challenger,
                    challenger_first,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired shallow selector boundary evidence"]
fn fixed_sm120_half_selector_paired_shallow_boundaries() {
    use FixedSm120HalfTile::M128N128Bk32S2 as B;
    use FixedSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            FixedShape {
                m: 3072,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4096,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4096,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4096,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4621,
                k: 384,
                n: 1536,
            },
            A,
            C,
        ),
        (
            FixedShape {
                m: 4621,
                k: 768,
                n: 1536,
            },
            A,
            C,
        ),
        (
            FixedShape {
                m: 3072,
                k: 768,
                n: 2304,
            },
            C,
            A,
        ),
        (
            FixedShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            B,
            A,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            for challenger_first in [true, false] {
                run_fixed_sm120_half_selector_boundary_cohort(
                    &ctx,
                    dtype,
                    shape,
                    incumbent,
                    challenger,
                    challenger_first,
                );
            }
        }
    }
}

fn run_fixed_sm120_half_selector_boundary_cohort(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    shape: FixedShape,
    incumbent: FixedSm120HalfTile,
    challenger: FixedSm120HalfTile,
    challenger_first: bool,
) {
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
    let operands = FixedFwdOperands {
        c: typed(&c, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    let incumbent = FixedTile::Sm120Half(incumbent);
    let challenger = FixedTile::Sm120Half(challenger);
    for _ in 0..128 {
        fixed_forward_with_tile(ctx, operands, shape, incumbent).expect("incumbent warmup");
        fixed_forward_with_tile(ctx, operands, shape, challenger).expect("challenger warmup");
    }
    ctx.stream.synchronize().expect("paired warmup sync");
    let incumbent_iterations = fixed_tile_window_iterations(ctx, operands, shape, incumbent);
    let challenger_iterations = fixed_tile_window_iterations(ctx, operands, shape, challenger);
    let mut incumbent_us = Vec::with_capacity(101);
    let mut challenger_us = Vec::with_capacity(101);
    let mut ratios = Vec::with_capacity(101);
    for _ in 0..101 {
        let (challenger_elapsed_us, incumbent_elapsed_us) = if challenger_first {
            (
                fixed_tile_window_us(ctx, operands, shape, challenger, challenger_iterations),
                fixed_tile_window_us(ctx, operands, shape, incumbent, incumbent_iterations),
            )
        } else {
            let incumbent_elapsed_us =
                fixed_tile_window_us(ctx, operands, shape, incumbent, incumbent_iterations);
            let challenger_elapsed_us =
                fixed_tile_window_us(ctx, operands, shape, challenger, challenger_iterations);
            (challenger_elapsed_us, incumbent_elapsed_us)
        };
        ratios.push(challenger_elapsed_us / incumbent_elapsed_us);
        challenger_us.push(challenger_elapsed_us);
        incumbent_us.push(incumbent_elapsed_us);
    }
    challenger_us.sort_by(f64::total_cmp);
    incumbent_us.sort_by(f64::total_cmp);
    ratios.sort_by(f64::total_cmp);
    let order = if challenger_first {
        "challenger_then_incumbent"
    } else {
        "incumbent_then_challenger"
    };
    println!(
        "dtype={dtype:?} m={} k={} n={} incumbent={incumbent:?} challenger={challenger:?} order={order} pairs=101 incumbent_iterations={incumbent_iterations} challenger_iterations={challenger_iterations} challenger_p50_us={:.6} challenger_p95_us={:.6} incumbent_p50_us={:.6} incumbent_p95_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
        shape.m,
        shape.k,
        shape.n,
        percentile(&challenger_us, 0.50),
        percentile(&challenger_us, 0.95),
        percentile(&incumbent_us, 0.50),
        percentile(&incumbent_us, 0.95),
        percentile(&ratios, 0.50),
        percentile(&ratios, 0.95),
    );
}

const FIXED_SM120_HALF_EXACT_JSONL_ENV: &str = "GEMM_BI_FIXED_HALF_EXACT_JSONL";

#[derive(Clone, Copy)]
struct FixedSm120HalfRequalificationCell {
    label: &'static str,
    dtype: WeightDtype,
    shape: FixedShape,
    has_bias: bool,
    candidate: FixedSm120HalfTile,
    incumbent: FixedSm120HalfTile,
}

fn sha256_file(path: &std::path::Path) -> String {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read SHA-256 input {path:?}: {error}"));
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixed_sm120_half_emit_requalification_cohort(
    ctx: &GpuCtx,
    writer: &mut BufWriter<File>,
    cell: FixedSm120HalfRequalificationCell,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    candidate_first: bool,
) -> (f64, f64) {
    let a = DtypedBuf::zeros(&ctx.stream, cell.shape.m * cell.shape.k, cell.dtype)
        .expect("paired A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, cell.shape.k * cell.shape.n, cell.dtype)
        .expect("paired B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, cell.shape.m * cell.shape.n, cell.dtype)
        .expect("paired output allocation");
    a.upload_f32(
        &ctx.stream,
        &synth(
            cell.shape.m * cell.shape.k,
            0xa132_0001 ^ cell.shape.m as u64,
        ),
    )
    .expect("paired A upload");
    b.upload_f32(
        &ctx.stream,
        &synth(
            cell.shape.k * cell.shape.n,
            0xb132_0002 ^ cell.shape.n as u64,
        ),
    )
    .expect("paired B upload");
    let operands = FixedFwdOperands {
        c: typed(&output, cell.dtype),
        x: typed(&a, cell.dtype),
        w: typed(&b, cell.dtype),
        bias_ptr,
    };
    let candidate = FixedTile::Sm120Half(cell.candidate);
    let incumbent = FixedTile::Sm120Half(cell.incumbent);
    for _ in 0..128 {
        fixed_forward_with_tile(ctx, operands, cell.shape, candidate).expect("candidate warmup");
        fixed_forward_with_tile(ctx, operands, cell.shape, incumbent).expect("incumbent warmup");
    }
    ctx.stream
        .synchronize()
        .expect("paired warmup synchronization");
    let candidate_iterations = fixed_tile_window_iterations(ctx, operands, cell.shape, candidate);
    let incumbent_iterations = fixed_tile_window_iterations(ctx, operands, cell.shape, incumbent);
    let order = if candidate_first {
        "candidate_then_incumbent"
    } else {
        "incumbent_then_candidate"
    };
    let bias = if bias_ptr.is_some() {
        "synthesized"
    } else {
        "none"
    };
    fixed_sm120_tf32_bd_environment_preflight(&format!(
        "HALF exact cell={} dtype={} m={} k={} n={} bias={bias} order={order}",
        cell.label,
        cell.dtype.as_str(),
        cell.shape.m,
        cell.shape.k,
        cell.shape.n,
    ))
    .unwrap_or_else(|error| panic!("HALF exact timed-cohort preflight failed: {error}"));
    let mut ratios = Vec::with_capacity(101);
    for pair_index in 0..101 {
        let (candidate_us, incumbent_us) = if candidate_first {
            (
                fixed_tile_window_us(ctx, operands, cell.shape, candidate, candidate_iterations),
                fixed_tile_window_us(ctx, operands, cell.shape, incumbent, incumbent_iterations),
            )
        } else {
            let incumbent_us =
                fixed_tile_window_us(ctx, operands, cell.shape, incumbent, incumbent_iterations);
            let candidate_us =
                fixed_tile_window_us(ctx, operands, cell.shape, candidate, candidate_iterations);
            (candidate_us, incumbent_us)
        };
        let ratio = candidate_us / incumbent_us;
        assert!(ratio.is_finite() && ratio > 0.0, "valid paired ratio");
        ratios.push(ratio);
        writeln!(
            writer,
            "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"window\",\"cell\":\"{}\",\"dtype\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"bias\":\"{bias}\",\"order\":\"{order}\",\"pair_index\":{pair_index},\"candidate_tile\":\"{:?}\",\"incumbent_tile\":\"{:?}\",\"candidate_iterations\":{candidate_iterations},\"incumbent_iterations\":{incumbent_iterations},\"candidate_us\":{candidate_us:.9},\"incumbent_us\":{incumbent_us:.9},\"ratio\":{ratio:.9}}}",
            cell.label,
            cell.dtype.as_str(),
            cell.shape.m,
            cell.shape.k,
            cell.shape.n,
            cell.candidate,
            cell.incumbent,
        )
        .expect("write paired window");
    }
    ratios.sort_by(f64::total_cmp);
    let ratio_p50 = percentile(&ratios, 0.50);
    let ratio_p95 = percentile(&ratios, 0.95);
    let passed = ratio_p50 < 1.0 && ratio_p95 < 1.0;
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"summary\",\"cell\":\"{}\",\"dtype\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"bias\":\"{bias}\",\"order\":\"{order}\",\"windows\":101,\"warmups\":128,\"pilot_iterations\":16,\"target_window_ms\":5.0,\"candidate_tile\":\"{:?}\",\"incumbent_tile\":\"{:?}\",\"candidate_iterations\":{candidate_iterations},\"incumbent_iterations\":{incumbent_iterations},\"ratio_p50\":{ratio_p50:.9},\"ratio_p95\":{ratio_p95:.9},\"passed\":{passed}}}",
        cell.label,
        cell.dtype.as_str(),
        cell.shape.m,
        cell.shape.k,
        cell.shape.n,
        cell.candidate,
        cell.incumbent,
    )
    .expect("write paired summary");
    writer.flush().expect("flush paired summary");
    println!(
        "cell={} dtype={} bias={bias} order={order} ratio_p50={ratio_p50:.9} ratio_p95={ratio_p95:.9} passed={passed}",
        cell.label,
        cell.dtype.as_str(),
    );
    (ratio_p50, ratio_p95)
}

#[test]
#[ignore = "requires a quiet RTX5090 with NVRTC 13.2 and a create-new JSONL sink"]
fn fixed_sm120_half_legacy_overlay_requalification() {
    use FixedSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let path = std::env::var_os(FIXED_SM120_HALF_EXACT_JSONL_ENV)
        .expect("GEMM_BI_FIXED_HALF_EXACT_JSONL must name a new JSONL sink");
    assert!(!path.is_empty(), "JSONL sink path must not be empty");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap_or_else(|error| panic!("create-new HALF exact JSONL sink {path:?}: {error}"));
    let mut writer = BufWriter::new(file);

    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let compiler = ctx.kernels.compiler_identity();
    assert_eq!(compiler.nvrtc_version, (13, 2));
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let rust_source_sha256 = sha256_file(&manifest.join("src/mamba_ssm/gpu/gemm_bi_fixed.rs"));
    let cuda_source_sha256 = sha256_file(&manifest.join("kernels/gemm_bi_fixed/sm120_tma.cu"));
    let executable_sha256 =
        sha256_file(&std::env::current_exe().expect("current qualification executable path"));
    let fixed_module_source_sha256 = digest_hex(&compiler.source_digest);
    let fixed_module_invocation_sha256 = digest_hex(&compiler.invocation_digest);
    let device_sha256 = sha256_bytes(format!("{:?}", device.identity()).as_bytes());
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"metadata\",\"device_cc\":\"12.0\",\"sm_count\":170,\"device_sha256\":\"{device_sha256}\",\"nvrtc_major\":{},\"nvrtc_minor\":{},\"compiler_target\":\"{:?}\",\"fixed_module_source_sha256\":\"{fixed_module_source_sha256}\",\"fixed_module_invocation_sha256\":\"{fixed_module_invocation_sha256}\",\"rust_source_sha256\":\"{rust_source_sha256}\",\"cuda_source_sha256\":\"{cuda_source_sha256}\",\"executable_sha256\":\"{executable_sha256}\",\"tuning_table_revision\":{TUNING_TABLE_REVISION}}}",
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        compiler.target,
    )
    .expect("write paired metadata");

    let cells = [
        FixedSm120HalfRequalificationCell {
            label: "A1_f16_none",
            dtype: WeightDtype::F16,
            shape: FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            has_bias: false,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A1_f16_bias",
            dtype: WeightDtype::F16,
            shape: FixedShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            has_bias: true,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A3_f16_none",
            dtype: WeightDtype::F16,
            shape: FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            has_bias: false,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A3_f16_bias",
            dtype: WeightDtype::F16,
            shape: FixedShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            has_bias: true,
            candidate: A,
            incumbent: C,
        },
    ];
    let bias =
        DtypedBuf::zeros(&ctx.stream, 1928, WeightDtype::F32).expect("paired bias allocation");
    bias.upload_f32(&ctx.stream, &synth(1928, 0xb1a5_0132))
        .expect("paired bias upload");
    let mut all_candidate_contexts_passed = true;
    let mut context_passed = Vec::with_capacity(cells.len());
    for cell in cells {
        let bias_ptr = cell.has_bias.then(|| bias.cached_ptr());
        let mut cell_passed = true;
        for candidate_first in [true, false] {
            let (p50, p95) = fixed_sm120_half_emit_requalification_cohort(
                &ctx,
                &mut writer,
                cell,
                bias_ptr,
                candidate_first,
            );
            cell_passed &= p50 < 1.0 && p95 < 1.0;
        }
        all_candidate_contexts_passed &= cell_passed;
        context_passed.push((cell.label, cell_passed));
    }
    let contexts_json = context_passed
        .iter()
        .map(|(label, passed)| format!(r#"{{"cell":"{label}","passed":{passed}}}"#))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"suite_decision\",\"all_candidate_contexts_passed\":{all_candidate_contexts_passed},\"production_retention_effect\":false,\"contexts\":[{contexts_json}]}}",
    )
    .expect("write paired suite decision");
    writer.flush().expect("flush paired JSONL");
}

fn run_fixed_sm120_half_selector_cell(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    shape: FixedShape,
    candidates: [FixedSm120HalfTile; 3],
) {
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
    let operands = FixedFwdOperands {
        c: typed(&c, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    let mut portable_tile = FixedTile::Legacy;
    let portable_us = average_us(ctx, || {
        portable_tile = fixed_forward(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .expect("portable auto launch");
    });
    for candidate in candidates {
        let elapsed_us = average_us(ctx, || {
            fixed_forward_with_tile(ctx, operands, shape, FixedTile::Sm120Half(candidate))
                .unwrap_or_else(|error| panic!("forced {candidate:?}: {error}"));
        });
        println!(
            "m={} k={} n={} portable={portable_tile:?} portable_us={portable_us:.3} tile={candidate:?} elapsed_us={elapsed_us:.3} over_portable={:.5}",
            shape.m,
            shape.k,
            shape.n,
            elapsed_us / portable_us,
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits mixed-output data"]
fn half_to_f32_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let legacy = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("legacy allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("candidate allocation");
            let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("cuBLAS allocation");
            let legacy_ops = FixedFwdOperands {
                c: typed(&legacy, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_ops = FixedFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..legacy_ops
            };
            let cublas_ops = FixedFwdOperands {
                c: typed(&cublas, WeightDtype::F32),
                ..legacy_ops
            };
            let legacy_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, legacy_ops, shape, FixedTile::Legacy)
                    .expect("legacy mixed-output launch");
            });
            let forced_us = [FixedTile::Tc16, FixedTile::Tc64, FixedTile::Tc128].map(|tile| {
                average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, candidate_ops, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                })
            });
            let mut selected = FixedTile::Legacy;
            let candidate_us = average_us(&ctx, || {
                selected = fixed_forward(
                    &ctx,
                    candidate_ops.c,
                    candidate_ops.x,
                    candidate_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("auto mixed-output launch");
            });
            assert!(matches!(
                selected,
                FixedTile::Tc16 | FixedTile::Tc64 | FixedTile::Tc128
            ));
            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(true);
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    cublas_ops.c,
                    cublas_ops.x,
                    cublas_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS mixed-output launch");
            });
            ctx.set_batch_invariant(true);
            ctx.set_fast_gemm(false);
            println!(
                "dtype={dtype:?} m={} k={} n={} legacy_us={legacy_us:.3} tc16_us={:.3} tc64_us={:.3} tc128_us={:.3} selected={selected:?} auto_us={candidate_us:.3} cublas_us={cublas_us:.3} auto_over_legacy={:.5} auto_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                forced_us[0],
                forced_us[1],
                forced_us[2],
                candidate_us / legacy_us,
                candidate_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits F32 candidate data"]
fn f32_s2_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for shape in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("baseline allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("cuBLAS allocation");
        let base_ops = FixedFwdOperands {
            c: typed(&baseline, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let candidate_ops = FixedFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            ..base_ops
        };
        let cublas_ops = FixedFwdOperands {
            c: typed(&cublas, WeightDtype::F32),
            ..base_ops
        };
        let baseline_us = average_us(&ctx, || {
            fixed_forward_f32_legacy_baseline(&ctx, base_ops, shape).expect("legacy Fixed launch");
        });
        let candidate_us = average_us(&ctx, || {
            fixed_forward(
                &ctx,
                candidate_ops.c,
                candidate_ops.x,
                candidate_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production S2 launch");
        });
        ctx.set_batch_invariant(false);
        ctx.set_fast_gemm(true);
        let cublas_us = average_us(&ctx, || {
            gpu_gemm_typed_forward_raw(
                &ctx,
                cublas_ops.c,
                cublas_ops.x,
                cublas_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("fast cuBLAS launch");
        });
        ctx.set_batch_invariant(true);
        ctx.set_fast_gemm(false);
        println!(
            "m={} k={} n={} legacy_us={baseline_us:.3} s2_us={candidate_us:.3} cublas_tf32_us={cublas_us:.3} s2_over_legacy={:.5} s2_over_cublas={:.5}",
            shape.m,
            shape.k,
            shape.n,
            candidate_us / baseline_us,
            candidate_us / cublas_us,
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn forced_fixed_portable_tiles_execute() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let shape = FixedShape {
        m: 128,
        k: 128,
        n: 128,
    };
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::Bf16).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::Bf16).expect("B allocation");
    let c =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::Bf16).expect("C allocation");
    let operands = FixedFwdOperands {
        c: TypedPtr {
            ptr: c.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        x: TypedPtr {
            ptr: a.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        w: TypedPtr {
            ptr: b.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        bias_ptr: None,
    };
    for tile in [
        FixedTile::Tc16,
        FixedTile::Tc64,
        FixedTile::Tc128,
        FixedTile::TcWn64,
    ] {
        fixed_forward_with_tile(&ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced {tile:?} launch: {error}"));
    }
    ctx.stream.synchronize().expect("forced launch sync");
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits performance data"]
fn fixed_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };

            ctx.set_fast_gemm(false);
            ctx.set_batch_invariant(true);
            ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
            let mut selected = FixedTile::Legacy;
            let fixed_us = average_us(&ctx, || {
                selected = fixed_forward(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("Fixed launch");
            });

            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(true);
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS launch");
            });
            println!(
                "dtype={dtype:?} m={} k={} n={} tile={selected:?} fixed_us={fixed_us:.3} cublas_fast_us={cublas_us:.3} fixed_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                fixed_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits forced-route performance data"]
fn fixed_forced_hot_shapes_smoke() {
    let shapes = [
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let tiles = [
        FixedTile::Tc16,
        FixedTile::Tc64,
        FixedTile::Tc128,
        FixedTile::TcW64,
        FixedTile::TcWn64,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for shape in shapes {
        let dtype = WeightDtype::Bf16;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
        let operands = FixedFwdOperands {
            c: typed(&c, dtype),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        for tile in tiles {
            let elapsed_us = average_us(&ctx, || {
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
            });
            println!(
                "m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits W64 selector data"]
fn fixed_w64_deep_k_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [1024usize, 2048, 4621] {
            for (k, n) in [(1024usize, 384usize), (1536, 768), (1928, 384), (2304, 768)] {
                let shape = FixedShape { m, k, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = FixedFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let square_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc128)
                        .expect("Tc128 launch");
                });
                let reuse_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::TcW64)
                        .expect("TcW64 launch");
                });
                println!(
                    "dtype={dtype:?} m={m} k={k} n={n} tc128_us={square_us:.3} tcw64_us={reuse_us:.3} tcw64_over_tc128={:.5}",
                    reuse_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_portable_resource_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let f32_registers = ctx
        .kernels
        .gemm_bi_f32_f32_s2
        .num_regs()
        .expect("F32 S2 register count");
    let f32_local_bytes = ctx
        .kernels
        .gemm_bi_f32_f32_s2
        .local_size_bytes()
        .expect("F32 S2 local size");
    println!("dtype=F32 tile=S2 registers={f32_registers} local_bytes={f32_local_bytes}");
    assert_eq!(f32_local_bytes, 0, "F32 S2 spills to local memory");
    for (tile, function) in [
        (
            FixedTile::Tf32M128S2,
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s2,
        ),
        (
            FixedTile::Tf32M128S3,
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s3,
        ),
        (FixedTile::Tf32M64S2, &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2),
        (FixedTile::Tf32M64S3, &ctx.kernels.gemm_bi_nn_tf32.m64n64_s3),
        (FixedTile::Tf32M16S4, &ctx.kernels.gemm_bi_nn_tf32.m16n32_s4),
    ] {
        let registers = function.num_regs().expect("TF32 register count");
        let local_bytes = function.local_size_bytes().expect("TF32 local size");
        println!("dtype=TF32 tile={tile:?} registers={registers} local_bytes={local_bytes}");
        assert_eq!(local_bytes, 0, "TF32 {tile:?} spills to local memory");
    }
    let m64 = &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2;
    let m64_occupancy = m64
        .occupancy_max_active_blocks_per_multiprocessor(128, 32_768, None)
        .expect("TF32 M64 occupancy");
    println!("dtype=TF32 tile=Tf32M64S2 occupancy_blocks_per_sm={m64_occupancy}");
    if ctx
        .stream
        .context()
        .compute_capability()
        .expect("CUDA compute capability")
        == (8, 9)
    {
        assert_eq!(m64_occupancy, 3, "SM89 TF32 M64 residency changed");
    }
    if let Some(kernels) = &ctx.kernels.gemm_bi_nn_tf32_sm120 {
        for (tile, function) in [
            (FixedTile::Tf32Sm120M128S2, &kernels.m128n64_s2),
            (FixedTile::Tf32Sm120M128S3, &kernels.m128n64_s3),
            (
                FixedTile::Tf32Sm120M64S2ProducerWarp,
                &kernels.m64n64_s2_producer_warp,
            ),
            (FixedTile::Tf32Sm120M64N128S2, &kernels.m64n128_s2),
            (FixedTile::Tf32Sm120M64N128S3, &kernels.m64n128_s3),
            (FixedTile::Tf32Sm120M64S2, &kernels.m64n64_s2),
        ] {
            let registers = function.num_regs().expect("SM120 TF32 register count");
            let local_bytes = function.local_size_bytes().expect("SM120 TF32 local size");
            println!("dtype=TF32 tile={tile:?} registers={registers} local_bytes={local_bytes}");
            assert_eq!(local_bytes, 0, "SM120 TF32 {tile:?} spills to local memory");
        }
        let m64 = &kernels.m64n64_s2;
        let carveout = m64
            .get_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            )
            .expect("SM120 TF32 M64 carveout");
        let occupancy = m64
            .occupancy_max_active_blocks_per_multiprocessor(128, 32_896, None)
            .expect("SM120 TF32 M64 occupancy");
        println!(
            "dtype=TF32 tile=Tf32Sm120M64S2 preferred_carveout={carveout} occupancy_blocks_per_sm={occupancy}"
        );
        let producer_occupancy = kernels
            .m64n64_s2_producer_warp
            .occupancy_max_active_blocks_per_multiprocessor(160, 32_896, None)
            .expect("SM120 TF32 producer-warp occupancy");
        println!(
            "dtype=TF32 tile=Tf32Sm120M64S2ProducerWarp occupancy_blocks_per_sm={producer_occupancy}"
        );
        assert!(
            producer_occupancy >= 3,
            "SM120 TF32 producer-warp candidate must preserve three resident CTAs per SM"
        );
    }
    for (output, kernels) in [
        ("half", ctx.kernels.gemm_bi_nn_half_sm120.as_ref()),
        ("f32", ctx.kernels.gemm_bi_nn_half_sm120_f32out.as_ref()),
    ] {
        if let Some(kernels) = kernels {
            for (tile, functions) in [
                (FixedSm120HalfTile::M64N64Bk64S2, &kernels.m64n64_bk64_s2),
                (FixedSm120HalfTile::M64N128Bk64S2, &kernels.m64n128_bk64_s2),
                (FixedSm120HalfTile::M128N64Bk32S3, &kernels.m128n64_bk32_s3),
                (
                    FixedSm120HalfTile::M128N128Bk32S2,
                    &kernels.m128n128_bk32_s2,
                ),
                (
                    FixedSm120HalfTile::M128N128Bk32S3,
                    &kernels.m128n128_bk32_s3,
                ),
            ] {
                for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                    let function = functions.get(dtype);
                    let registers = function.num_regs().expect("SM120 half register count");
                    let local_bytes = function.local_size_bytes().expect("SM120 half local size");
                    println!(
                        "dtype={dtype:?} output={output} tile={tile:?} registers={registers} local_bytes={local_bytes}"
                    );
                    assert_eq!(
                        local_bytes, 0,
                        "SM120 half {dtype:?}->{output} {tile:?} spills"
                    );
                }
            }
        }
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (tile, function) in [
            (
                FixedTile::Tc16,
                ctx.kernels.gemm_bi_nn_tc16_typed.get(dtype),
            ),
            (
                FixedTile::Tc64,
                ctx.kernels.gemm_bi_nn_tc64_typed.get(dtype),
            ),
            (
                FixedTile::Tc128,
                ctx.kernels.gemm_bi_nn_tc128_typed.get(dtype),
            ),
            (
                FixedTile::TcW64,
                ctx.kernels.gemm_bi_nn_tcw64_typed.get(dtype),
            ),
            (
                FixedTile::TcWn64,
                ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dtype),
            ),
        ] {
            let registers = function.num_regs().expect("register count");
            let local_bytes = function.local_size_bytes().expect("local size");
            let static_shared_bytes = function.shared_size_bytes().expect("shared size");
            println!(
                "dtype={dtype:?} tile={tile:?} registers={registers} local_bytes={local_bytes} static_shared_bytes={static_shared_bytes}"
            );
            assert_eq!(local_bytes, 0, "{dtype:?} {tile:?} spills to local memory");
            if tile == FixedTile::Tc16 {
                assert_eq!(static_shared_bytes, 24_576);
                if ctx
                    .stream
                    .context()
                    .compute_capability()
                    .expect("CUDA compute capability")
                    == (8, 9)
                {
                    assert_eq!(
                        function
                            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
                            .expect("Tc16 occupancy"),
                        4,
                        "SM89 Tc16 residency changed",
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an RTX5090 with the Fixed module loaded by NVRTC 13.2"]
fn fixed_sm120_half_loaded_resources_are_safe() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    let kernels = ctx
        .kernels
        .gemm_bi_nn_half_sm120
        .as_ref()
        .expect("Fixed SM120 half module");
    for (tile, functions, threads, dynamic_shared_bytes) in [
        (
            FixedSm120HalfTile::M64N64Bk64S2,
            &kernels.m64n64_bk64_s2,
            128,
            32_896,
        ),
        (
            FixedSm120HalfTile::M128N64Bk32S3,
            &kernels.m128n64_bk32_s3,
            256,
            36_992,
        ),
        (
            FixedSm120HalfTile::M64N128Bk64S2,
            &kernels.m64n128_bk64_s2,
            256,
            49_280,
        ),
        (
            FixedSm120HalfTile::M128N128Bk32S2,
            &kernels.m128n128_bk32_s2,
            256,
            32_896,
        ),
        (
            FixedSm120HalfTile::M128N128Bk32S3,
            &kernels.m128n128_bk32_s3,
            256,
            49_280,
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let function = functions.get(dtype);
            let registers = function.num_regs().expect("SM120 half register count");
            let local_bytes = function.local_size_bytes().expect("SM120 half local bytes");
            let static_shared_bytes = function
                .shared_size_bytes()
                .expect("SM120 half static shared bytes");
            let preferred_carveout = function
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                )
                .expect("SM120 half preferred shared-memory carveout");
            let max_dynamic_shared_bytes = function
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                )
                .expect("SM120 half maximum dynamic shared memory");
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, dynamic_shared_bytes, None)
                .expect("SM120 half occupancy");
            println!(
                "{{\"schema\":\"MambaBiFixedSm120HalfResourceV1\",\"dtype\":\"{}\",\"tile\":\"{tile:?}\",\"registers\":{registers},\"local_bytes\":{local_bytes},\"static_shared_bytes\":{static_shared_bytes},\"threads\":{threads},\"dynamic_shared_bytes\":{dynamic_shared_bytes},\"max_dynamic_shared_bytes\":{max_dynamic_shared_bytes},\"preferred_shared_memory_carveout\":{preferred_carveout},\"occupancy_blocks_per_sm\":{occupancy}}}",
                dtype.as_str(),
            );
            assert!(
                registers <= 128,
                "{dtype:?} {tile:?} register safety ceiling"
            );
            assert_eq!(local_bytes, 0, "{dtype:?} {tile:?} local memory");
            assert_eq!(
                static_shared_bytes, 0,
                "{dtype:?} {tile:?} static shared memory"
            );
            assert_eq!(
                preferred_carveout, -1,
                "{dtype:?} {tile:?} must retain the CUDA default shared-memory carveout"
            );
            assert!(
                max_dynamic_shared_bytes
                    >= i32::try_from(dynamic_shared_bytes)
                        .expect("SM120 half dynamic shared memory fits i32"),
                "{dtype:?} {tile:?} max dynamic shared-memory opt-in {max_dynamic_shared_bytes} is below launch requirement {dynamic_shared_bytes}"
            );
            assert!(occupancy >= 2, "{dtype:?} {tile:?} residency");
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector performance data"]
fn fixed_thin_selector_smoke() {
    let shapes = [
        FixedShape {
            m: 65,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 96,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 128,
            k: 1536,
            n: 1536,
        },
        FixedShape {
            m: 128,
            k: 2560,
            n: 1536,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 1928,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 160,
            k: 1536,
            n: 1536,
        },
        FixedShape {
            m: 129,
            k: 1537,
            n: 1535,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for tile in [FixedTile::Tc16, FixedTile::Tc64, FixedTile::Tc128] {
                let elapsed_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    shape.m, shape.k, shape.n,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits dense selector data"]
fn fixed_thin_selector_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [65usize, 80, 96, 112, 128, 129, 160] {
            for n in [512usize, 1024, 1536, 1928, 2304] {
                let shape = FixedShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = FixedFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k=768 n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector boundary data"]
fn fixed_thin_selector_large_m_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [192usize, 256, 320, 384, 512, 768] {
            for n in [512usize, 1024, 1536] {
                let shape = FixedShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = FixedFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k=768 n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector K-axis data"]
fn fixed_thin_selector_k_axis_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let mn_pairs = [
        (256usize, 512usize),
        (320, 512),
        (128, 1024),
        (160, 1024),
        (96, 1536),
        (112, 1536),
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, n) in mn_pairs {
            for k in [64usize, 128, 256, 384, 768, 1536, 2560] {
                let shape = FixedShape { m, k, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = FixedFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k={k} n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM89 CUDA device and emits paired selector evidence"]
fn fixed_thin_selector_sm89_paired_boundary() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9), "SM89 required");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let shapes = [
        FixedShape {
            m: 96,
            k: 64,
            n: 1536,
        },
        FixedShape {
            m: 96,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 96,
            k: 2560,
            n: 1536,
        },
        FixedShape {
            m: 112,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 112,
            k: 1536,
            n: 1536,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 129,
            k: 768,
            n: 1024,
        },
        FixedShape {
            m: 160,
            k: 768,
            n: 1024,
        },
        FixedShape {
            m: 160,
            k: 1536,
            n: 1024,
        },
        FixedShape {
            m: 160,
            k: 2560,
            n: 1024,
        },
        FixedShape {
            m: 320,
            k: 768,
            n: 512,
        },
        FixedShape {
            m: 336,
            k: 768,
            n: 512,
        },
        FixedShape {
            m: 512,
            k: 768,
            n: 512,
        },
        FixedShape {
            m: 576,
            k: 768,
            n: 512,
        },
        FixedShape {
            m: 256,
            k: 768,
            n: 1024,
        },
        FixedShape {
            m: 288,
            k: 768,
            n: 1024,
        },
        FixedShape {
            m: 176,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 192,
            k: 768,
            n: 1536,
        },
        FixedShape {
            m: 96,
            k: 768,
            n: 1928,
        },
        FixedShape {
            m: 112,
            k: 768,
            n: 1928,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 1928,
        },
        FixedShape {
            m: 112,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 128,
            k: 768,
            n: 2304,
        },
    ];

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for _ in 0..64 {
                fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc16).expect("warm Tc16");
                fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc64).expect("warm Tc64");
            }
            ctx.stream.synchronize().expect("selector warmup sync");
            let thin_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, FixedTile::Tc16);
            let square_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, FixedTile::Tc64);
            let mut ratios = Vec::with_capacity(101);
            for window in 0..101 {
                let (thin_us, square_us) = if window % 2 == 0 {
                    (
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            FixedTile::Tc16,
                            thin_iterations,
                        ),
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            FixedTile::Tc64,
                            square_iterations,
                        ),
                    )
                } else {
                    let square_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        FixedTile::Tc64,
                        square_iterations,
                    );
                    let thin_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        FixedTile::Tc16,
                        thin_iterations,
                    );
                    (thin_us, square_us)
                };
                ratios.push(thin_us / square_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                concat!(
                    "{{\"schema\":\"MambaBiFixedThinSelectorPairedV1\",",
                    "\"dtype\":\"{:?}\",\"m\":{},\"k\":{},\"n\":{},",
                    "\"windows\":101,\"thin_iterations\":{},\"square_iterations\":{},",
                    "\"ratio_p05\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
                ),
                dtype,
                shape.m,
                shape.k,
                shape.n,
                thin_iterations,
                square_iterations,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM89 CUDA device and emits Tc16 occupancy evidence"]
fn fixed_tc16_occupancy_cliff_paired_census() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (8, 9),
        "Tc16 occupancy census is qualified on SM89",
    );
    assert_eq!(device.multiprocessor_count(), 142);
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    let shapes = [
        FixedShape {
            m: 64,
            k: 768,
            n: 3392,
        },
        FixedShape {
            m: 64,
            k: 768,
            n: 3424,
        },
        FixedShape {
            m: 64,
            k: 768,
            n: 4544,
        },
        FixedShape {
            m: 64,
            k: 1536,
            n: 3424,
        },
        FixedShape {
            m: 1,
            k: 768,
            n: 32768,
        },
        FixedShape {
            m: 64,
            k: 768,
            n: 4096,
        },
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = FixedFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for _ in 0..10 {
                fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc16)
                    .expect("Tc16 warmup");
                fixed_forward_with_tile(&ctx, operands, shape, FixedTile::Tc64)
                    .expect("Tc64 warmup");
            }
            ctx.stream.synchronize().expect("occupancy warmup sync");
            let iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, FixedTile::Tc16).max(
                    fixed_tile_window_iterations(&ctx, operands, shape, FixedTile::Tc64),
                );
            let mut ratios = Vec::with_capacity(101);
            for round in 0..101 {
                let (tc16_us, tc64_us) = if round & 1 == 0 {
                    let thin =
                        fixed_tile_window_us(&ctx, operands, shape, FixedTile::Tc16, iterations);
                    let square =
                        fixed_tile_window_us(&ctx, operands, shape, FixedTile::Tc64, iterations);
                    (thin, square)
                } else {
                    let square =
                        fixed_tile_window_us(&ctx, operands, shape, FixedTile::Tc64, iterations);
                    let thin =
                        fixed_tile_window_us(&ctx, operands, shape, FixedTile::Tc16, iterations);
                    (thin, square)
                };
                ratios.push(tc16_us / tc64_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} m={} k={} n={} iterations={} tc16_over_tc64_p05={:.6} p50={:.6} p95={:.6}",
                shape.m,
                shape.k,
                shape.n,
                iterations,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FixedSm120Tf32BdComparison {
    id: &'static str,
    cell: &'static str,
    shape: FixedShape,
    dtype: WeightDtype,
    incumbent: FixedTile,
    candidate: FixedTile,
    incumbent_gap_close_ratio: f64,
}

const FIXED_SM120_TF32_BD_PAIRED_COMPARISONS: [FixedSm120Tf32BdComparison; 3] = [
    FixedSm120Tf32BdComparison {
        id: "b_m128_vs_sm120_m64",
        cell: "B",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: FixedTile::Tf32Sm120M64S2,
        candidate: FixedTile::Tf32Sm120M128S2,
        incumbent_gap_close_ratio: 0.936040362,
    },
    FixedSm120Tf32BdComparison {
        id: "d_m128_vs_sm120_m64",
        cell: "D",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: FixedTile::Tf32Sm120M64S2,
        candidate: FixedTile::Tf32Sm120M128S2,
        incumbent_gap_close_ratio: 0.974839395,
    },
    FixedSm120Tf32BdComparison {
        id: "d_portable_m64_vs_sm120_m64",
        cell: "D",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: FixedTile::Tf32Sm120M64S2,
        candidate: FixedTile::Tf32M64S2,
        incumbent_gap_close_ratio: 0.0,
    },
];

const FIXED_SM120_TF32_BD_WARMUPS: usize = 128;
const FIXED_SM120_TF32_BD_PILOT_ITERATIONS: usize = 16;
const FIXED_SM120_TF32_BD_TARGET_WINDOW_MS: f64 = 5.0;
const FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER: usize = 101;
const FIXED_SM120_TF32_BD_ROUTE_ORDERS: [&str; 2] =
    ["candidate_then_incumbent", "incumbent_then_candidate"];
const FIXED_SM120_TF32_BD_BIASES: [&str; 2] = ["none", "synthesized"];
const FIXED_SM120_TF32_BD_VENDOR_ORDERS: [&str; 2] =
    ["candidate_then_vendor", "vendor_then_candidate"];
const FIXED_SM120_TF32_BD_SCHEMA: &str = "MambaBiFixedTf32BdForcedPairedV1";
const FIXED_SM120_TF32_BD_SUITE: &str = "fixed_sm120_tf32_bd_paired_forced_routes";
const FIXED_SM120_TF32_BD_JSONL_ENV: &str = "GEMM_BI_TF32_BD_JSONL";
const FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE: &str = "resource_registers";
const FIXED_SM120_TF32_BD_A_SEED: u64 = 0xa170_7f32_0000_0001;
const FIXED_SM120_TF32_BD_B_SEED: u64 = 0xb170_7f32_0000_0002;
const FIXED_SM120_TF32_BD_BIAS_SEED: u64 = 0xb1a5_7f32_0000_0003;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedSm120Tf32BdRouteSpec {
    tile_label: &'static str,
    symbol: &'static str,
    block_m: usize,
    block_n: usize,
    block_threads: u32,
    dynamic_shared_bytes: usize,
    minimum_occupancy: u32,
}

#[derive(Debug, Clone, Copy)]
struct FixedSm120Tf32BdResourceSnapshot {
    spec: FixedSm120Tf32BdRouteSpec,
    grid: usize,
    registers: i32,
    local_bytes: i32,
    static_shared_bytes: i32,
    occupancy_blocks_per_sm: u32,
    preferred_shared_memory_carveout: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FixedSm120Tf32BdResourceDisposition {
    Pass,
    ExpectedPortableRegisterRejection,
    UnexpectedResourceDefect {
        failed_gate: &'static str,
        reason: String,
    },
}

#[derive(Debug, Default)]
struct FixedSm120Tf32BdRecordCounts {
    preflight: Cell<usize>,
    bit_gate: Cell<usize>,
    window: Cell<usize>,
    summary: Cell<usize>,
    rejection: Cell<usize>,
}

impl FixedSm120Tf32BdRecordCounts {
    fn increment(&self, record_type: &str) {
        let counter = match record_type {
            "preflight" => &self.preflight,
            "bit_gate" => &self.bit_gate,
            "window" => &self.window,
            "summary" => &self.summary,
            "rejection" => &self.rejection,
            _ => panic!("unknown TF32 B/D record type {record_type}"),
        };
        counter.set(counter.get() + 1);
    }

    fn has_no_measurement_records(&self) -> bool {
        self.bit_gate.get() == 0 && self.window.get() == 0 && self.summary.get() == 0
    }
}

struct FixedSm120Tf32BdJsonlSink {
    writer: RefCell<BufWriter<File>>,
}

impl FixedSm120Tf32BdJsonlSink {
    fn from_env() -> Result<Self, String> {
        let path = std::env::var_os(FIXED_SM120_TF32_BD_JSONL_ENV)
            .ok_or_else(|| format!("{FIXED_SM120_TF32_BD_JSONL_ENV} must name the JSONL sink"))?;
        if path.is_empty() {
            return Err(format!("{FIXED_SM120_TF32_BD_JSONL_ENV} must not be empty"));
        }
        let file = File::create(&path)
            .map_err(|error| format!("create TF32 B/D JSONL sink {path:?}: {error}"))?;
        eprintln!("TF32 B/D JSONL sink: {path:?}");
        Ok(Self {
            writer: RefCell::new(BufWriter::new(file)),
        })
    }

    fn emit(&self, record: &str) {
        let mut writer = self.writer.borrow_mut();
        writeln!(writer, "{record}").expect("write TF32 B/D JSONL record");
        writer.flush().expect("flush TF32 B/D JSONL record");
        println!("{record}");
    }
}

struct FixedSm120Tf32BdComparisonEmitter<'a> {
    sink: &'a FixedSm120Tf32BdJsonlSink,
    counts: FixedSm120Tf32BdRecordCounts,
}

impl<'a> FixedSm120Tf32BdComparisonEmitter<'a> {
    fn new(sink: &'a FixedSm120Tf32BdJsonlSink) -> Self {
        Self {
            sink,
            counts: FixedSm120Tf32BdRecordCounts::default(),
        }
    }

    fn emit(&self, record_type: &str, record: String) {
        self.counts.increment(record_type);
        self.sink.emit(&record);
    }
}

#[derive(Debug, Clone)]
struct FixedSm120Tf32BdSummary {
    comparison_kind: &'static str,
    comparator_order: &'static str,
    candidate_p05_us: f64,
    candidate_p50_us: f64,
    candidate_p95_us: f64,
    comparator_p05_us: f64,
    comparator_p50_us: f64,
    comparator_p95_us: f64,
    ratio_p05: f64,
    ratio_p50: f64,
    ratio_p95: f64,
}

#[derive(Clone, Copy)]
struct FixedSm120Tf32BdRecordContext<'a> {
    comparison: FixedSm120Tf32BdComparison,
    bias: &'a str,
    device_cc: (u32, u32),
    sm_count: u32,
    emitter: &'a FixedSm120Tf32BdComparisonEmitter<'a>,
}

fn fixed_sm120_tf32_bd_route_spec(tile: FixedTile) -> FixedSm120Tf32BdRouteSpec {
    match tile {
        FixedTile::Tf32Sm120M64S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32Sm120M64S2",
            symbol: "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2",
            block_m: 64,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 32_896,
            minimum_occupancy: 3,
        },
        FixedTile::Tf32Sm120M128S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32Sm120M128S2",
            symbol: "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2",
            block_m: 128,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 49_280,
            minimum_occupancy: 2,
        },
        FixedTile::Tf32M64S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32M64S2",
            symbol: "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
            block_m: 64,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 32_768,
            minimum_occupancy: 3,
        },
        _ => panic!("{tile:?} is outside the frozen TF32 B/D paired inventory"),
    }
}

fn fixed_sm120_tf32_bd_resource_snapshot(
    ctx: &GpuCtx,
    shape: FixedShape,
    tile: FixedTile,
) -> Result<FixedSm120Tf32BdResourceSnapshot, String> {
    let spec = fixed_sm120_tf32_bd_route_spec(tile);
    let function = match tile {
        FixedTile::Tf32Sm120M64S2 => {
            &ctx.kernels
                .gemm_bi_nn_tf32_sm120
                .as_ref()
                .ok_or("Fixed SM120 TF32 function holder is absent")?
                .m64n64_s2
        }
        FixedTile::Tf32Sm120M128S2 => {
            &ctx.kernels
                .gemm_bi_nn_tf32_sm120
                .as_ref()
                .ok_or("Fixed SM120 TF32 function holder is absent")?
                .m128n64_s2
        }
        FixedTile::Tf32M64S2 => &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2,
        _ => {
            return Err(format!(
                "{tile:?} is outside the TF32 B/D resource inventory"
            ));
        }
    };
    let registers = function
        .num_regs()
        .map_err(|error| format!("read {} registers: {error:?}", spec.symbol))?;
    let local_bytes = function
        .local_size_bytes()
        .map_err(|error| format!("read {} local bytes: {error:?}", spec.symbol))?;
    let static_shared_bytes = function
        .shared_size_bytes()
        .map_err(|error| format!("read {} static shared bytes: {error:?}", spec.symbol))?;
    let occupancy_blocks_per_sm = function
        .occupancy_max_active_blocks_per_multiprocessor(
            spec.block_threads,
            spec.dynamic_shared_bytes,
            None,
        )
        .map_err(|error| format!("read {} occupancy: {error:?}", spec.symbol))?;
    let preferred_shared_memory_carveout = function
        .get_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        )
        .map_err(|error| format!("read {} preferred carveout: {error:?}", spec.symbol))?;
    Ok(FixedSm120Tf32BdResourceSnapshot {
        spec,
        grid: shape.m.div_ceil(spec.block_m) * shape.n.div_ceil(spec.block_n),
        registers,
        local_bytes,
        static_shared_bytes,
        occupancy_blocks_per_sm,
        preferred_shared_memory_carveout,
    })
}

fn fixed_sm120_tf32_bd_resource_disposition(
    comparison: FixedSm120Tf32BdComparison,
    arm: &str,
    tile: FixedTile,
    snapshot: FixedSm120Tf32BdResourceSnapshot,
) -> FixedSm120Tf32BdResourceDisposition {
    let expected = fixed_sm120_tf32_bd_route_spec(tile);
    if snapshot.spec != expected
        || snapshot.grid
            != comparison.shape.m.div_ceil(expected.block_m)
                * comparison.shape.n.div_ceil(expected.block_n)
    {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_mapping",
            reason: format!(
                "{arm} route metadata is incoherent: expected {expected:?}, observed {snapshot:?}"
            ),
        };
    }
    if snapshot.registers <= 0 {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_registers",
            reason: format!(
                "{arm} {} reported invalid register count {}",
                snapshot.spec.symbol, snapshot.registers
            ),
        };
    }
    if snapshot.local_bytes != 0 {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_local_bytes",
            reason: format!(
                "{arm} {} uses {} local bytes, required 0",
                snapshot.spec.symbol, snapshot.local_bytes
            ),
        };
    }
    if snapshot.occupancy_blocks_per_sm < expected.minimum_occupancy {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_occupancy",
            reason: format!(
                "{arm} {} occupancy {} is below required {}",
                snapshot.spec.symbol, snapshot.occupancy_blocks_per_sm, expected.minimum_occupancy
            ),
        };
    }
    if snapshot.registers > 128 {
        if comparison.id == "d_portable_m64_vs_sm120_m64"
            && arm == "candidate"
            && tile == FixedTile::Tf32M64S2
            && snapshot.spec.symbol == "gemm_bi_nn_tf32_v1_m64n64_bk32_s2"
            && snapshot.spec.dynamic_shared_bytes == 32_768
        {
            return FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection;
        }
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_registers",
            reason: format!(
                "{arm} {} uses {} registers, limit is 128",
                snapshot.spec.symbol, snapshot.registers
            ),
        };
    }
    FixedSm120Tf32BdResourceDisposition::Pass
}

fn fixed_sm120_tf32_bd_digest(bits: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for word in bits {
        hasher.update(word.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn fixed_sm120_tf32_bd_json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn fixed_sm120_tf32_bd_common_json(
    comparison: FixedSm120Tf32BdComparison,
    bias: &str,
    device_cc: (u32, u32),
    sm_count: u32,
    incumbent_iterations: usize,
    candidate_iterations: usize,
) -> String {
    let incumbent = fixed_sm120_tf32_bd_route_spec(comparison.incumbent);
    let candidate = fixed_sm120_tf32_bd_route_spec(comparison.candidate);
    format!(
        concat!(
            "\"schema\":\"{}\",\"suite\":\"{}\",",
            "\"comparison_id\":\"{}\",\"cell\":\"{}\",",
            "\"shape\":{{\"m\":{},\"k\":{},\"n\":{}}},",
            "\"device\":{{\"cc\":\"{}.{}\",\"sm_count\":{}}},",
            "\"dtype\":\"f32\",\"f32_policy\":\"allow_deterministic_tf32_v1\",",
            "\"bias\":\"{}\",",
            "\"seed\":{{\"a\":\"0x{:016x}\",\"b\":\"0x{:016x}\",\"bias\":\"0x{:016x}\"}},",
            "\"input_lengths\":{{\"a\":{},\"b\":{},\"bias\":{}}},\"output_len\":{},",
            "\"warmups\":{},\"pilot_iterations\":{},\"target_window_ms\":{:.1},",
            "\"windows_per_order\":{},",
            "\"incumbent\":{{\"tile\":\"{}\",\"symbol\":\"{}\",\"iterations\":{}}},",
            "\"candidate\":{{\"tile\":\"{}\",\"symbol\":\"{}\",\"iterations\":{}}}"
        ),
        FIXED_SM120_TF32_BD_SCHEMA,
        FIXED_SM120_TF32_BD_SUITE,
        comparison.id,
        comparison.cell,
        comparison.shape.m,
        comparison.shape.k,
        comparison.shape.n,
        device_cc.0,
        device_cc.1,
        sm_count,
        bias,
        FIXED_SM120_TF32_BD_A_SEED,
        FIXED_SM120_TF32_BD_B_SEED,
        FIXED_SM120_TF32_BD_BIAS_SEED,
        comparison.shape.m * comparison.shape.k,
        comparison.shape.k * comparison.shape.n,
        comparison.shape.n,
        comparison.shape.m * comparison.shape.n,
        FIXED_SM120_TF32_BD_WARMUPS,
        FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
        FIXED_SM120_TF32_BD_TARGET_WINDOW_MS,
        FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER,
        incumbent.tile_label,
        incumbent.symbol,
        incumbent_iterations,
        candidate.tile_label,
        candidate.symbol,
        candidate_iterations,
    )
}

fn fixed_sm120_tf32_bd_reject(
    comparison: FixedSm120Tf32BdComparison,
    bias: &str,
    device_cc: (u32, u32),
    sm_count: u32,
    failed_gate: &str,
    reason: impl AsRef<str>,
    emitter: &FixedSm120Tf32BdComparisonEmitter<'_>,
) -> ! {
    emitter.emit(
        "rejection",
        format!(
            "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"{}\",\"reason\":\"{}\",\"promotion_eligible\":false}}",
            fixed_sm120_tf32_bd_common_json(comparison, bias, device_cc, sm_count, 0, 0),
            fixed_sm120_tf32_bd_json_escape(failed_gate),
            fixed_sm120_tf32_bd_json_escape(reason.as_ref()),
        ),
    );
    panic!(
        "TF32 B/D comparison {} rejected at {}: {}",
        comparison.id,
        failed_gate,
        reason.as_ref()
    );
}

fn fixed_sm120_tf32_bd_emit_bit_gate(
    record: FixedSm120Tf32BdRecordContext<'_>,
    arm: &str,
    phase: &str,
    replay_index: Option<usize>,
    output_sha256: &str,
    reference_sha256: &str,
) {
    let replay_index = replay_index
        .map(|index| index.to_string())
        .unwrap_or_else(|| "null".to_owned());
    record.emitter.emit(
        "bit_gate",
        format!(
            "{{{},\"record_type\":\"bit_gate\",\"arm\":\"{}\",\"phase\":\"{}\",\"replay_index\":{},\"output_sha256\":\"{}\",\"reference_sha256\":\"{}\",\"exact_bits_equal\":true,\"passed\":true}}",
            fixed_sm120_tf32_bd_common_json(
                record.comparison,
                record.bias,
                record.device_cc,
                record.sm_count,
                0,
                0,
            ),
            arm,
            phase,
            replay_index,
            output_sha256,
            reference_sha256,
        ),
    );
}

fn fixed_sm120_tf32_bd_forced_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_custom(ctx, F32TriadPolicy::AllowDeterministicTf32V1);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record forced Fixed window start");
    for _ in 0..iterations {
        fixed_forward_with_tile(ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced Fixed {tile:?}: {error}"));
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record forced Fixed window end");
    f64::from(start.elapsed_ms(&end).expect("measure forced Fixed window")) * 1000.0
        / iterations as f64
}

fn fixed_sm120_tf32_bd_forced_iterations(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    tile: FixedTile,
) -> usize {
    let pilot_us = fixed_sm120_tf32_bd_forced_window_us(
        ctx,
        operands,
        shape,
        tile,
        FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
    );
    fixed_auto_vendor_iterations(pilot_us)
}

fn fixed_sm120_tf32_bd_environment_preflight(label: &str) -> Result<(), String> {
    let applications = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("run compute-application preflight: {error}"))?;
    if !applications.status.success() {
        return Err(format!(
            "compute-application preflight exited with {}",
            applications.status
        ));
    }
    let own_pid = std::process::id();
    let competing_pids = String::from_utf8_lossy(&applications.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|&pid| pid != own_pid)
        .collect::<Vec<_>>();
    if !competing_pids.is_empty() {
        return Err(format!(
            "competing compute applications: {competing_pids:?}"
        ));
    }

    let mut last_snapshot = None;
    for _ in 0..50 {
        let snapshot = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.sm,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|error| format!("run GPU telemetry preflight: {error}"))?;
        if !snapshot.status.success() {
            return Err(format!(
                "GPU telemetry preflight exited with {}",
                snapshot.status
            ));
        }
        let line = String::from_utf8_lossy(&snapshot.stdout)
            .lines()
            .next()
            .ok_or("GPU telemetry preflight returned no rows")?
            .to_owned();
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(format!(
                "GPU telemetry preflight returned malformed row {line:?}"
            ));
        }
        let gpu_util = fields[0]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU utilization {:?}: {error}", fields[0]))?;
        let memory_util = fields[1]
            .parse::<u32>()
            .map_err(|error| format!("parse memory utilization {:?}: {error}", fields[1]))?;
        let used_mib = fields[2]
            .parse::<u32>()
            .map_err(|error| format!("parse used memory {:?}: {error}", fields[2]))?;
        let sm_clock_mhz = fields[3]
            .parse::<u32>()
            .map_err(|error| format!("parse SM clock {:?}: {error}", fields[3]))?;
        let temperature_c = fields[4]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU temperature {:?}: {error}", fields[4]))?;
        let pstate = fields[5];
        last_snapshot = Some(format!(
            "gpu_util={gpu_util}% memory_util={memory_util}% used={used_mib}MiB sm_clock={sm_clock_mhz}MHz temperature={temperature_c}C pstate={pstate}"
        ));
        if gpu_util <= 1 && memory_util <= 1 && sm_clock_mhz > 0 && temperature_c > 0 {
            eprintln!(
                "TF32 B/D cohort preflight {label}: {} (the process's own allocations are excluded from the launch-time <=128 MiB gate)",
                last_snapshot.as_deref().unwrap_or("telemetry unavailable")
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "GPU did not return to <=1% compute and memory utilization; last snapshot: {}",
        last_snapshot.as_deref().unwrap_or("none")
    ))
}

fn fixed_sm120_tf32_bd_summary(
    comparison_kind: &'static str,
    comparator_order: &'static str,
    candidate_us: &[f64],
    comparator_us: &[f64],
    ratios: &[f64],
) -> FixedSm120Tf32BdSummary {
    let mut candidate_sorted = candidate_us.to_vec();
    let mut comparator_sorted = comparator_us.to_vec();
    let mut ratio_sorted = ratios.to_vec();
    candidate_sorted.sort_by(f64::total_cmp);
    comparator_sorted.sort_by(f64::total_cmp);
    ratio_sorted.sort_by(f64::total_cmp);
    FixedSm120Tf32BdSummary {
        comparison_kind,
        comparator_order,
        candidate_p05_us: percentile(&candidate_sorted, 0.05),
        candidate_p50_us: percentile(&candidate_sorted, 0.50),
        candidate_p95_us: percentile(&candidate_sorted, 0.95),
        comparator_p05_us: percentile(&comparator_sorted, 0.05),
        comparator_p50_us: percentile(&comparator_sorted, 0.50),
        comparator_p95_us: percentile(&comparator_sorted, 0.95),
        ratio_p05: percentile(&ratio_sorted, 0.05),
        ratio_p50: percentile(&ratio_sorted, 0.50),
        ratio_p95: percentile(&ratio_sorted, 0.95),
    }
}

#[test]
fn fixed_sm120_tf32_bd_paired_inventory_is_frozen() {
    assert_eq!(FIXED_SM120_TF32_BD_PAIRED_COMPARISONS.len(), 3);
    assert_eq!(
        FIXED_SM120_TF32_BD_PAIRED_COMPARISONS,
        [
            FixedSm120Tf32BdComparison {
                id: "b_m128_vs_sm120_m64",
                cell: "B",
                shape: FixedShape {
                    m: 4621,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: FixedTile::Tf32Sm120M64S2,
                candidate: FixedTile::Tf32Sm120M128S2,
                incumbent_gap_close_ratio: 0.936040362,
            },
            FixedSm120Tf32BdComparison {
                id: "d_m128_vs_sm120_m64",
                cell: "D",
                shape: FixedShape {
                    m: 2048,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: FixedTile::Tf32Sm120M64S2,
                candidate: FixedTile::Tf32Sm120M128S2,
                incumbent_gap_close_ratio: 0.974839395,
            },
            FixedSm120Tf32BdComparison {
                id: "d_portable_m64_vs_sm120_m64",
                cell: "D",
                shape: FixedShape {
                    m: 2048,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: FixedTile::Tf32Sm120M64S2,
                candidate: FixedTile::Tf32M64S2,
                incumbent_gap_close_ratio: 0.0,
            },
        ]
    );
    assert_eq!(FIXED_SM120_TF32_BD_WARMUPS, 128);
    assert_eq!(FIXED_SM120_TF32_BD_PILOT_ITERATIONS, 16);
    assert_eq!(FIXED_SM120_TF32_BD_TARGET_WINDOW_MS, 5.0);
    assert_eq!(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER, 101);
    assert_eq!(
        FIXED_SM120_TF32_BD_ROUTE_ORDERS,
        ["candidate_then_incumbent", "incumbent_then_candidate"]
    );
    assert_eq!(FIXED_SM120_TF32_BD_BIASES, ["none", "synthesized"]);
    assert_eq!(
        FIXED_SM120_TF32_BD_VENDOR_ORDERS,
        ["candidate_then_vendor", "vendor_then_candidate"]
    );
    let mut ids = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS
        .iter()
        .map(|comparison| comparison.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 3, "comparison ids must be unique");
}

#[test]
fn fixed_sm120_tf32_bd_expected_portable_rejection_is_terminal() {
    let comparison = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS[2];
    let sealed_snapshot = FixedSm120Tf32BdResourceSnapshot {
        spec: fixed_sm120_tf32_bd_route_spec(FixedTile::Tf32M64S2),
        grid: 1152,
        registers: 167,
        local_bytes: 0,
        static_shared_bytes: 0,
        occupancy_blocks_per_sm: 3,
        preferred_shared_memory_carveout: -1,
    };
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            FixedTile::Tf32M64S2,
            sealed_snapshot,
        ),
        FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection
    );
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            FixedTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                registers: 129,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection,
        "the durable rejection law is registers > 128, not exactly 167"
    );
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            FixedTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                registers: 128,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::Pass
    );
    assert!(matches!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            FixedTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                local_bytes: 4,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect { .. }
    ));
    let counts = FixedSm120Tf32BdRecordCounts::default();
    assert!(counts.has_no_measurement_records());
    assert_eq!(
        FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE,
        "resource_registers"
    );
}

#[test]
#[ignore = "requires a quiet 170-SM CC12.0 CUDA device and emits raw paired TF32 B/D evidence"]
fn fixed_sm120_tf32_bd_paired_forced_routes() {
    let sink = FixedSm120Tf32BdJsonlSink::from_env().unwrap_or_else(|error| panic!("{error}"));
    let device = GpuDevice::new(0).expect("CUDA device");
    let device_cc = device.compute_capability;
    let sm_count = device.multiprocessor_count();
    let first = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS[0];
    let first_emitter = FixedSm120Tf32BdComparisonEmitter::new(&sink);
    if device_cc != (12, 0) || sm_count != 170 {
        fixed_sm120_tf32_bd_reject(
            first,
            "none",
            device_cc,
            sm_count,
            "device_identity",
            format!(
                "requires physical CC12.0 with 170 SMs, found CC{}.{} with {} SMs",
                device_cc.0, device_cc.1, sm_count
            ),
            &first_emitter,
        );
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
    assert!(ctx.tf32(), "fast cuBLAS TF32 must remain enabled");

    for comparison in FIXED_SM120_TF32_BD_PAIRED_COMPARISONS {
        let emitter = FixedSm120Tf32BdComparisonEmitter::new(&sink);
        let shape = comparison.shape;
        let output_len = shape.m * shape.n;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, comparison.dtype)
            .expect("TF32 B/D A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, comparison.dtype)
            .expect("TF32 B/D B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, shape.n, comparison.dtype)
            .expect("TF32 B/D bias allocation");
        let incumbent_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D incumbent output allocation");
        let candidate_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D candidate output allocation");
        let vendor_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D vendor output allocation");
        a.upload_f32(
            &ctx.stream,
            &synth(shape.m * shape.k, FIXED_SM120_TF32_BD_A_SEED),
        )
        .expect("TF32 B/D A upload");
        b.upload_f32(
            &ctx.stream,
            &synth(shape.k * shape.n, FIXED_SM120_TF32_BD_B_SEED),
        )
        .expect("TF32 B/D B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, FIXED_SM120_TF32_BD_BIAS_SEED))
            .expect("TF32 B/D bias upload");
        let incumbent_operands = FixedFwdOperands {
            c: typed(&incumbent_output, comparison.dtype),
            x: typed(&a, comparison.dtype),
            w: typed(&b, comparison.dtype),
            bias_ptr: None,
        };
        let candidate_operands = FixedFwdOperands {
            c: typed(&candidate_output, comparison.dtype),
            ..incumbent_operands
        };
        let vendor_operands = FixedFwdOperands {
            c: typed(&vendor_output, comparison.dtype),
            ..incumbent_operands
        };

        let mut expected_portable_rejection = false;
        for (arm, tile, operands) in [
            ("incumbent", comparison.incumbent, incumbent_operands),
            ("candidate", comparison.candidate, candidate_operands),
        ] {
            configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
            if let Err(error) = fixed_forward_with_tile(&ctx, operands, shape, tile) {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "direct_exec_preflight",
                    format!("{arm} direct Fixed enqueue failed: {error}"),
                    &emitter,
                );
            }
            if let Err(error) = ctx.stream.synchronize() {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "direct_exec_preflight",
                    format!("{arm} direct Fixed synchronization failed: {error:?}"),
                    &emitter,
                );
            }
            let resource =
                fixed_sm120_tf32_bd_resource_snapshot(&ctx, shape, tile).unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "resource_snapshot",
                        error,
                        &emitter,
                    )
                });
            match fixed_sm120_tf32_bd_resource_disposition(comparison, arm, tile, resource) {
                FixedSm120Tf32BdResourceDisposition::Pass => emitter.emit(
                    "preflight",
                    format!(
                        "{{{},\"record_type\":\"preflight\",\"arm\":\"{}\",\"execution\":\"direct_fixed_forward_with_tile\",\"preflight_phase\":\"before_warmup\",\"symbol\":\"{}\",\"grid\":{},\"block_threads\":{},\"dynamic_shared_bytes\":{},\"static_shared_bytes\":{},\"registers\":{},\"local_bytes\":{},\"occupancy_blocks_per_sm\":{},\"preferred_shared_memory_carveout\":{},\"passed\":true}}",
                        fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, 0, 0),
                        arm,
                        resource.spec.symbol,
                        resource.grid,
                        resource.spec.block_threads,
                        resource.spec.dynamic_shared_bytes,
                        resource.static_shared_bytes,
                        resource.registers,
                        resource.local_bytes,
                        resource.occupancy_blocks_per_sm,
                        resource.preferred_shared_memory_carveout,
                    ),
                ),
                FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection => {
                    emitter.emit(
                        "rejection",
                        format!(
                            "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"{}\",\"reason\":\"candidate {} uses {} registers, limit is 128\",\"expected_rejection\":true,\"promotion_eligible\":false}}",
                            fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, 0, 0),
                            FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE,
                            resource.spec.symbol,
                            resource.registers,
                        ),
                    );
                    expected_portable_rejection = true;
                    break;
                }
                FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
                    failed_gate,
                    reason,
                } => fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    failed_gate,
                    reason,
                    &emitter,
                ),
            }
        }
        if expected_portable_rejection {
            assert_eq!(comparison.id, "d_portable_m64_vs_sm120_m64");
            assert!(
                emitter.counts.has_no_measurement_records(),
                "expected portable rejection must not emit bit, window, or summary records"
            );
            assert_eq!(
                emitter.counts.rejection.get(),
                1,
                "expected portable rejection must emit exactly one rejection"
            );
            continue;
        }

        for bias_name in FIXED_SM120_TF32_BD_BIASES {
            let bias_ptr = match bias_name {
                "none" => None,
                "synthesized" => Some(bias.cached_ptr()),
                _ => unreachable!("frozen bias inventory"),
            };
            let record = FixedSm120Tf32BdRecordContext {
                comparison,
                bias: bias_name,
                device_cc,
                sm_count,
                emitter: &emitter,
            };
            let incumbent_with_bias = FixedFwdOperands {
                bias_ptr,
                ..incumbent_operands
            };
            let candidate_with_bias = FixedFwdOperands {
                bias_ptr,
                ..candidate_operands
            };
            configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
            fixed_forward_with_tile(&ctx, incumbent_with_bias, shape, comparison.incumbent)
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        format!("incumbent eager enqueue failed: {error}"),
                        &emitter,
                    )
                });
            ctx.stream.synchronize().unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    bias_name,
                    device_cc,
                    sm_count,
                    "eager_bit_gate",
                    format!("incumbent eager synchronization failed: {error:?}"),
                    &emitter,
                )
            });
            let incumbent_bits = f32_bits(&ctx, &incumbent_output, output_len);
            let incumbent_digest = fixed_sm120_tf32_bd_digest(&incumbent_bits);
            fixed_sm120_tf32_bd_emit_bit_gate(
                record,
                "incumbent",
                "eager",
                None,
                &incumbent_digest,
                &incumbent_digest,
            );

            let mut candidate_bits = Vec::new();
            for _ in 0..2 {
                fixed_forward_with_tile(&ctx, candidate_with_bias, shape, comparison.candidate)
                    .unwrap_or_else(|error| {
                        fixed_sm120_tf32_bd_reject(
                            comparison,
                            bias_name,
                            device_cc,
                            sm_count,
                            "eager_bit_gate",
                            format!("candidate eager enqueue failed: {error}"),
                            &emitter,
                        )
                    });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        format!("candidate eager synchronization failed: {error:?}"),
                        &emitter,
                    )
                });
                candidate_bits = f32_bits(&ctx, &candidate_output, output_len);
                if candidate_bits != incumbent_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        "candidate bits differ from incumbent bits",
                        &emitter,
                    );
                }
                let candidate_digest = fixed_sm120_tf32_bd_digest(&candidate_bits);
                if candidate_digest != incumbent_digest {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_digest_gate",
                        "candidate digest differs from incumbent digest",
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "candidate",
                    "eager",
                    None,
                    &candidate_digest,
                    &incumbent_digest,
                );
            }

            let candidate_run =
                || fixed_forward_with_tile(&ctx, candidate_with_bias, shape, comparison.candidate);
            let candidate_graph = unsafe { capture_into_graph(&ctx.stream, candidate_run) }
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_capture_gate",
                        format!("candidate graph capture failed: {error}"),
                        &emitter,
                    )
                });
            for replay_index in 0..10 {
                candidate_graph.launch().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} launch failed: {error:?}"),
                        &emitter,
                    )
                });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} sync failed: {error:?}"),
                        &emitter,
                    )
                });
                let replay_bits = f32_bits(&ctx, &candidate_output, output_len);
                if replay_bits != candidate_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} changed output bits"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "candidate",
                    "graph_replay",
                    Some(replay_index),
                    &fixed_sm120_tf32_bd_digest(&replay_bits),
                    &fixed_sm120_tf32_bd_digest(&candidate_bits),
                );
            }

            let incumbent_run =
                || fixed_forward_with_tile(&ctx, incumbent_with_bias, shape, comparison.incumbent);
            let incumbent_graph = unsafe { capture_into_graph(&ctx.stream, incumbent_run) }
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_capture_gate",
                        format!("incumbent graph capture failed: {error}"),
                        &emitter,
                    )
                });
            for replay_index in 0..10 {
                incumbent_graph.launch().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} launch failed: {error:?}"),
                        &emitter,
                    )
                });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} sync failed: {error:?}"),
                        &emitter,
                    )
                });
                let replay_bits = f32_bits(&ctx, &incumbent_output, output_len);
                if replay_bits != incumbent_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} changed output bits"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "incumbent",
                    "graph_replay",
                    Some(replay_index),
                    &fixed_sm120_tf32_bd_digest(&replay_bits),
                    &incumbent_digest,
                );
            }

            let prefix_shape = FixedShape {
                m: 1,
                k: shape.k,
                n: shape.n,
            };
            for (arm, tile, full_bits) in [
                ("incumbent", comparison.incumbent, &incumbent_bits),
                ("candidate", comparison.candidate, &candidate_bits),
            ] {
                let prefix_output = DtypedBuf::zeros(&ctx.stream, shape.n, comparison.dtype)
                    .expect("TF32 B/D prefix output allocation");
                let prefix_operands = FixedFwdOperands {
                    c: typed(&prefix_output, comparison.dtype),
                    x: typed(&a, comparison.dtype),
                    w: typed(&b, comparison.dtype),
                    bias_ptr,
                };
                fixed_forward_with_tile(&ctx, prefix_operands, prefix_shape, tile).unwrap_or_else(
                    |error| {
                        fixed_sm120_tf32_bd_reject(
                            comparison,
                            bias_name,
                            device_cc,
                            sm_count,
                            "prefix_gate",
                            format!("{arm} prefix enqueue failed: {error}"),
                            &emitter,
                        )
                    },
                );
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "prefix_gate",
                        format!("{arm} prefix synchronization failed: {error:?}"),
                        &emitter,
                    )
                });
                let prefix_bits = f32_bits(&ctx, &prefix_output, shape.n);
                let reference_row = &full_bits[..shape.n];
                if prefix_bits != reference_row {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "prefix_gate",
                        format!("{arm} leading m=1 output differs from full row zero"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    arm,
                    "prefix",
                    None,
                    &fixed_sm120_tf32_bd_digest(&prefix_bits),
                    &fixed_sm120_tf32_bd_digest(reference_row),
                );
            }
        }

        configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            fixed_forward_with_tile(&ctx, incumbent_operands, shape, comparison.incumbent)
                .expect("TF32 B/D incumbent warmup");
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D incumbent warmup sync");
        configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            fixed_forward_with_tile(&ctx, candidate_operands, shape, comparison.candidate)
                .expect("TF32 B/D candidate warmup");
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D candidate warmup sync");
        configure_fixed_auto_vendor_vendor(&ctx, F32TriadPolicy::AllowDeterministicTf32V1);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            launch_fixed_auto_vendor_vendor(&ctx, vendor_operands, shape);
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D vendor warmup sync");

        let incumbent_iterations = fixed_sm120_tf32_bd_forced_iterations(
            &ctx,
            incumbent_operands,
            shape,
            comparison.incumbent,
        );
        let candidate_iterations = fixed_sm120_tf32_bd_forced_iterations(
            &ctx,
            candidate_operands,
            shape,
            comparison.candidate,
        );
        let vendor_pilot_us = fixed_auto_vendor_vendor_window_us(
            &ctx,
            vendor_operands,
            shape,
            F32TriadPolicy::AllowDeterministicTf32V1,
            FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
        );
        let vendor_iterations = fixed_auto_vendor_iterations(vendor_pilot_us);

        let mut window_records = Vec::with_capacity(4 * FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
        let mut summaries = Vec::with_capacity(4);
        for comparator_order in FIXED_SM120_TF32_BD_ROUTE_ORDERS {
            fixed_sm120_tf32_bd_environment_preflight(&format!(
                "{} {comparator_order}",
                comparison.id
            ))
            .unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "environment_preflight",
                    error,
                    &emitter,
                )
            });
            let candidate_first = comparator_order == "candidate_then_incumbent";
            let mut candidate_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut incumbent_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut ratios = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            for pair_index in 0..FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER {
                let (candidate_us, incumbent_us) = if candidate_first {
                    (
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            comparison.candidate,
                            candidate_iterations,
                        ),
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            incumbent_operands,
                            shape,
                            comparison.incumbent,
                            incumbent_iterations,
                        ),
                    )
                } else {
                    let incumbent_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        incumbent_operands,
                        shape,
                        comparison.incumbent,
                        incumbent_iterations,
                    );
                    let candidate_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        candidate_operands,
                        shape,
                        comparison.candidate,
                        candidate_iterations,
                    );
                    (candidate_us, incumbent_us)
                };
                if !candidate_us.is_finite()
                    || candidate_us <= 0.0
                    || !incumbent_us.is_finite()
                    || incumbent_us <= 0.0
                {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "timing_value",
                        format!(
                            "non-positive or non-finite route timing candidate={candidate_us} incumbent={incumbent_us}"
                        ),
                        &emitter,
                    );
                }
                let ratio = candidate_us / incumbent_us;
                candidate_times.push(candidate_us);
                incumbent_times.push(incumbent_us);
                ratios.push(ratio);
                window_records.push(format!(
                    "{{{},\"record_type\":\"window\",\"comparison_kind\":\"candidate_vs_incumbent\",\"comparator_order\":\"{}\",\"pair_index\":{},\"first_arm\":\"{}\",\"second_arm\":\"{}\",\"candidate_us\":{:.9},\"comparator_us\":{:.9},\"ratio\":{:.9},\"candidate_iterations\":{},\"comparator_iterations\":{}}}",
                    fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, incumbent_iterations, candidate_iterations),
                    comparator_order,
                    pair_index,
                    if candidate_first { "candidate" } else { "incumbent" },
                    if candidate_first { "incumbent" } else { "candidate" },
                    candidate_us,
                    incumbent_us,
                    ratio,
                    candidate_iterations,
                    incumbent_iterations,
                ));
            }
            summaries.push(fixed_sm120_tf32_bd_summary(
                "candidate_vs_incumbent",
                comparator_order,
                &candidate_times,
                &incumbent_times,
                &ratios,
            ));
        }

        for comparator_order in FIXED_SM120_TF32_BD_VENDOR_ORDERS {
            fixed_sm120_tf32_bd_environment_preflight(&format!(
                "{} {comparator_order}",
                comparison.id
            ))
            .unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "environment_preflight",
                    error,
                    &emitter,
                )
            });
            let candidate_first = comparator_order == "candidate_then_vendor";
            let mut candidate_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut vendor_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut ratios = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            for pair_index in 0..FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER {
                let (candidate_us, vendor_us) = if candidate_first {
                    (
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            comparison.candidate,
                            candidate_iterations,
                        ),
                        fixed_auto_vendor_vendor_window_us(
                            &ctx,
                            vendor_operands,
                            shape,
                            F32TriadPolicy::AllowDeterministicTf32V1,
                            vendor_iterations,
                        ),
                    )
                } else {
                    let vendor_us = fixed_auto_vendor_vendor_window_us(
                        &ctx,
                        vendor_operands,
                        shape,
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        vendor_iterations,
                    );
                    let candidate_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        candidate_operands,
                        shape,
                        comparison.candidate,
                        candidate_iterations,
                    );
                    (candidate_us, vendor_us)
                };
                if !candidate_us.is_finite()
                    || candidate_us <= 0.0
                    || !vendor_us.is_finite()
                    || vendor_us <= 0.0
                {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "timing_value",
                        format!(
                            "non-positive or non-finite vendor timing candidate={candidate_us} vendor={vendor_us}"
                        ),
                        &emitter,
                    );
                }
                let ratio = candidate_us / vendor_us;
                candidate_times.push(candidate_us);
                vendor_times.push(vendor_us);
                ratios.push(ratio);
                window_records.push(format!(
                    "{{{},\"record_type\":\"window\",\"comparison_kind\":\"candidate_vs_vendor\",\"comparator_order\":\"{}\",\"pair_index\":{},\"first_arm\":\"{}\",\"second_arm\":\"{}\",\"candidate_us\":{:.9},\"comparator_us\":{:.9},\"ratio\":{:.9},\"candidate_iterations\":{},\"comparator_iterations\":{},\"comparator_tile\":\"fast_cublas_tf32_allowed\"}}",
                    fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, incumbent_iterations, candidate_iterations),
                    comparator_order,
                    pair_index,
                    if candidate_first { "candidate" } else { "vendor" },
                    if candidate_first { "vendor" } else { "candidate" },
                    candidate_us,
                    vendor_us,
                    ratio,
                    candidate_iterations,
                    vendor_iterations,
                ));
            }
            summaries.push(fixed_sm120_tf32_bd_summary(
                "candidate_vs_vendor",
                comparator_order,
                &candidate_times,
                &vendor_times,
                &ratios,
            ));
        }

        let route_summaries = &summaries[..2];
        let vendor_summaries = &summaries[2..];
        let route_order_gate = route_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 0.995 && summary.ratio_p95 < 1.0);
        let vendor_order_gate = vendor_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 1.0 && summary.ratio_p95 < 1.0);
        let gap_close_gate = route_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= comparison.incumbent_gap_close_ratio);
        let is_m128_candidate = comparison.candidate == FixedTile::Tf32Sm120M128S2;
        let promotion_eligible =
            is_m128_candidate && route_order_gate && vendor_order_gate && gap_close_gate;
        let vendor_verdict = if vendor_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 0.995 && summary.ratio_p95 < 1.0)
        {
            "clear_vendor_win"
        } else if vendor_order_gate {
            "parity_admissible"
        } else {
            "rejected"
        };

        for record in window_records {
            emitter.emit("window", record);
        }
        for summary in &summaries {
            let comparator_tile = if summary.comparison_kind == "candidate_vs_vendor" {
                ",\"comparator_tile\":\"fast_cublas_tf32_allowed\""
            } else {
                ""
            };
            let verdict = if summary.comparison_kind == "candidate_vs_vendor" {
                format!(",\"vendor_verdict\":\"{vendor_verdict}\"")
            } else {
                String::new()
            };
            emitter.emit(
                "summary",
                format!(
                    "{{{},\"record_type\":\"summary\",\"comparison_kind\":\"{}\",\"comparator_order\":\"{}\",\"candidate_p05_us\":{:.9},\"candidate_p50_us\":{:.9},\"candidate_p95_us\":{:.9},\"comparator_p05_us\":{:.9},\"comparator_p50_us\":{:.9},\"comparator_p95_us\":{:.9},\"ratio_p05\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9},\"direct_preflight_passed\":true,\"resource_gate_passed\":true,\"exact_bits_passed\":true,\"graph_replay_passed\":true,\"prefix_gate_passed\":true,\"environment_preflight_passed\":true,\"promotion_eligible\":{}{}{}}}",
                    fixed_sm120_tf32_bd_common_json(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        incumbent_iterations,
                        candidate_iterations
                    ),
                    summary.comparison_kind,
                    summary.comparator_order,
                    summary.candidate_p05_us,
                    summary.candidate_p50_us,
                    summary.candidate_p95_us,
                    summary.comparator_p05_us,
                    summary.comparator_p50_us,
                    summary.comparator_p95_us,
                    summary.ratio_p05,
                    summary.ratio_p50,
                    summary.ratio_p95,
                    promotion_eligible,
                    comparator_tile,
                    verdict,
                ),
            );
        }
        if is_m128_candidate && !promotion_eligible {
            emitter.emit(
                "rejection",
                format!(
                    "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"strict_decision_rule\",\"reason\":\"one or more paired order or vendor-gap thresholds failed\",\"route_order_gate\":{},\"vendor_order_gate\":{},\"gap_close_gate\":{},\"promotion_eligible\":false}}",
                    fixed_sm120_tf32_bd_common_json(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        incumbent_iterations,
                        candidate_iterations
                    ),
                    route_order_gate,
                    vendor_order_gate,
                    gap_close_gate,
                ),
            );
        }
    }
}

#[derive(Clone, Copy)]
struct FixedAutoVendorCell {
    label: &'static str,
    shape: FixedShape,
    expected: FixedTile,
}

#[derive(Clone, Copy)]
struct FixedAutoVendorRow {
    name: &'static str,
    input_dtype: WeightDtype,
    output_dtype: WeightDtype,
    policy: F32TriadPolicy,
    policy_name: &'static str,
    cells: &'static [FixedAutoVendorCell],
}

const FIXED_AUTO_VENDOR_HALF_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m512_k1928_n2304",
        shape: FixedShape {
            m: 512,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n1536",
        shape: FixedShape {
            m: 1024,
            k: 1928,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n1928",
        shape: FixedShape {
            m: 1024,
            k: 1928,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n2304",
        shape: FixedShape {
            m: 1024,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n1536",
        shape: FixedShape {
            m: 1536,
            k: 1928,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n1928",
        shape: FixedShape {
            m: 1536,
            k: 1928,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n2304",
        shape: FixedShape {
            m: 1536,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n1536",
        shape: FixedShape {
            m: 2048,
            k: 1928,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n1928",
        shape: FixedShape {
            m: 2048,
            k: 1928,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n2304",
        shape: FixedShape {
            m: 2048,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n1536",
        shape: FixedShape {
            m: 3072,
            k: 1928,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n1928",
        shape: FixedShape {
            m: 3072,
            k: 1928,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n2304",
        shape: FixedShape {
            m: 3072,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m4621_k1928_n1928",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m4621_k1928_n2304",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1032_n1536",
        shape: FixedShape {
            m: 1536,
            k: 1032,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1032_n1536",
        shape: FixedShape {
            m: 2048,
            k: 1032,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k384_n1536",
        shape: FixedShape {
            m: 1536,
            k: 384,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k768_n1536",
        shape: FixedShape {
            m: 1536,
            k: 768,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k384_n1928",
        shape: FixedShape {
            m: 1536,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k384_n1928",
        shape: FixedShape {
            m: 2048,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k768_n1928",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k384_n1928",
        shape: FixedShape {
            m: 3072,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n1536",
        shape: FixedShape {
            m: 4096,
            k: 768,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k384_n1928",
        shape: FixedShape {
            m: 4096,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n1928",
        shape: FixedShape {
            m: 4096,
            k: 768,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n2304",
        shape: FixedShape {
            m: 4096,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k768_n1928",
        shape: FixedShape {
            m: 1536,
            k: 768,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k384_n1536",
        shape: FixedShape {
            m: 2048,
            k: 384,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k768_n1536",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k768_n1928",
        shape: FixedShape {
            m: 3072,
            k: 768,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k384_n1536",
        shape: FixedShape {
            m: 4096,
            k: 384,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4621_k384_n1536",
        shape: FixedShape {
            m: 4621,
            k: 384,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4621_k768_n1536",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k768_n2304",
        shape: FixedShape {
            m: 3072,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k512_n1928",
        shape: FixedShape {
            m: 1536,
            k: 512,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k520_n1928",
        shape: FixedShape {
            m: 1536,
            k: 520,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k512_n1536",
        shape: FixedShape {
            m: 4096,
            k: 512,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k520_n1536",
        shape: FixedShape {
            m: 4096,
            k: 520,
            n: 1536,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
];

const FIXED_AUTO_VENDOR_TF32_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: FixedTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: FixedTile::Tf32Sm120M64S2,
    },
];

const FIXED_AUTO_VENDOR_MIXED_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: FixedTile::Sm120Half(FixedSm120HalfTile::M64N64Bk64S2),
    },
];

const FIXED_AUTO_VENDOR_EXACT_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: FixedTile::F32N128S2,
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::F32N128S2,
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: FixedTile::Legacy,
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: FixedTile::Legacy,
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: FixedTile::Legacy,
    },
];

fn fixed_auto_vendor_expected_exact_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
) -> FixedTile {
    match device_cc {
        (12, 0) => cell.expected,
        (12, 1) => FixedTile::Legacy,
        _ => panic!(
            "AUTO/vendor census does not admit CC{}.{}",
            device_cc.0, device_cc.1
        ),
    }
}

fn fixed_auto_vendor_expected_half_tile(
    cell: FixedAutoVendorCell,
    dtype: WeightDtype,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
    has_bias: bool,
) -> FixedTile {
    if device_cc == (12, 0)
        && sm_count == 170
        && nvrtc_version == (13, 2)
        && matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        let dims = (cell.shape.m, cell.shape.k, cell.shape.n);
        let s3 = dims == (4621, 768, 2304)
            || (!has_bias
                && (matches!(dims, (2048, 1928, 2304) | (1536, 1032, 1536))
                    || (dtype == WeightDtype::F16
                        && matches!(
                            dims,
                            (1024, 1928, 1928)
                                | (1024, 1928, 2304)
                                | (1536, 1928, 1536)
                                | (2048, 768, 2304)
                                | (3072, 768, 2304)
                                | (4621, 768, 1536)
                        ))));
        if s3 {
            return FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S3);
        }
    }
    if device_cc == (12, 0)
        && sm_count == 170
        && nvrtc_version == (13, 2)
        && matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16)
        && matches!(
            (cell.shape.m, cell.shape.k, cell.shape.n),
            (512, 1928, 2304)
                | (1536, 1928, 2304)
                | (1536, 1928, 1928)
                | (3072, 1928, 1928)
                | (4096, 520, 1536)
                | (2048, 1928, 1536)
                | (1024, 1928, 1928)
                | (1024, 1928, 2304)
                | (1536, 1032, 1536)
                | (1536, 1928, 1536)
                | (4621, 1928, 1928)
                | (1536, 768, 1536)
        )
    {
        return FixedTile::Sm120Half(FixedSm120HalfTile::M64N128Bk64S2);
    }
    cell.expected
}

fn fixed_auto_vendor_expected_tf32_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
) -> FixedTile {
    if device_cc == (12, 0) && sm_count == 170 && nvrtc_version == (13, 2) {
        match (cell.shape.m, cell.shape.k, cell.shape.n) {
            (4621, 768, 2304) => return FixedTile::Tf32Sm120M128S2,
            (2048, 768, 2304) => return FixedTile::Tf32Sm120M64S2ProducerWarp,
            _ => {}
        }
    }
    cell.expected
}

fn fixed_auto_vendor_expected_mixed_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
) -> FixedTile {
    if device_cc == (12, 0) && sm_count == 170 && nvrtc_version == (13, 2) {
        return cell.expected;
    }
    if cell.label == "hot_b" {
        FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2)
    } else {
        cell.expected
    }
}

#[test]
fn fixed_auto_vendor_exact_expectations_are_device_specific() {
    let hot_a = FIXED_AUTO_VENDOR_EXACT_CELLS[0];
    let hot_b = FIXED_AUTO_VENDOR_EXACT_CELLS[1];
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 0)),
        FixedTile::F32N128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 0)),
        FixedTile::F32N128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 1)),
        FixedTile::Legacy
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 1)),
        FixedTile::Legacy
    );
}

#[test]
fn fixed_auto_vendor_tf32_expectation_tracks_the_qualified_compiler_cells() {
    let hot_b = FIXED_AUTO_VENDOR_TF32_CELLS[1];
    let hot_d = FIXED_AUTO_VENDOR_TF32_CELLS[3];
    assert_eq!(
        fixed_auto_vendor_expected_tf32_tile(hot_b, (12, 0), 170, (13, 2)),
        FixedTile::Tf32Sm120M128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_tf32_tile(hot_d, (12, 0), 170, (13, 2)),
        FixedTile::Tf32Sm120M64S2ProducerWarp
    );
    for (device_cc, sm_count, nvrtc_version) in [
        ((12, 0), 169, (13, 2)),
        ((12, 1), 170, (13, 2)),
        ((12, 0), 170, (12, 8)),
        ((12, 0), 170, (13, 0)),
        ((12, 0), 170, (13, 3)),
    ] {
        assert_eq!(
            fixed_auto_vendor_expected_tf32_tile(hot_b, device_cc, sm_count, nvrtc_version,),
            FixedTile::Tf32Sm120M64S2
        );
        assert_eq!(
            fixed_auto_vendor_expected_tf32_tile(hot_d, device_cc, sm_count, nvrtc_version,),
            FixedTile::Tf32Sm120M64S2
        );
    }
}

#[test]
fn fixed_auto_vendor_mixed_expectation_tracks_the_qualified_compiler_cell() {
    let hot_b = FIXED_AUTO_VENDOR_MIXED_CELLS[1];
    assert_eq!(
        fixed_auto_vendor_expected_mixed_tile(hot_b, (12, 0), 170, (13, 2)),
        FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S3)
    );
    for (device_cc, sm_count, nvrtc_version) in [
        ((12, 0), 169, (13, 2)),
        ((12, 1), 170, (13, 2)),
        ((12, 0), 170, (12, 8)),
        ((12, 0), 170, (13, 0)),
    ] {
        assert_eq!(
            fixed_auto_vendor_expected_mixed_tile(hot_b, device_cc, sm_count, nvrtc_version,),
            FixedTile::Sm120Half(FixedSm120HalfTile::M128N128Bk32S2)
        );
    }
}

fn configure_fixed_auto_vendor_custom(ctx: &GpuCtx, policy: F32TriadPolicy) {
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_bi_tensor_cores(false);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(policy);
}

fn configure_fixed_auto_vendor_vendor(ctx: &GpuCtx, policy: F32TriadPolicy) {
    ctx.set_batch_invariant(false);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_bi_tensor_cores(false);
    ctx.set_fast_gemm(true);
    ctx.set_f32_triad_policy(policy);
}

fn launch_fixed_auto_vendor_custom(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
) -> FixedTile {
    fixed_forward(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production Fixed AUTO launch")
}

fn launch_fixed_auto_vendor_vendor(ctx: &GpuCtx, operands: FixedFwdOperands, shape: FixedShape) {
    gpu_gemm_typed_forward_raw(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("fast cuBLAS launch");
}

fn fixed_auto_vendor_custom_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    policy: F32TriadPolicy,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_custom(ctx, policy);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed AUTO window start");
    for _ in 0..iterations {
        launch_fixed_auto_vendor_custom(ctx, operands, shape);
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed AUTO window end");
    f64::from(start.elapsed_ms(&end).expect("measure Fixed AUTO window")) * 1000.0
        / iterations as f64
}

fn fixed_auto_vendor_vendor_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    policy: F32TriadPolicy,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_vendor(ctx, policy);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record fast cuBLAS window start");
    for _ in 0..iterations {
        launch_fixed_auto_vendor_vendor(ctx, operands, shape);
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record fast cuBLAS window end");
    f64::from(start.elapsed_ms(&end).expect("measure fast cuBLAS window")) * 1000.0
        / iterations as f64
}

fn fixed_auto_vendor_iterations(pilot_us: f64) -> usize {
    assert!(
        pilot_us.is_finite() && pilot_us > 0.0,
        "pilot latency must be finite and positive, got {pilot_us}"
    );
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

fn run_fixed_auto_vendor_cell(
    ctx: &GpuCtx,
    device_cc: (u32, u32),
    sm_count: u32,
    row: FixedAutoVendorRow,
    cell: FixedAutoVendorCell,
    has_bias: bool,
) {
    let shape = cell.shape;
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, row.input_dtype).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, row.input_dtype).expect("B allocation");
    let custom = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, row.output_dtype)
        .expect("custom allocation");
    let vendor = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, row.output_dtype)
        .expect("vendor allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
        .expect("B upload");
    let bias = has_bias.then(|| {
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5_0005))
            .expect("bias upload");
        bias
    });
    let custom_operands = FixedFwdOperands {
        c: typed(&custom, row.output_dtype),
        x: typed(&a, row.input_dtype),
        w: typed(&b, row.input_dtype),
        bias_ptr: bias.as_ref().map(DtypedBuf::cached_ptr),
    };
    let vendor_operands = FixedFwdOperands {
        c: typed(&vendor, row.output_dtype),
        ..custom_operands
    };

    configure_fixed_auto_vendor_custom(ctx, row.policy);
    let selected = launch_fixed_auto_vendor_custom(ctx, custom_operands, shape);
    let expected = if row.name == "f32_exact" {
        fixed_auto_vendor_expected_exact_tile(cell, device_cc)
    } else if row.name == "tf32" {
        fixed_auto_vendor_expected_tf32_tile(
            cell,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
        )
    } else if row.output_dtype == WeightDtype::F32
        && matches!(row.input_dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        fixed_auto_vendor_expected_mixed_tile(
            cell,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
        )
    } else if row.output_dtype == row.input_dtype
        && matches!(row.input_dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        fixed_auto_vendor_expected_half_tile(
            cell,
            row.input_dtype,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
            has_bias,
        )
    } else {
        cell.expected
    };
    assert_eq!(
        selected, expected,
        "unexpected production Fixed AUTO tile for row={} cell={} M{} K{} N{}",
        row.name, cell.label, shape.m, shape.k, shape.n
    );
    ctx.stream
        .synchronize()
        .expect("selector assertion synchronization");

    configure_fixed_auto_vendor_custom(ctx, row.policy);
    for _ in 0..128 {
        launch_fixed_auto_vendor_custom(ctx, custom_operands, shape);
    }
    ctx.stream.synchronize().expect("Fixed AUTO warmup sync");
    configure_fixed_auto_vendor_vendor(ctx, row.policy);
    for _ in 0..128 {
        launch_fixed_auto_vendor_vendor(ctx, vendor_operands, shape);
    }
    ctx.stream.synchronize().expect("fast cuBLAS warmup sync");

    let custom_pilot_us =
        fixed_auto_vendor_custom_window_us(ctx, custom_operands, shape, row.policy, 16);
    let vendor_pilot_us =
        fixed_auto_vendor_vendor_window_us(ctx, vendor_operands, shape, row.policy, 16);
    let custom_iterations = fixed_auto_vendor_iterations(custom_pilot_us);
    let vendor_iterations = fixed_auto_vendor_iterations(vendor_pilot_us);

    for custom_first in [true, false] {
        let mut custom_us = Vec::with_capacity(101);
        let mut vendor_us = Vec::with_capacity(101);
        let mut ratios = Vec::with_capacity(101);
        for _ in 0..101 {
            let (custom_elapsed_us, vendor_elapsed_us) = if custom_first {
                (
                    fixed_auto_vendor_custom_window_us(
                        ctx,
                        custom_operands,
                        shape,
                        row.policy,
                        custom_iterations,
                    ),
                    fixed_auto_vendor_vendor_window_us(
                        ctx,
                        vendor_operands,
                        shape,
                        row.policy,
                        vendor_iterations,
                    ),
                )
            } else {
                let vendor_elapsed_us = fixed_auto_vendor_vendor_window_us(
                    ctx,
                    vendor_operands,
                    shape,
                    row.policy,
                    vendor_iterations,
                );
                let custom_elapsed_us = fixed_auto_vendor_custom_window_us(
                    ctx,
                    custom_operands,
                    shape,
                    row.policy,
                    custom_iterations,
                );
                (custom_elapsed_us, vendor_elapsed_us)
            };
            assert!(
                custom_elapsed_us.is_finite()
                    && custom_elapsed_us > 0.0
                    && vendor_elapsed_us.is_finite()
                    && vendor_elapsed_us > 0.0,
                "measured latencies must be finite and positive"
            );
            custom_us.push(custom_elapsed_us);
            vendor_us.push(vendor_elapsed_us);
            ratios.push(custom_elapsed_us / vendor_elapsed_us);
        }
        custom_us.sort_by(f64::total_cmp);
        vendor_us.sort_by(f64::total_cmp);
        ratios.sort_by(f64::total_cmp);
        let comparator_order = if custom_first {
            "custom_then_vendor"
        } else {
            "vendor_then_custom"
        };
        println!(
            concat!(
                "{{\"schema\":\"MambaBiFixedAutoVendorV2\",",
                "\"suite\":\"fixed_production_auto_vs_fast_cublas_hot_and_selector_census\",",
                "\"tuning_table_revision\":{},\"device_cc\":\"{}.{}\",\"sm_count\":{},",
                "\"row\":\"{}\",\"input_dtype\":\"{}\",\"output_dtype\":\"{}\",\"f32_policy\":\"{}\",",
                "\"op\":\"nn\",\"bias\":{},\"cell_label\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                "\"auto_tile\":\"{:?}\",\"cublas_mode\":\"fast_tf32_allowed\",",
                "\"comparator_order\":\"{}\",\"warmups\":128,\"pilot_iterations\":16,",
                "\"target_window_ms\":5.0,\"windows\":101,",
                "\"custom_iterations\":{},\"vendor_iterations\":{},",
                "\"custom_p50_us\":{:.9},\"custom_p95_us\":{:.9},",
                "\"vendor_p50_us\":{:.9},\"vendor_p95_us\":{:.9},",
                "\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
            ),
            TUNING_TABLE_REVISION,
            device_cc.0,
            device_cc.1,
            sm_count,
            row.name,
            row.input_dtype.as_str(),
            row.output_dtype.as_str(),
            row.policy_name,
            has_bias,
            cell.label,
            shape.m,
            shape.k,
            shape.n,
            selected,
            comparator_order,
            custom_iterations,
            vendor_iterations,
            percentile(&custom_us, 0.50),
            percentile(&custom_us, 0.95),
            percentile(&vendor_us, 0.50),
            percentile(&vendor_us, 0.95),
            percentile(&ratios, 0.50),
            percentile(&ratios, 0.95),
        );
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits production AUTO/vendor evidence"]
fn fixed_production_auto_vs_fast_cublas_hot_and_selector_census() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let device_cc = device.compute_capability;
    let sm_count = device.multiprocessor_count();
    if !matches!(device_cc, (12, 0) | (12, 1)) || sm_count != 170 {
        eprintln!(
            "skipping production Fixed AUTO/vendor census on CC{}.{} with {} SMs; CC12.0/12.1 with exactly 170 SMs required",
            device_cc.0, device_cc.1, sm_count
        );
        return;
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(ctx.tf32(), "fast cuBLAS TF32 must remain enabled");
    println!();
    let rows = [
        FixedAutoVendorRow {
            name: "bf16",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::Bf16,
            policy: F32TriadPolicy::ExactScalarFmaV1,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_HALF_CELLS,
        },
        FixedAutoVendorRow {
            name: "f16",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F16,
            policy: F32TriadPolicy::ExactScalarFmaV1,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_HALF_CELLS,
        },
        FixedAutoVendorRow {
            name: "bf16_f32",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFmaV1,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_MIXED_CELLS,
        },
        FixedAutoVendorRow {
            name: "f16_f32",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFmaV1,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_MIXED_CELLS,
        },
        FixedAutoVendorRow {
            name: "tf32",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::AllowDeterministicTf32V1,
            policy_name: "allow_deterministic_tf32_v1",
            cells: FIXED_AUTO_VENDOR_TF32_CELLS,
        },
        FixedAutoVendorRow {
            name: "f32_exact",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFmaV1,
            policy_name: "exact_scalar_fma_v1",
            cells: FIXED_AUTO_VENDOR_EXACT_CELLS,
        },
    ];
    let row_filter = std::env::var("MAMBA_FIXED_AUTO_VENDOR_ROW").ok();
    if let Some(filter) = row_filter.as_deref() {
        assert!(
            rows.iter().any(|row| row.name == filter),
            "unknown Fixed AUTO/vendor row filter: {filter}"
        );
    }
    let cell_filter = std::env::var("MAMBA_FIXED_AUTO_VENDOR_CELL").ok();
    if let Some(filter) = cell_filter.as_deref() {
        assert!(
            rows.iter()
                .flat_map(|row| row.cells)
                .any(|cell| cell.label == filter),
            "unknown Fixed AUTO/vendor cell filter: {filter}"
        );
    }
    let has_bias = match std::env::var("MAMBA_FIXED_AUTO_VENDOR_BIAS").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(value) => panic!("MAMBA_FIXED_AUTO_VENDOR_BIAS must be 0 or 1, got {value}"),
    };
    for row in rows {
        if row_filter
            .as_deref()
            .is_some_and(|filter| filter != row.name)
        {
            continue;
        }
        for &cell in row.cells {
            if cell_filter
                .as_deref()
                .is_some_and(|filter| filter != cell.label)
            {
                continue;
            }
            run_fixed_auto_vendor_cell(&ctx, device_cc, sm_count, row, cell, has_bias);
        }
    }
}

const F32_N128_GUARD: usize = 32;
const F32_N128_SENTINEL: f32 = 19.25;

fn f32_n128_offset_buffer(ctx: &GpuCtx, logical: &[f32], offset: usize, label: &str) -> DtypedBuf {
    let storage_len = offset + logical.len().max(1);
    let mut storage = vec![-31.5f32; storage_len];
    if !logical.is_empty() {
        storage[offset..offset + logical.len()].copy_from_slice(logical);
    }
    let buffer = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
        .unwrap_or_else(|error| panic!("{label} allocation: {error}"));
    buffer
        .upload_f32(&ctx.stream, &storage)
        .unwrap_or_else(|error| panic!("{label} upload: {error}"));
    buffer
}

fn f32_n128_guarded_output(
    ctx: &GpuCtx,
    output_len: usize,
    output_offset: usize,
    poison: f32,
    label: &str,
) -> DtypedBuf {
    let storage_len = F32_N128_GUARD + output_offset + output_len + F32_N128_GUARD;
    let mut storage = vec![F32_N128_SENTINEL; storage_len];
    let start = F32_N128_GUARD + output_offset;
    storage[start..start + output_len].fill(poison);
    let output = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
        .unwrap_or_else(|error| panic!("{label} allocation: {error}"));
    output
        .upload_f32(&ctx.stream, &storage)
        .unwrap_or_else(|error| panic!("{label} initialization: {error}"));
    output
}

fn f32_n128_output_operands(
    output: &DtypedBuf,
    output_offset: usize,
    a: &DtypedBuf,
    b: &DtypedBuf,
    input_offset: usize,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
) -> FixedFwdOperands {
    let element_bytes = WeightDtype::F32.size_bytes() as u64;
    FixedFwdOperands {
        c: TypedPtr {
            ptr: output.cached_ptr() + ((F32_N128_GUARD + output_offset) as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        x: TypedPtr {
            ptr: a.cached_ptr() + (input_offset as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        w: TypedPtr {
            ptr: b.cached_ptr() + (input_offset as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        bias_ptr,
    }
}

fn f32_n128_logical_and_guards(
    ctx: &GpuCtx,
    output: &DtypedBuf,
    output_len: usize,
    output_offset: usize,
    label: &str,
) -> Vec<u32> {
    let all = f32_bits(
        ctx,
        output,
        F32_N128_GUARD + output_offset + output_len + F32_N128_GUARD,
    );
    let start = F32_N128_GUARD + output_offset;
    let sentinel = F32_N128_SENTINEL.to_bits();
    assert!(
        all[..start].iter().all(|&bits| bits == sentinel),
        "{label} leading guard changed"
    );
    assert!(
        all[start + output_len..]
            .iter()
            .all(|&bits| bits == sentinel),
        "{label} trailing guard changed"
    );
    all[start..start + output_len].to_vec()
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_f32_n128_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let measured_device =
        device.compute_capability == (12, 0) && device.multiprocessor_count() == 170;
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    let shapes = [
        FixedShape { m: 1, k: 0, n: 1 },
        FixedShape {
            m: 63,
            k: 31,
            n: 127,
        },
        FixedShape {
            m: 64,
            k: 32,
            n: 128,
        },
        FixedShape {
            m: 65,
            k: 33,
            n: 129,
        },
        FixedShape {
            m: 63,
            k: 63,
            n: 129,
        },
        FixedShape {
            m: 64,
            k: 64,
            n: 128,
        },
        FixedShape {
            m: 65,
            k: 65,
            n: 127,
        },
        FixedShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        FixedShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        FixedShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        FixedShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];

    for shape in shapes {
        let a_host = synth(shape.m * shape.k, 0xa128_0001 ^ shape.k as u64);
        let b_host = synth(shape.k * shape.n, 0xb128_0001 ^ shape.n as u64);
        let bias_host = synth(shape.n, 0xb1a5_1280 ^ shape.m as u64);
        let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "bias");
        let output_len = shape.m * shape.n;
        for input_offset in [0usize, 1] {
            let a = f32_n128_offset_buffer(&ctx, &a_host, input_offset, "A");
            let b = f32_n128_offset_buffer(&ctx, &b_host, input_offset, "B");
            for output_offset in [0usize, 1] {
                for bias_ptr in [None, Some(bias.cached_ptr())] {
                    let oracle = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        -3.0,
                        "legacy oracle",
                    );
                    let auto = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        5.0,
                        "production AUTO",
                    );
                    let legacy = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        7.0,
                        "incumbent S2",
                    );
                    let candidate_a = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        -7.0,
                        "candidate A",
                    );
                    let candidate_b = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        9.0,
                        "candidate B",
                    );
                    let oracle_ops = f32_n128_output_operands(
                        &oracle,
                        output_offset,
                        &a,
                        &b,
                        input_offset,
                        bias_ptr,
                    );
                    let auto_ops = FixedFwdOperands {
                        c: f32_n128_output_operands(
                            &auto,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let legacy_ops = FixedFwdOperands {
                        c: f32_n128_output_operands(
                            &legacy,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let candidate_a_ops = FixedFwdOperands {
                        c: f32_n128_output_operands(
                            &candidate_a,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let candidate_b_ops = FixedFwdOperands {
                        c: f32_n128_output_operands(
                            &candidate_b,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };

                    fixed_forward_f32_legacy_baseline(&ctx, oracle_ops, shape)
                        .expect("legacy oracle launch");
                    fixed_forward_with_tile(&ctx, legacy_ops, shape, FixedTile::Legacy)
                        .expect("forced incumbent S2 launch");
                    let selected = fixed_forward(
                        &ctx,
                        auto_ops.c,
                        auto_ops.x,
                        auto_ops.w,
                        bias_ptr,
                        (shape.m, shape.k, shape.n),
                    )
                    .expect("production S2 launch");
                    let expected_auto = if measured_device
                        && matches!(
                            (shape.m, shape.k, shape.n),
                            (4621, 384, 1928) | (4621, 768, 2304)
                        ) {
                        FixedTile::F32N128S2
                    } else {
                        FixedTile::Legacy
                    };
                    assert_eq!(selected, expected_auto, "production exact-F32 AUTO route");
                    fixed_forward_with_tile(&ctx, candidate_a_ops, shape, FixedTile::F32N128S2)
                        .expect("first forced N128 launch");
                    fixed_forward_with_tile(&ctx, candidate_b_ops, shape, FixedTile::F32N128S2)
                        .expect("second forced N128 launch");
                    ctx.stream.synchronize().expect("N128 bit-gate sync");

                    let label = format!(
                        "M{} K{} N{} bias={} input_offset={} output_offset={}",
                        shape.m,
                        shape.k,
                        shape.n,
                        bias_ptr.is_some(),
                        input_offset,
                        output_offset
                    );
                    let oracle_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &oracle,
                        output_len,
                        output_offset,
                        &format!("oracle {label}"),
                    );
                    let auto_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &auto,
                        output_len,
                        output_offset,
                        &format!("AUTO {label}"),
                    );
                    let legacy_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &legacy,
                        output_len,
                        output_offset,
                        &format!("incumbent S2 {label}"),
                    );
                    let candidate_a_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &candidate_a,
                        output_len,
                        output_offset,
                        &format!("candidate A {label}"),
                    );
                    let candidate_b_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &candidate_b,
                        output_len,
                        output_offset,
                        &format!("candidate B {label}"),
                    );
                    assert_eq!(candidate_a_bits, oracle_bits, "candidate/oracle {label}");
                    assert_eq!(
                        candidate_a_bits, legacy_bits,
                        "candidate/incumbent S2 {label}"
                    );
                    assert_eq!(auto_bits, oracle_bits, "AUTO/oracle {label}");
                    assert_eq!(candidate_a_bits, candidate_b_bits, "repeatability {label}");
                }
            }
        }
    }

    let shape = FixedShape {
        m: 65,
        k: 65,
        n: 127,
    };
    let a_host = synth(shape.m * shape.k, 0xa128_0a11);
    let b_host = synth(shape.k * shape.n, 0xb128_0b11);
    let bias_host = synth(shape.n, 0xb128_b1a5);
    let a = f32_n128_offset_buffer(&ctx, &a_host, 1, "graph A");
    let b = f32_n128_offset_buffer(&ctx, &b_host, 1, "graph B");
    let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "graph bias");
    let output = f32_n128_guarded_output(&ctx, shape.m * shape.n, 1, -13.0, "graph output");
    let operands = f32_n128_output_operands(&output, 1, &a, &b, 1, Some(bias.cached_ptr()));
    let run = || fixed_forward_with_tile(&ctx, operands, shape, FixedTile::F32N128S2);
    run().expect("eager N128 launch");
    ctx.stream.synchronize().expect("eager N128 sync");
    let eager = f32_n128_logical_and_guards(&ctx, &output, shape.m * shape.n, 1, "eager");
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture N128 graph");
    for replay in 0..10 {
        graph.launch().expect("N128 graph launch");
        ctx.stream.synchronize().expect("N128 graph sync");
        assert_eq!(
            f32_n128_logical_and_guards(&ctx, &output, shape.m * shape.n, 1, "graph"),
            eager,
            "N128 graph replay {replay} changed bits"
        );
    }

    let k = 65usize;
    let n = 127usize;
    let large_a_host = synth(65 * k, 0xa128_0065);
    let b_host = synth(k * n, 0xb128_0065);
    let bias_host = synth(n, 0xb1a5_0065);
    let small_a = f32_n128_offset_buffer(&ctx, &large_a_host[..64 * k], 0, "prefix A64");
    let large_a = f32_n128_offset_buffer(&ctx, &large_a_host, 0, "prefix A65");
    let b = f32_n128_offset_buffer(&ctx, &b_host, 0, "prefix B");
    let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "prefix bias");
    let small = f32_n128_guarded_output(&ctx, 64 * n, 0, -17.0, "prefix C64");
    let large = f32_n128_guarded_output(&ctx, 65 * n, 0, 23.0, "prefix C65");
    let small_shape = FixedShape { m: 64, k, n };
    let large_shape = FixedShape { m: 65, k, n };
    fixed_forward_with_tile(
        &ctx,
        f32_n128_output_operands(&small, 0, &small_a, &b, 0, Some(bias.cached_ptr())),
        small_shape,
        FixedTile::F32N128S2,
    )
    .expect("M64 prefix launch");
    fixed_forward_with_tile(
        &ctx,
        f32_n128_output_operands(&large, 0, &large_a, &b, 0, Some(bias.cached_ptr())),
        large_shape,
        FixedTile::F32N128S2,
    )
    .expect("M65 prefix launch");
    ctx.stream.synchronize().expect("prefix sync");
    assert_eq!(
        f32_n128_logical_and_guards(&ctx, &small, 64 * n, 0, "M64 prefix"),
        f32_n128_logical_and_guards(&ctx, &large, 65 * n, 0, "M65 prefix")[..64 * n],
        "N128 first 64 rows changed when M grew from 64 to 65"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_f32_n128_resource_gate_sm120() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(
        device.compute_capability,
        (12, 0),
        "resource gate requires CC12.0"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_fixed_f32_n128_resources(&ctx);
}

fn assert_fixed_f32_n128_resources(ctx: &GpuCtx) {
    let function = &ctx.kernels.gemm_bi_f32_f32_n128_s2;
    let static_shared_bytes = function
        .shared_size_bytes()
        .expect("N128 static shared size");
    let local_bytes = function.local_size_bytes().expect("N128 local size");
    let registers = function.num_regs().expect("N128 registers");
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
        .expect("N128 occupancy");
    println!(
        "symbol=gemm_bi_f32_f32_n128_s2 static_shared_bytes={static_shared_bytes} dynamic_shared_bytes=0 local_bytes={local_bytes} registers={registers} occupancy_blocks_per_sm={occupancy}"
    );
    assert_eq!(static_shared_bytes, 49_152);
    assert_eq!(local_bytes, 0);
    assert!(
        registers <= 128,
        "N128 uses {registers} registers per thread"
    );
    assert!(occupancy >= 2, "N128 occupancy is {occupancy} blocks/SM");
}

fn fixed_f32_n128_baseline_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    iterations: usize,
) -> f64 {
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    fixed_tile_window_us(ctx, operands, shape, FixedTile::Legacy, iterations)
}

fn fixed_f32_n128_candidate_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    iterations: usize,
) -> f64 {
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    fixed_tile_window_us(ctx, operands, shape, FixedTile::F32N128S2, iterations)
}

fn fixed_f32_n128_cublas_window_us(
    ctx: &GpuCtx,
    operands: FixedFwdOperands,
    shape: FixedShape,
    iterations: usize,
) -> f64 {
    ctx.set_batch_invariant(false);
    ctx.set_fast_gemm(true);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record cuBLAS diagnostic start");
    for _ in 0..iterations {
        gpu_gemm_typed_forward_raw(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .expect("fast cuBLAS diagnostic launch");
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record cuBLAS diagnostic end");
    f64::from(start.elapsed_ms(&end).expect("measure cuBLAS diagnostic")) * 1000.0
        / iterations as f64
}

fn f32_n128_calibrated_iterations(pilot_us: f64) -> usize {
    assert!(pilot_us.is_finite() && pilot_us > 0.0);
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired F32 N128 evidence"]
fn fixed_f32_n128_paired_a_e() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(
        device.compute_capability,
        (12, 0),
        "paired gate requires CC12.0"
    );
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "paired gate requires 170 SMs"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(ctx.tf32(), "fast cuBLAS diagnostic requires TF32 enabled");
    assert_fixed_f32_n128_resources(&ctx);
    let cells = [
        (
            "A",
            FixedShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
        ),
        (
            "B",
            FixedShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
        ),
        (
            "C",
            FixedShape {
                m: 4621,
                k: 1928,
                n: 384,
            },
        ),
        (
            "D",
            FixedShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
        ),
        (
            "E",
            FixedShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ),
    ];
    let mut eligible_points = Vec::new();
    let mut rejected_points = Vec::new();
    let mut paired_records = 0;
    let mut cublas_records = 0;
    for (label, shape) in cells {
        let a = f32_n128_offset_buffer(
            &ctx,
            &synth(shape.m * shape.k, 0xa128_ae00 ^ shape.m as u64),
            0,
            "paired A",
        );
        let b = f32_n128_offset_buffer(
            &ctx,
            &synth(shape.k * shape.n, 0xb128_ae00 ^ shape.n as u64),
            0,
            "paired B",
        );
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired baseline output");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired candidate output");
        let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired cuBLAS output");
        let baseline_ops = FixedFwdOperands {
            c: typed(&baseline, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let candidate_ops = FixedFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            ..baseline_ops
        };
        let cublas_ops = FixedFwdOperands {
            c: typed(&cublas, WeightDtype::F32),
            ..baseline_ops
        };

        fixed_forward_f32_legacy_baseline(&ctx, baseline_ops, shape)
            .expect("paired bit oracle launch");
        fixed_forward_with_tile(&ctx, candidate_ops, shape, FixedTile::F32N128S2)
            .expect("paired bit candidate launch");
        ctx.stream.synchronize().expect("paired bit-gate sync");
        assert_eq!(
            f32_bits(&ctx, &candidate, shape.m * shape.n),
            f32_bits(&ctx, &baseline, shape.m * shape.n),
            "paired candidate/oracle bit gate for cell {label}"
        );

        for _ in 0..128 {
            fixed_forward_with_tile(&ctx, candidate_ops, shape, FixedTile::F32N128S2)
                .expect("warm N128 candidate");
        }
        ctx.stream.synchronize().expect("N128 warmup sync");
        for _ in 0..128 {
            fixed_forward_with_tile(&ctx, baseline_ops, shape, FixedTile::Legacy)
                .expect("warm incumbent S2");
        }
        ctx.stream.synchronize().expect("S2 warmup sync");
        let candidate_iterations = f32_n128_calibrated_iterations(
            fixed_f32_n128_candidate_window_us(&ctx, candidate_ops, shape, 16),
        );
        let baseline_iterations = f32_n128_calibrated_iterations(
            fixed_f32_n128_baseline_window_us(&ctx, baseline_ops, shape, 16),
        );

        let mut cell_eligible = true;
        for candidate_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            let mut candidate_us = Vec::with_capacity(101);
            let mut baseline_us = Vec::with_capacity(101);
            for _ in 0..101 {
                let (candidate_elapsed, baseline_elapsed) = if candidate_first {
                    (
                        fixed_f32_n128_candidate_window_us(
                            &ctx,
                            candidate_ops,
                            shape,
                            candidate_iterations,
                        ),
                        fixed_f32_n128_baseline_window_us(
                            &ctx,
                            baseline_ops,
                            shape,
                            baseline_iterations,
                        ),
                    )
                } else {
                    let baseline_elapsed = fixed_f32_n128_baseline_window_us(
                        &ctx,
                        baseline_ops,
                        shape,
                        baseline_iterations,
                    );
                    let candidate_elapsed = fixed_f32_n128_candidate_window_us(
                        &ctx,
                        candidate_ops,
                        shape,
                        candidate_iterations,
                    );
                    (candidate_elapsed, baseline_elapsed)
                };
                assert!(
                    candidate_elapsed.is_finite()
                        && candidate_elapsed > 0.0
                        && baseline_elapsed.is_finite()
                        && baseline_elapsed > 0.0,
                    "paired exact-arm latencies must be finite and positive"
                );
                candidate_us.push(candidate_elapsed);
                baseline_us.push(baseline_elapsed);
                ratios.push(candidate_elapsed / baseline_elapsed);
            }
            candidate_us.sort_by(f64::total_cmp);
            baseline_us.sort_by(f64::total_cmp);
            ratios.sort_by(f64::total_cmp);
            assert_eq!(ratios.len(), 101, "paired exact-arm protocol window count");
            let order = if candidate_first {
                "candidate_then_baseline"
            } else {
                "baseline_then_candidate"
            };
            let ratio_p95 = percentile(&ratios, 0.95);
            println!(
                concat!(
                    "{{\"schema\":\"MambaBiFixedF32N128PairedV1\",",
                    "\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                    "\"order\":\"{}\",\"warmups\":128,\"pilot_iterations\":16,",
                    "\"target_window_ms\":5.0,\"pairs\":101,",
                    "\"candidate_iterations\":{},",
                    "\"baseline_iterations\":{},",
                    "\"candidate_p50_us\":{:.9},\"baseline_p50_us\":{:.9},",
                    "\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
                ),
                label,
                shape.m,
                shape.k,
                shape.n,
                order,
                candidate_iterations,
                baseline_iterations,
                percentile(&candidate_us, 0.50),
                percentile(&baseline_us, 0.50),
                percentile(&ratios, 0.50),
                ratio_p95,
            );
            cell_eligible &= ratio_p95 <= 0.98;
            paired_records += 1;
        }
        if cell_eligible {
            eligible_points.push(label);
        } else {
            rejected_points.push(label);
        }

        ctx.set_batch_invariant(false);
        ctx.set_fast_gemm(true);
        for _ in 0..128 {
            gpu_gemm_typed_forward_raw(
                &ctx,
                cublas_ops.c,
                cublas_ops.x,
                cublas_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("warm fast cuBLAS diagnostic");
        }
        ctx.stream
            .synchronize()
            .expect("cuBLAS diagnostic warmup sync");
        let cublas_iterations = f32_n128_calibrated_iterations(fixed_f32_n128_cublas_window_us(
            &ctx, cublas_ops, shape, 16,
        ));
        for candidate_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            for _ in 0..101 {
                let (candidate_elapsed, cublas_elapsed) = if candidate_first {
                    (
                        fixed_f32_n128_candidate_window_us(
                            &ctx,
                            candidate_ops,
                            shape,
                            candidate_iterations,
                        ),
                        fixed_f32_n128_cublas_window_us(&ctx, cublas_ops, shape, cublas_iterations),
                    )
                } else {
                    let cublas_elapsed =
                        fixed_f32_n128_cublas_window_us(&ctx, cublas_ops, shape, cublas_iterations);
                    let candidate_elapsed = fixed_f32_n128_candidate_window_us(
                        &ctx,
                        candidate_ops,
                        shape,
                        candidate_iterations,
                    );
                    (candidate_elapsed, cublas_elapsed)
                };
                assert!(
                    candidate_elapsed.is_finite()
                        && candidate_elapsed > 0.0
                        && cublas_elapsed.is_finite()
                        && cublas_elapsed > 0.0,
                    "candidate/cuBLAS latencies must be finite and positive"
                );
                ratios.push(candidate_elapsed / cublas_elapsed);
            }
            ratios.sort_by(f64::total_cmp);
            assert_eq!(ratios.len(), 101, "candidate/cuBLAS protocol window count");
            let order = if candidate_first {
                "candidate_then_cublas"
            } else {
                "cublas_then_candidate"
            };
            println!(
                "{{\"schema\":\"MambaBiFixedF32N128CublasDiagnosticV1\",\"cell\":\"{label}\",\"m\":{},\"k\":{},\"n\":{},\"order\":\"{order}\",\"pairs\":101,\"candidate_iterations\":{candidate_iterations},\"cublas_iterations\":{cublas_iterations},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}",
                shape.m,
                shape.k,
                shape.n,
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
            cublas_records += 1;
        }
    }
    assert_eq!(paired_records, 10, "five cells times two exact-arm orders");
    assert_eq!(
        cublas_records, 10,
        "five cells times two candidate/cuBLAS orders"
    );
    assert_eq!(
        eligible_points,
        vec!["A", "B"],
        "only both-order qualifying points may enter AUTO"
    );
    assert_eq!(
        rejected_points,
        vec!["C", "D", "E"],
        "the rejected general-route controls must remain ineligible"
    );
    let decision = r#"{"schema":"MambaBiFixedF32N128DecisionV1","general_route_eligible":false,"eligible_points":[{"cell":"A","m":4621,"k":384,"n":1928,"device_cc":"12.0","sm_count":170},{"cell":"B","m":4621,"k":768,"n":2304,"device_cc":"12.0","sm_count":170}],"rejected_points":["C","D","E"],"required_p95_max":0.98}"#;
    println!("{decision}");
}
