//! Direct forced-kernel coverage for cp.async reads at exact allocation ends.
//!
//! This test deliberately gives every operand its exact physical span. The
//! optional aligned prefix is before the logical subview; there is no trailing
//! storage after any tested logical input. Run the ignored test under memcheck:
//!
//! ```text
//! compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --error-exitcode 99 cargo test --features cuda --test sm80_cp_async_exact_allocation_sanitizer sm80_exact_allocation_cp_async_memcheck -- --ignored --exact --nocapture
//! ```

#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, Tf32PhysicalRoute, Tf32PortableRoute,
    Tf32PortableStages, Tf32PortableTile, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

#[derive(Clone, Copy, Debug)]
enum NnSchedule {
    Tc128,
    Tc64,
    Thin16,
}

#[derive(Clone, Copy, Debug)]
struct TypedCase {
    name: &'static str,
    schedule: NnSchedule,
    dims: (usize, usize, usize),
    a_prefix: usize,
}

const TYPED_CASES: [TypedCase; 7] = [
    TypedCase {
        name: "typed_tc128_a_tail",
        schedule: NnSchedule::Tc128,
        dims: (128, 7, 128),
        a_prefix: 0,
    },
    TypedCase {
        name: "typed_tc128_b_tail",
        schedule: NnSchedule::Tc128,
        dims: (128, 8, 127),
        a_prefix: 0,
    },
    TypedCase {
        name: "typed_tc64_a_tail",
        schedule: NnSchedule::Tc64,
        dims: (64, 7, 64),
        a_prefix: 0,
    },
    TypedCase {
        name: "typed_tc64_b_tail",
        schedule: NnSchedule::Tc64,
        dims: (64, 8, 63),
        a_prefix: 0,
    },
    TypedCase {
        name: "typed_thin16_a_tail",
        schedule: NnSchedule::Thin16,
        dims: (16, 7, 32),
        a_prefix: 0,
    },
    TypedCase {
        name: "typed_thin16_b_tail",
        schedule: NnSchedule::Thin16,
        dims: (16, 8, 31),
        a_prefix: 0,
    },
    TypedCase {
        name: "aligned_subview_at_allocation_end",
        schedule: NnSchedule::Tc64,
        dims: (64, 8, 64),
        // Eight half elements preserve 16-byte alignment. The allocation has
        // exactly prefix + M*K elements, so the subview ends at its final byte.
        a_prefix: 8,
    },
];

fn regular_tf32_route() -> Tf32PhysicalRoute {
    Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
        tile: Tf32PortableTile::M64N64,
        stages: Tf32PortableStages::S2,
    })
}

fn run_tf32_case(
    ctx: &GpuCtx,
    name: &str,
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    route: Tf32PhysicalRoute,
) -> Result<(), String> {
    let request = PhysicalQualificationRequest::contiguous(
        op,
        dims,
        PhysicalQualificationRoute::Tf32Forced(route),
    );
    let mut launch = qualify_physical_launch(ctx, request)
        .map_err(|error| format!("{name}: prepare direct TF32 route: {error}"))?;
    launch
        .seed_f32_operands(ctx, 0x5a17)
        .map_err(|error| format!("{name}: seed exact operands: {error}"))?;
    launch
        .measure_graph_window_ms(ctx, 1)
        .map_err(|error| format!("{name}: replay exact route: {error}"))?;
    launch
        .f32_output_bits(ctx)
        .map_err(|error| format!("{name}: synchronize exact route: {error}"))?;
    Ok(())
}

fn exact_typed_buffer(
    ctx: &GpuCtx,
    logical_elements: usize,
    prefix_elements: usize,
    dtype: WeightDtype,
) -> Result<(DtypedBuf, u64), String> {
    let allocation_elements = prefix_elements
        .checked_add(logical_elements)
        .ok_or_else(|| "typed exact allocation extent overflow".to_string())?;
    let buffer = DtypedBuf::zeros(&ctx.stream, allocation_elements, dtype)?;
    let values = (0..allocation_elements)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    buffer.upload_f32(&ctx.stream, &values)?;
    let byte_offset = prefix_elements
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| "typed exact subview byte offset overflow".to_string())?;
    let logical_ptr = buffer.cached_ptr() + byte_offset as u64;
    Ok((buffer, logical_ptr))
}

fn run_typed_case(ctx: &GpuCtx, dtype: WeightDtype, case: TypedCase) -> Result<(), String> {
    let (m, k, n) = case.dims;
    let (output, output_ptr) = exact_typed_buffer(ctx, m * n, 0, dtype)?;
    let (a, a_ptr) = exact_typed_buffer(ctx, m * k, case.a_prefix, dtype)?;
    let (b, b_ptr) = exact_typed_buffer(ctx, k * n, 0, dtype)?;
    let (function, bm, bn, threads, shared_mem_bytes) = match case.schedule {
        NnSchedule::Tc128 => (&ctx.kernels.gemm_bi_nn_tc_typed, 128, 128, 256, 71_680),
        NnSchedule::Tc64 => (&ctx.kernels.gemm_bi_nn_tc64_typed, 64, 64, 128, 0),
        NnSchedule::Thin16 => (&ctx.kernels.gemm_bi_nn_tc16_typed, 16, 32, 128, 0),
    };
    let config = cudarc::driver::LaunchConfig {
        grid_dim: ((m.div_ceil(bm) * n.div_ceil(bn)) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let bias_ptr = 0u64;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let (m, n, k) = (m as i32, n as i32, k as i32);
    let (lda, ldb, ldc) = (k, n, n);
    let mut builder = ctx.stream.launch_builder(function.get(dtype));
    builder.arg(&output_ptr);
    builder.arg(&a_ptr);
    builder.arg(&b_ptr);
    builder.arg(&bias_ptr);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&m);
    builder.arg(&n);
    builder.arg(&k);
    builder.arg(&lda);
    builder.arg(&ldb);
    builder.arg(&ldc);
    unsafe { builder.launch(config) }
        .map_err(|error| format!("{} {dtype:?}: direct launch: {error:?}", case.name))?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("{} {dtype:?}: synchronize: {error:?}", case.name))?;
    let mut observed = vec![0.0; output.len_elems()];
    output.download_f32(&ctx.stream, &mut observed)?;
    drop((a, b));
    Ok(())
}

#[test]
#[ignore = "requires an SM80+ CUDA GPU under compute-sanitizer"]
fn sm80_exact_allocation_cp_async_memcheck() -> Result<(), String> {
    let device = GpuDevice::new(0)?;
    if device.compute_capability < (8, 0) {
        return Err(format!(
            "exact-allocation cp.async coverage requires SM80+, found {:?}",
            device.compute_capability
        ));
    }
    let ctx = GpuCtx::new(&device)?;

    for (name, op, dims) in [
        ("tf32_nn_exact_tail", ResolvedGemmOp::Nn, (64, 3, 63)),
        ("tf32_tn_exact_tail", ResolvedGemmOp::Tn, (32, 63, 63)),
        ("tf32_nt_exact_tail", ResolvedGemmOp::Nt, (64, 64, 63)),
    ] {
        run_tf32_case(&ctx, name, op, dims, regular_tf32_route())?;
    }
    run_tf32_case(
        &ctx,
        "tf32_nn_splitk_exact_tail",
        ResolvedGemmOp::Nn,
        (16, 3, 31),
        Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        }),
    )?;
    run_tf32_case(
        &ctx,
        "tf32_nt_splitk_exact_tail",
        ResolvedGemmOp::Nt,
        (16, 32, 3),
        Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S3,
        }),
    )?;

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for case in TYPED_CASES {
            run_typed_case(&ctx, dtype, case)?;
        }
    }
    Ok(())
}
