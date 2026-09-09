#![cfg(feature = "cuda")]

//! Live actual-AUTO qualification for the exact 11 frozen Ada half-Triad cells.

use mamba_rs::mamba_ssm::gpu::context::{GpuCtx, HalfTriadPolicy};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, SM89_HALF_AUTO_CELLS,
    SM89_HALF_KERNEL_SPECS, presize_physical_qualification_suite, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, PolicyDtype, digest_hex};

#[test]
#[ignore = "requires an exclusive RTX 6000 Ada and explicit CUDA 12.8/13.0/13.2 selection"]
fn sm89_half_actual_auto_qualification() -> Result<(), String> {
    let device = GpuDevice::new(0)?;
    let ctx = GpuCtx::new(&device)?;
    let compiler = ctx
        .kernels
        .triad_sm89_half_compiler_identity()
        .ok_or_else(|| {
            format!(
                "TriadSm89Half did not bind: {:?}",
                ctx.kernels.triad_sm89_half_rejection()
            )
        })?;
    let artifact = ctx
        .kernels
        .triad_sm89_half_artifact_identity()
        .ok_or_else(|| "TriadSm89Half has no artifact identity".to_string())?;
    let route = ctx.gemm_route();

    if compiler.target.as_str() != "sm_89"
        || route.device.compute_capability != (8, 9)
        || artifact.module_kind != ModuleKind::TriadSm89Half
    {
        return Err(format!(
            "live module is not exact SM89: target={} cc={:?} kind={:?}",
            compiler.target.as_str(),
            route.device.compute_capability,
            artifact.module_kind
        ));
    }
    if SM89_HALF_AUTO_CELLS.len() != 11 {
        return Err(format!(
            "SM89 half AUTO registry has {} cells instead of 11",
            SM89_HALF_AUTO_CELLS.len()
        ));
    }

    let exclusions = ctx.kernels.triad_sm89_half_exclusions();
    for spec in SM89_HALF_KERNEL_SPECS {
        if let Some((_, reason)) = exclusions.iter().find(|(symbol, _)| *symbol == spec.symbol) {
            println!(
                "{{\"schema\":\"MambaTriadSm89HalfAutoSymbolV1\",\"symbol\":\"{}\",\"excluded\":true,\"reason\":{:?}}}",
                spec.symbol, reason
            );
            continue;
        }
        let function = ctx
            .kernels
            .triad_sm89_half_function(spec.route, spec.dtype)
            .ok_or_else(|| format!("{} is neither bound nor excluded", spec.symbol))?;
        let registers = function
            .num_regs()
            .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?;
        let local_bytes = function
            .local_size_bytes()
            .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?;
        let static_shared_bytes = function
            .shared_size_bytes()
            .map_err(|error| format!("query {} shared memory: {error:?}", spec.symbol))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        println!(
            "{{\"schema\":\"MambaTriadSm89HalfAutoSymbolV1\",\"symbol\":\"{}\",\"excluded\":false,\"registers\":{},\"local_bytes\":{},\"static_shared_bytes\":{},\"dynamic_shared_bytes\":{},\"threads\":{},\"active_ctas\":{}}}",
            spec.symbol,
            registers,
            local_bytes,
            static_shared_bytes,
            spec.dynamic_shared_bytes,
            spec.threads,
            occupancy,
        );
    }

    let requests = SM89_HALF_AUTO_CELLS
        .iter()
        .map(|&(op, dtype, dims, _)| {
            PhysicalQualificationRequest::contiguous(
                op,
                dims,
                PhysicalQualificationRoute::HalfPolicy {
                    dtype,
                    tensor_cores: true,
                    half_policy: HalfTriadPolicy::TiledParityV1,
                },
            )
        })
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &requests)?;

    for (&(op, dtype, dims, route), request) in
        SM89_HALF_AUTO_CELLS.iter().zip(requests.iter().copied())
    {
        let spec = SM89_HALF_KERNEL_SPECS
            .iter()
            .find(|spec| spec.route == route && spec.dtype == dtype)
            .ok_or_else(|| format!("missing spec for {op:?}/{dtype:?}/{dims:?}/{route:?}"))?;
        let expected_columns =
            if op == mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn {
                dims.2
            } else {
                dims.1
            };
        let expected_grid = u32::try_from(dims.0.div_ceil(spec.tile.0 as usize))
            .ok()
            .and_then(|rows| {
                u32::try_from(expected_columns.div_ceil(spec.tile.1 as usize))
                    .ok()
                    .and_then(|columns| rows.checked_mul(columns))
            })
            .ok_or_else(|| "SM89 half qualification grid overflow".to_string())?;

        let qualified = qualify_physical_launch(&ctx, request)?;
        let evidence = qualified.evidence();
        if evidence.nodes().len() != 1
            || evidence.single_launch_symbol() != Some(spec.symbol)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm89Half)
            || evidence.uniform_execution_dtype()
                != Some(match dtype {
                    mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::Bf16 => PolicyDtype::Bf16,
                    mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::F16 => PolicyDtype::F16,
                    mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::F32 => unreachable!(),
                })
            || evidence.single_launch_tile() != Some(spec.tile)
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "actual AUTO identity mismatch for {op:?}/{dtype:?}/{dims:?}: {:?}",
                evidence.nodes()
            ));
        }
        let node = evidence.nodes()[0];
        if node.launch.grid_dim != (expected_grid, 1, 1)
            || node.launch.block_dim != (spec.threads, 1, 1)
            || node.launch.shared_mem_bytes != spec.dynamic_shared_bytes
        {
            return Err(format!(
                "actual AUTO launch mismatch for {}: {:?}",
                spec.symbol, node.launch
            ));
        }
        qualified.measure_graph_window_ms(&ctx, 1)?;
        let guards = qualified.validate_red_zones(&ctx)?;
        if guards.allocation_count() < 3 || guards.element_count() == 0 {
            return Err(format!("{} did not validate its guards", spec.symbol));
        }
        println!(
            "{{\"schema\":\"MambaTriadSm89HalfActualAutoV1\",\"op\":\"{:?}\",\"dtype\":\"{:?}\",\"dims\":[{},{},{}],\"symbol\":\"{}\",\"module\":\"TriadSm89Half\",\"grid\":[{},1,1],\"block\":[{},1,1],\"dynamic_shared_bytes\":{},\"eager_graph_equal\":true,\"guard_allocations\":{},\"guard_elements\":{}}}",
            op,
            dtype,
            dims.0,
            dims.1,
            dims.2,
            spec.symbol,
            expected_grid,
            spec.threads,
            spec.dynamic_shared_bytes,
            guards.allocation_count(),
            guards.element_count(),
        );
    }

    println!(
        "{{\"schema\":\"MambaTriadSm89HalfActualAutoIdentityV1\",\"target\":\"{}\",\"cc\":[{},{}],\"sms\":{},\"nvrtc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"driver_build_digest\":\"{}\",\"symbols\":{},\"auto_cells\":{},\"exclusions\":{}}}",
        compiler.target.as_str(),
        route.device.compute_capability.0,
        route.device.compute_capability.1,
        route.device.multiprocessor_count,
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        digest_hex(&artifact.compile_key),
        digest_hex(&artifact.artifact_digest),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
        digest_hex(&route.device.driver.build_digest),
        SM89_HALF_KERNEL_SPECS.len(),
        SM89_HALF_AUTO_CELLS.len(),
        exclusions.len(),
    );
    Ok(())
}
