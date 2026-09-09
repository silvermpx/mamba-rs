#![cfg(feature = "cuda")]

//! Live compile/ABI/resource qualification for the sealed Ada exact-F32 module.
//! This deliberately does not launch or select any kernel; Batch B1 leaves AUTO unchanged.

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::SM89_EXACT_F32_KERNEL_SPECS;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, digest_hex};

#[test]
#[ignore = "requires an exclusive SM89 GPU and explicit CUDA 12.8/13.0/13.2 selection"]
fn sm89_exact_f32_module_abi_and_resources_qualify() -> Result<(), String> {
    let device = GpuDevice::new(0)?;
    let ctx = mamba_rs::mamba_ssm::gpu::context::GpuCtx::new(&device)?;
    let route = ctx.gemm_route();
    let compiler = ctx
        .kernels
        .triad_sm89_exact_f32_compiler_identity()
        .ok_or_else(|| {
            format!(
                "TriadSm89ExactF32 did not bind: {:?}",
                ctx.kernels.triad_sm89_exact_f32_rejection()
            )
        })?;
    let artifact = ctx
        .kernels
        .triad_sm89_exact_f32_artifact_identity()
        .ok_or_else(|| "TriadSm89ExactF32 has no artifact identity".to_string())?;

    if route.device.compute_capability != (8, 9)
        || compiler.target.as_str() != "sm_89"
        || artifact.module_kind != ModuleKind::TriadSm89ExactF32
        || ctx.kernels.artifact_set_identity().sm89_exact_f32 != Some(artifact)
    {
        return Err(format!(
            "live exact-F32 module is not the canonical SM89 binding: cc={:?} target={} kind={:?}",
            route.device.compute_capability,
            compiler.target.as_str(),
            artifact.module_kind,
        ));
    }

    let exclusions = ctx.kernels.triad_sm89_exact_f32_exclusions();
    if !exclusions.is_empty() {
        return Err(format!(
            "TriadSm89ExactF32 has resource/ABI exclusions: {exclusions:?}"
        ));
    }

    for spec in SM89_EXACT_F32_KERNEL_SPECS {
        let function = ctx
            .kernels
            .triad_sm89_exact_f32_function(spec.symbol)
            .ok_or_else(|| format!("{} is neither bound nor excluded", spec.symbol))?;
        let registers = u32::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} returned negative register count", spec.symbol))?;
        let local_bytes = u32::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} returned negative local memory", spec.symbol))?;
        let static_shared_bytes = u32::try_from(
            function
                .shared_size_bytes()
                .map_err(|error| format!("query {} shared memory: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} returned negative shared memory", spec.symbol))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.block.0,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;

        if registers > spec.register_cap
            || local_bytes != 0
            || static_shared_bytes != spec.static_shared_bytes
            || max_threads < i32::try_from(spec.block.0).unwrap_or(i32::MAX)
            || occupancy < spec.occupancy_gate
        {
            return Err(format!(
                "{} misses its sealed resource gate: registers={registers}/{} local={local_bytes} static_shared={static_shared_bytes}/{} max_threads={max_threads}/{} occupancy={occupancy}/{}",
                spec.symbol,
                spec.register_cap,
                spec.static_shared_bytes,
                spec.block.0,
                spec.occupancy_gate,
            ));
        }
        println!(
            "{{\"schema\":\"MambaTriadSm89ExactF32ModuleSymbolV1\",\"symbol\":\"{}\",\"registers\":{},\"register_cap\":{},\"local_bytes\":{},\"static_shared_bytes\":{},\"dynamic_shared_bytes\":{},\"threads\":{},\"active_ctas\":{},\"occupancy_gate\":{}}}",
            spec.symbol,
            registers,
            spec.register_cap,
            local_bytes,
            static_shared_bytes,
            spec.dynamic_shared_bytes,
            spec.block.0,
            occupancy,
            spec.occupancy_gate,
        );
    }

    println!(
        "{{\"schema\":\"MambaTriadSm89ExactF32ModuleIdentityV1\",\"target\":\"{}\",\"cc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"driver_build_digest\":\"{}\",\"symbols\":{}}}",
        compiler.target.as_str(),
        route.device.compute_capability.0,
        route.device.compute_capability.1,
        digest_hex(&artifact.compile_key),
        digest_hex(&artifact.artifact_digest),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
        digest_hex(&route.device.driver.build_digest),
        SM89_EXACT_F32_KERNEL_SPECS.len(),
    );
    Ok(())
}
