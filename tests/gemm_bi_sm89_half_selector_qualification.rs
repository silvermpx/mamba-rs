#![cfg(feature = "cuda")]

//! Phase-1 live checkpoint for the isolated Ada half-Triad module. AUTO is
//! deliberately empty until this test has printed and root has reviewed one
//! literal identity for every supported toolkit.

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{SM89_HALF_AUTO_CELLS, SM89_HALF_KERNEL_SPECS};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, digest_hex};

#[test]
#[ignore = "requires an exclusive RTX 6000 Ada and explicit toolkit selection"]
fn sm89_half_phase_one_live_identity() -> Result<(), String> {
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
    if !SM89_HALF_AUTO_CELLS.is_empty() {
        return Err("phase 1 must not activate an AUTO half cell".into());
    }

    let exclusions = ctx.kernels.triad_sm89_half_exclusions();
    for spec in SM89_HALF_KERNEL_SPECS {
        if let Some((_, reason)) = exclusions.iter().find(|(symbol, _)| *symbol == spec.symbol) {
            println!(
                "{{\"schema\":\"MambaTriadSm89HalfPhase1SymbolV1\",\"symbol\":\"{}\",\"excluded\":true,\"reason\":{:?}}}",
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
            "{{\"schema\":\"MambaTriadSm89HalfPhase1SymbolV1\",\"symbol\":\"{}\",\"excluded\":false,\"registers\":{},\"local_bytes\":{},\"static_shared_bytes\":{},\"dynamic_shared_bytes\":{},\"threads\":{},\"active_ctas\":{}}}",
            spec.symbol,
            registers,
            local_bytes,
            static_shared_bytes,
            spec.dynamic_shared_bytes,
            spec.threads,
            occupancy,
        );
    }

    println!(
        "{{\"schema\":\"MambaTriadSm89HalfPhase1IdentityV1\",\"target\":\"{}\",\"cc\":[{},{}],\"sms\":{},\"nvrtc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"driver_build_digest\":\"{}\",\"symbols\":{},\"auto_cells\":{},\"exclusions\":{}}}",
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
