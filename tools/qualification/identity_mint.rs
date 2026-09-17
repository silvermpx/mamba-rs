//! Print the compile identity of every module a frozen cohort can pin, for
//! the toolkit on PATH: the composed source, the NVRTC invocation, the PTX
//! it produces and the header closure it read, computed the way the module
//! loader computes them but without a device, so the compute_120 and
//! sm_90a modules mint on any box with the toolkit installed.
//!
//! A source change that moves no bits (a rename, a moved file) still moves
//! every one of these, and the dispatcher's cohorts decline silently when
//! they no longer match. Re-pin them in one pass: run this once per
//! toolkit on the reference tree and on the changed tree
//! (`cargo test --release --features cuda,qualification --test identity_mint
//! -- --ignored --nocapture`, logs kept per tree as `mint-<toolkit>.log`),
//! then `tools/identity_repin.py <old_logs> <new_logs> --apply`.
//! `MAMBA_RS_MINT_ONLY=Kind,Kind` restricts the table; `MAMBA_RS_MINT_DUMP=<dir>`
//! also writes each composed source.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::mint_module_identity;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, digest_hex};

/// One module the cohorts can pin: kind, target, state capacity, and the
/// device compute capability the Fixed composition is gated on.
struct MintedModule {
    kind: ModuleKind,
    target: &'static str,
    state_cap: usize,
    device_cc: Option<(i32, i32)>,
}

const fn module(
    kind: ModuleKind,
    target: &'static str,
    state_cap: usize,
    device_cc: Option<(i32, i32)>,
) -> MintedModule {
    MintedModule {
        kind,
        target,
        state_cap,
        device_cc,
    }
}

#[test]
#[ignore = "prints the identities of the toolkit on PATH; run on demand"]
fn print_module_identities() {
    let table = [
        module(ModuleKind::Fixed, "sm_80", 16, Some((8, 0))),
        module(ModuleKind::Fixed, "sm_89", 16, Some((8, 9))),
        module(ModuleKind::Fixed, "sm_89", 32, Some((8, 9))),
        module(ModuleKind::Fixed, "sm_89", 64, Some((8, 9))),
        module(ModuleKind::Fixed, "compute_120", 16, Some((12, 0))),
        module(ModuleKind::Fixed, "compute_120", 32, Some((12, 0))),
        module(ModuleKind::Fixed, "compute_120", 64, Some((12, 0))),
        module(ModuleKind::TriadScalar, "sm_80", 16, None),
        module(ModuleKind::TriadScalar, "sm_89", 16, None),
        module(ModuleKind::TriadScalar, "compute_120", 16, None),
        module(ModuleKind::TriadSm80, "sm_80", 16, None),
        module(ModuleKind::TriadSm80, "sm_89", 16, None),
        module(ModuleKind::TriadSm80, "compute_120", 16, None),
        module(ModuleKind::TriadSm89Half, "sm_89", 16, None),
        module(ModuleKind::TriadSm89ExactF32, "sm_89", 16, None),
        module(ModuleKind::TriadSm89ExactF32D128, "sm_89", 16, None),
        module(ModuleKind::TriadSm89Tf32Joint, "sm_89", 16, None),
        module(ModuleKind::TriadSm89Finalist, "sm_89", 16, None),
        module(ModuleKind::TriadSm90a, "sm_90a", 16, None),
        module(ModuleKind::TriadSm100, "compute_100a", 16, None),
        module(ModuleKind::TriadSm120, "compute_120", 16, None),
    ];
    let only = std::env::var("MAMBA_RS_MINT_ONLY").ok();
    for entry in &table {
        let (kind, arch, cap, cc) = (entry.kind, entry.target, entry.state_cap, entry.device_cc);
        if let Some(only) = &only
            && !only.split(',').any(|name| name == format!("{kind:?}"))
        {
            continue;
        }
        match mint_module_identity(kind, arch, cap, cc) {
            Ok(id) => println!(
                "IDENTITY {{\"module\":\"{:?}\",\"target\":\"{}\",\"cap\":{},\"device_cc\":{},\"nvrtc\":[{},{}],\"source\":\"{}\",\"compile_key\":\"{}\",\"invocation\":\"{}\",\"artifact\":\"{}\",\"header\":\"{}\",\"library\":\"{}\"}}",
                id.module_kind,
                id.target,
                id.state_cap,
                id.device_cc
                    .map_or("null".to_string(), |(major, minor)| format!(
                        "[{major},{minor}]"
                    )),
                id.nvrtc_version.0,
                id.nvrtc_version.1,
                digest_hex(&id.source_digest),
                id.compile_key
                    .map_or("null".to_string(), |key| digest_hex(&key)),
                digest_hex(&id.invocation_digest),
                digest_hex(&id.artifact_digest),
                digest_hex(&id.header_manifest_digest),
                digest_hex(&id.nvrtc_library_domain),
            ),
            Err(error) => eprintln!("SKIP {kind:?} {arch} cap{cap}: {error}"),
        }
    }
}
