//! Cross-architecture compile gates: the full kernel source must compile
//! for every supported target, on any build box, with no GPU of that
//! generation present. NVRTC emits PTX for the requested architecture
//! regardless of the local device, so a kernel that breaks on Hopper or
//! Blackwell is caught here instead of on rented hardware.
//!
//! Compilation is necessary, not sufficient: launch behavior and output
//! bits are qualified per architecture on real hardware before that
//! architecture's dispatch cells are enabled.
#![cfg(feature = "cuda")]

fn compose(fragments: &[&str]) -> String {
    fragments
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn fixed_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/mamba_ssm.cu"),
        include_str!("../kernels/mamba_ssm_parallel.cu"),
        include_str!("../kernels/conv1d.cu"),
        include_str!("../kernels/activations.cu"),
        include_str!("../kernels/norms.cu"),
        include_str!("../kernels/elementwise.cu"),
        include_str!("../kernels/loss_scaler.cu"),
        include_str!("../kernels/grad_clip.cu"),
        include_str!("../kernels/adamw.cu"),
        include_str!("../kernels/gemm_bi_fixed/common.cuh"),
        include_str!("../kernels/gemm_bi_fixed/ffma.cuh"),
        include_str!("../kernels/gemm_bi_fixed/wmma_legacy.cuh"),
        include_str!("../kernels/gemm_bi_fixed/matvec.cuh"),
        include_str!("../kernels/gemm_bi_fixed/mma16.cuh"),
        include_str!("../kernels/gemm_bi_fixed/sm90_wgmma.cuh"),
    ])
}

fn scalar_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/scalar.cu"),
    ])
}

fn sm80_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm80.cu"),
    ])
}

fn compile_for(arch: &'static str) {
    let group_m = if matches!(arch, "sm_80" | "sm_86" | "sm_87") {
        8
    } else {
        16
    };
    for (kind, source) in [
        ("Fixed", fixed_blob()),
        ("TriadScalar", scalar_blob()),
        ("TriadSm80", sm80_blob()),
    ] {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                "-DNDEBUG".to_string(),
                format!("-DSGB_GROUP_M={group_m}"),
                "-DMAMBA_RS_STATE_CAP=256".to_string(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        if let Err(error) = cudarc::nvrtc::compile_ptx_with_opts(source, opts) {
            panic!("{kind} kernel module does not compile for {arch}: {error}");
        }
    }
}

#[test]
fn compiles_for_sm80() {
    compile_for("sm_80");
}

#[test]
fn compiles_for_sm89() {
    compile_for("sm_89");
}

#[test]
fn compiles_for_sm90a() {
    compile_for("sm_90a");
}

#[test]
fn compiles_for_sm100a() {
    compile_for("sm_100a");
}

#[test]
fn compiles_for_sm120() {
    compile_for("sm_120");
}
