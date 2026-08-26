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

fn m1_blob() -> String {
    [
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
        include_str!("../kernels/gemm_bi_fixed/ffma.cu"),
        include_str!("../kernels/gemm_bi_fixed/wmma_legacy.cu"),
        include_str!("../kernels/gemm_bi_fixed/matvec.cu"),
        include_str!("../kernels/gemm_bi_fixed/mma16.cu"),
        include_str!("../kernels/gemm_bi_fixed/tcw64.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm90_wgmma.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm100_tcgen05.cu"),
        include_str!("../kernels/gemm_bi_triad.cu"),
    ]
    .iter()
    .map(|s| {
        s.lines()
            .filter(|l| !l.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .collect::<Vec<_>>()
    .join("\n")
}

fn compile_for(arch: &'static str) -> String {
    let src = m1_blob();
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        options: vec![
            "--fmad=true".to_string(),
            "-DNDEBUG".to_string(),
            "-DSGB_GROUP_M=16".to_string(),
            "-DMAMBA_RS_STATE_CAP=256".to_string(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
        Ok(ptx) => ptx.to_src(),
        Err(e) => panic!("kernel blob does not compile for {arch}: {e}"),
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
fn compiles_for_sm103a() {
    compile_for("sm_103a");
}

#[test]
fn compiles_for_sm120() {
    compile_for("sm_120");
}

/// NVRTC acceptance alone does not prove a target-legal module: the
/// front end will happily emit PTX whose opcodes the assembler then
/// rejects for the target. When a ptxas binary is reachable, assemble
/// the emitted PTX for the architecture-specific targets and fail on
/// any diagnostic; without one (a box with no toolkit) the step skips
/// loudly and the NVRTC gates still run.
fn ptxas_path() -> Option<std::path::PathBuf> {
    // The runtime NVRTC and the assembler need not come from the same
    // toolkit (dynamic loading picks the newest NVRTC on the loader
    // path), so take the NEWEST ptxas the box has - an older assembler
    // cannot read a newer front end's PTX version at all.
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for e in entries.flatten() {
            let p = e.path().join("bin/ptxas");
            if p.exists() {
                candidates.push(p);
            }
        }
    }
    if let Ok(home) = std::env::var("CUDA_HOME") {
        let p = std::path::Path::new(&home).join("bin/ptxas");
        if p.exists() {
            candidates.push(p);
        }
    }
    if let Ok(out) = std::process::Command::new("which").arg("ptxas").output() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            candidates.push(s.into());
        }
    }
    let release = |p: &std::path::PathBuf| -> (u32, u32) {
        let Ok(out) = std::process::Command::new(p).arg("--version").output() else {
            return (0, 0);
        };
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let Some(pos) = text.find("release ") else {
            return (0, 0);
        };
        let tail = &text[pos + "release ".len()..];
        let num: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let mut it = num.split('.');
        let maj = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let min = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        (maj, min)
    };
    candidates.into_iter().max_by_key(release)
}

#[test]
fn family_targets_assemble_under_ptxas() {
    let Some(ptxas) = ptxas_path() else {
        println!("no ptxas on this box; assembly gate skipped (NVRTC gates still ran)");
        return;
    };
    for arch in ["sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_120"] {
        let ptx = compile_for(arch);
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("arch_gate_{arch}.ptx"));
        std::fs::write(&src_path, &ptx).expect("write ptx");
        let out = std::process::Command::new(&ptxas)
            .arg(format!("-arch={arch}"))
            .arg(&src_path)
            .arg("-o")
            .arg(dir.join(format!("arch_gate_{arch}.cubin")))
            .output()
            .expect("run ptxas");
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        if !out.status.success() && stderr.contains("Unsupported .version") {
            println!(
                "ptxas on this box predates the loaded NVRTC's PTX version; \
                 assembly gate skipped ({})",
                stderr.lines().next().unwrap_or("")
            );
            return;
        }
        assert!(
            out.status.success(),
            "ptxas rejected the {arch} module:\n{stderr}"
        );
    }
}

/// The tcgen05 rung must exist exactly where its hardware exists: the
/// datacenter-Blackwell family targets emit the opcode family, and every
/// other architecture's PTX stays provably free of it (an sm_100 device
/// without the family feature, or any other arch, lands on the portable
/// mma.sync ladder and must never meet a stray tcgen05 instruction).
#[test]
fn tcgen05_only_in_blackwell_family_ptx() {
    for arch in ["sm_100a", "sm_103a"] {
        let ptx = compile_for(arch);
        assert!(
            ptx.contains("tcgen05.mma") && ptx.contains("tcgen05.alloc"),
            "{arch} PTX lost the tcgen05 rung"
        );
    }
    for arch in ["sm_80", "sm_89", "sm_90a", "sm_120"] {
        let ptx = compile_for(arch);
        assert!(
            !ptx.contains("tcgen05"),
            "{arch} PTX must not contain tcgen05 instructions"
        );
    }
}
