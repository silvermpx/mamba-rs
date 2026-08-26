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

fn sm90a_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm90a.cu"),
    ])
}

fn sm100_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm100.cu"),
    ])
}

fn sm120_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm120.cu"),
    ])
}

fn sm100_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["m128n64", "m128n128"] {
            for stages in ["s2", "s3", "s4"] {
                for schedule in ["c4", "p8"] {
                    for dtype in ["bf16", "f16"] {
                        symbols.push(format!(
                            "sgemm_bi_{op}_sm100_tcgen_{tile}_bk64_{stages}_{schedule}_{dtype}"
                        ));
                    }
                }
            }
        }
    }
    symbols
}

fn sm120_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["64x64", "128x64", "64x128", "128x128"] {
            for bk in ["bk32", "bk64"] {
                for stages in ["s2", "s3"] {
                    for dtype in ["bf16", "f16"] {
                        symbols.push(format!(
                            "sgemm_bi_{op}_sm120_tma_{tile}_{bk}_{stages}_{dtype}"
                        ));
                    }
                }
            }
        }
    }
    symbols
}

fn nvrtc_version() -> (i32, i32) {
    let mut major = 0;
    let mut minor = 0;
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(
        result,
        cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS,
        "query NVRTC version"
    );
    (major, minor)
}

fn ptxas() -> std::path::PathBuf {
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(path) = std::env::var_os(variable) {
            let candidate = std::path::PathBuf::from(path).join("bin/ptxas");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    let standard = std::path::PathBuf::from("/usr/local/cuda/bin/ptxas");
    if standard.is_file() {
        standard
    } else {
        "ptxas".into()
    }
}

fn assemble_sm100(ptx: &str, target: &str, checker: bool) -> std::process::Output {
    let directory = tempfile::tempdir().expect("SM100 ptxas tempdir");
    let input = directory.path().join("triad-sm100.ptx");
    let output = directory.path().join("triad-sm100.cubin");
    std::fs::write(&input, ptx).expect("write SM100 PTX");
    let mut command = std::process::Command::new(ptxas());
    command.arg(format!("-arch={target}"));
    if checker {
        command.arg("-g-tmem-access-check");
    }
    command
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run ptxas for SM100")
}

fn assemble_sm120(ptx: &str, target: &str) -> std::process::Output {
    let directory = tempfile::tempdir().expect("SM120 ptxas tempdir");
    let input = directory.path().join("triad-sm120.ptx");
    let output = directory.path().join("triad-sm120.cubin");
    std::fs::write(&input, ptx).expect("write SM120 PTX");
    std::process::Command::new(ptxas())
        .arg(format!("-arch={target}"))
        .arg("-v")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run ptxas for SM120")
}

fn ptx_entry<'a>(ptx: &'a str, symbol: &str) -> &'a str {
    let marker = format!(".entry {symbol}(");
    let start = ptx
        .find(&marker)
        .unwrap_or_else(|| panic!("missing PTX entry {symbol}"));
    let tail = &ptx[start..];
    let end = tail[marker.len()..]
        .find("\n.visible .entry ")
        .map(|offset| marker.len() + offset)
        .unwrap_or(tail.len());
    &tail[..end]
}

fn ptx_parameters<'a>(entry: &'a str, symbol: &str) -> &'a str {
    let marker = format!(".entry {symbol}(");
    entry
        .split_once(&marker)
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(parameters, _)| parameters))
        .unwrap_or_else(|| panic!("parameter list for {symbol}"))
}

fn has_exact_maxntid(entry: &str, threads: u32) -> bool {
    let canonical = format!(".maxntid {threads}");
    let explicit = format!(".maxntid {threads}, 1, 1");
    entry
        .lines()
        .map(str::trim)
        .any(|line| line == canonical || line == explicit)
}

fn metric_before(line: &str, marker: &str) -> Option<u64> {
    let prefix = line.split_once(marker)?.0;
    prefix.split_ascii_whitespace().last()?.parse().ok()
}

fn contains_opcode_prefix(source: &str, prefix: &str) -> bool {
    source.match_indices(prefix).any(|(offset, _)| {
        source[..offset].chars().next_back().is_none_or(|previous| {
            !previous.is_ascii_alphanumeric() && !matches!(previous, '_' | '.')
        })
    })
}

fn assert_zero_local_resources(report: &str, target: &str) {
    for line in report.lines() {
        for marker in [
            " bytes stack frame",
            " bytes spill stores",
            " bytes spill loads",
        ] {
            if let Some(value) = metric_before(line, marker) {
                assert_eq!(value, 0, "{target} uses local resources: {line}");
            }
        }
    }
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

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_90a"),
        options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(sm90a_blob(), opts)
        .expect("TriadSm90a kernel module must compile for exact sm_90a");
    let ptx = std::str::from_utf8(image.as_bytes().expect("SM90a PTX image"))
        .expect("SM90a PTX must be UTF-8");
    assert!(ptx.lines().any(|line| line.trim() == ".target sm_90a"));
    for symbol in [
        "sgemm_bi_nn_sm90a_wgmma_wg1_bf16",
        "sgemm_bi_nn_sm90a_wgmma_wg1_f16",
        "sgemm_bi_tn_sm90a_wgmma_wg1_bf16",
        "sgemm_bi_tn_sm90a_wgmma_wg1_f16",
        "sgemm_bi_nt_sm90a_wgmma_wg1_bf16",
        "sgemm_bi_nt_sm90a_wgmma_wg1_f16",
        "sgemm_bi_nn_sm90a_wgmma_wg2_bf16",
        "sgemm_bi_nn_sm90a_wgmma_wg2_f16",
        "sgemm_bi_tn_sm90a_wgmma_wg2_bf16",
        "sgemm_bi_tn_sm90a_wgmma_wg2_f16",
        "sgemm_bi_nt_sm90a_wgmma_wg2_bf16",
        "sgemm_bi_nt_sm90a_wgmma_wg2_f16",
    ] {
        assert_eq!(ptx.matches(&format!(".entry {symbol}(")).count(), 1);
    }
    for instruction in [
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx",
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
        "setmaxnreg.dec.sync.aligned.u32",
        "setmaxnreg.inc.sync.aligned.u32",
    ] {
        assert!(
            ptx.contains(instruction),
            "SM90a PTX is missing {instruction}"
        );
    }
    assert!(
        !ptx.split_ascii_whitespace()
            .any(|token| token.starts_with("atom.") || token.starts_with("red."))
    );
}

#[test]
fn compiles_for_sm100a() {
    compile_for("sm_100a");
}

#[test]
fn compiles_for_sm103() {
    compile_for("sm_103");
}

#[test]
fn compiles_exact_sm100_family_triad_modules() {
    for (requested, emitted) in [
        ("compute_100f", "sm_100f"),
        ("compute_100a", "sm_100a"),
        ("compute_103f", "sm_103f"),
        ("compute_103a", "sm_103a"),
    ] {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(requested),
            options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let image = cudarc::nvrtc::compile_ptx_with_opts(sm100_blob(), opts)
            .unwrap_or_else(|error| panic!("TriadSm100 must compile for {requested}: {error}"));
        let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
            image.as_bytes().expect("SM100 PTX image"),
        )
        .expect("SM100 PTX must be canonical UTF-8");
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {emitted}")),
            "{requested} emitted the wrong PTX target"
        );
        for symbol in sm100_symbols() {
            assert_eq!(
                ptx.matches(&format!(".entry {symbol}(")).count(),
                1,
                "{requested} entry census for {symbol}"
            );
            let marker = format!(".entry {symbol}(");
            let parameters = ptx
                .split_once(&marker)
                .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
                .unwrap_or_else(|| panic!("{requested} parameter list for {symbol}"));
            assert_eq!(
                parameters.matches(".param").count(),
                5,
                "{requested} ABI parameter count for {symbol}"
            );
            assert!(
                parameters.contains(&format!(".param .align 4 .b8 {symbol}_param_4[40]")),
                "{requested} 40-byte parameter bundle for {symbol}: {parameters}"
            );
        }
        for instruction in [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "mbarrier.arrive.expect_tx",
            "tcgen05.alloc.cta_group::1",
            "tcgen05.relinquish_alloc_permit.cta_group::1",
            "tcgen05.dealloc.cta_group::1",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one",
            "tcgen05.fence::before_thread_sync",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
            "tcgen05.st.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::st.sync.aligned",
        ] {
            assert!(
                ptx.contains(instruction),
                "{requested} is missing {instruction}"
            );
        }
        assert!(!ptx.contains("cta_group::2"));
        assert!(!ptx.contains("tcgen05.ld.red"));
        assert!(!ptx.contains("wgmma."));
        assert!(!ptx.contains("multicast"));
        assert!(!ptx.split_ascii_whitespace().any(|token| {
            token.starts_with("atom.")
                || token.starts_with("red.")
                || token.starts_with("atom::")
                || token.starts_with("red::")
        }));
        let assembly = assemble_sm100(&ptx, emitted, true);
        assert!(
            assembly.status.success(),
            "ptxas -g-tmem-access-check failed for {emitted}: {}",
            String::from_utf8_lossy(&assembly.stderr)
        );
    }
}

#[test]
fn ordinary_sm100_targets_fail_offline_tcgen_assembly() {
    for (requested, emitted) in [("compute_100", "sm_100"), ("compute_103", "sm_103")] {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(requested),
            options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let image = cudarc::nvrtc::compile_ptx_with_opts(sm100_blob(), opts)
            .unwrap_or_else(|error| panic!("NVRTC ordinary-target probe failed: {error}"));
        let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
            image.as_bytes().expect("ordinary SM100 PTX image"),
        )
        .expect("ordinary SM100 PTX must be canonical UTF-8");
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {emitted}")),
            "{requested} emitted an unexpected target"
        );
        let assembly = assemble_sm100(&ptx, emitted, false);
        assert!(
            !assembly.status.success(),
            "ordinary target {emitted} illegally admitted TCGEN05"
        );
    }
}

#[test]
fn compiles_for_sm120() {
    compile_for("sm_120");
}

#[test]
fn compiles_for_sm121_when_the_active_nvrtc_supports_it() {
    if nvrtc_version() >= (12, 9) {
        compile_for("sm_121");
    }
}

#[test]
fn compiles_generic_sm120_triad_modules_with_exact_ptx_contract() {
    let version = nvrtc_version();
    let mut targets = vec![("compute_120", "sm_120")];
    if version >= (12, 9) {
        targets.push(("compute_121", "sm_121"));
    }
    for (requested, emitted) in targets {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(requested),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                "-DNDEBUG".to_string(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let image = cudarc::nvrtc::compile_ptx_with_opts(sm120_blob(), opts)
            .unwrap_or_else(|error| panic!("TriadSm120 must compile for {requested}: {error}"));
        let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
            image.as_bytes().expect("SM120 PTX image"),
        )
        .expect("SM120 PTX must be canonical UTF-8");
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {emitted}")),
            "{requested} emitted the wrong PTX target"
        );
        assert!(!ptx.contains(".target sm_120a"));
        assert!(!ptx.contains(".target sm_120f"));
        assert!(!ptx.contains(".target sm_121a"));
        assert!(!ptx.contains(".target sm_121f"));

        let map_alignment = if version.0 >= 13 { 128 } else { 64 };
        for symbol in sm120_symbols() {
            let entry = ptx_entry(&ptx, &symbol);
            let parameters = ptx_parameters(entry, &symbol);
            assert_eq!(
                parameters.matches(".param").count(),
                5,
                "{requested} ABI parameter count for {symbol}"
            );
            assert_eq!(
                parameters
                    .matches(&format!(".param .align {map_alignment} .b8"))
                    .count(),
                2,
                "{requested} tensor-map ABI for {symbol}: {parameters}"
            );
            assert_eq!(
                parameters.matches("[128]").count(),
                2,
                "{requested} tensor-map sizes for {symbol}: {parameters}"
            );
            assert!(
                parameters.contains(&format!(".param .align 4 .b8 {symbol}_param_4[40]")),
                "{requested} 40-byte parameter bundle for {symbol}: {parameters}"
            );

            let threads = if symbol.contains("_64x64_") {
                128
            } else if symbol.contains("_128x128_") {
                512
            } else {
                256
            };
            assert!(
                has_exact_maxntid(entry, threads),
                "{requested} launch bounds for {symbol}"
            );

            for required in [
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mbarrier.init.shared::cta.b64",
                "fence.mbarrier_init.release.cluster",
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "mbarrier.arrive.release.cta.shared::cta.b64",
            ] {
                assert!(
                    entry.contains(required),
                    "{requested}/{symbol} is missing {required}"
                );
            }

            let dtype = if symbol.ends_with("_bf16") {
                "bf16"
            } else {
                "f16"
            };
            let mma = format!("mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32");
            let (mma_count, a_loads, b_loads) = if symbol.contains("_bk32_") {
                (16, 4, 8)
            } else {
                (32, 8, 16)
            };
            assert_eq!(
                entry.matches(&mma).count(),
                mma_count,
                "{requested} MMA census for {symbol}"
            );
            let a_instruction = if symbol.contains("_tn_") {
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            } else {
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            };
            let b_instruction = if symbol.contains("_nt_") {
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            } else {
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
            };
            assert_eq!(
                entry.matches(a_instruction).count(),
                a_loads,
                "{requested} A ldmatrix census for {symbol}"
            );
            assert_eq!(
                entry.matches(b_instruction).count(),
                b_loads,
                "{requested} B ldmatrix census for {symbol}"
            );
            assert!(!entry.contains("call.uni"), "device call in {symbol}");
        }

        for forbidden in [
            "tcgen05",
            "tmem",
            "wgmma.",
            "setmaxnreg",
            "multicast",
            "shared::cluster",
            "cta_group::2",
            "multimem",
            "mapa",
            "clusterlaunchcontrol",
            "griddepcontrol",
            ".callprototype",
        ] {
            assert!(!ptx.contains(forbidden), "{requested} contains {forbidden}");
        }
        for forbidden_opcode in ["atom.", "red.", "redux."] {
            assert!(
                !contains_opcode_prefix(&ptx, forbidden_opcode),
                "{requested} contains opcode {forbidden_opcode}"
            );
        }

        let assembly = assemble_sm120(&ptx, emitted);
        assert!(
            assembly.status.success(),
            "ptxas failed for {emitted}: {}",
            String::from_utf8_lossy(&assembly.stderr)
        );
        let report = String::from_utf8_lossy(&assembly.stderr);
        assert_zero_local_resources(&report, emitted);
        for symbol in sm120_symbols() {
            assert!(
                report.contains(&symbol),
                "{emitted} ptxas resource report omitted {symbol}"
            );
        }
    }
}
