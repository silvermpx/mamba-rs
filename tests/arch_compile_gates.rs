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

const SCALAR_NT_M2N16_SYMBOL: &str = "gemm_bi_nt_m2n16_bk64_splitk32_v1";
const SCALAR_NT_M2N16_FRAGMENT: &str = "kernels/gemm_bi_triad/scalar_nt_m2n16.cu";
const SCALAR_TN_M16N16_SYMBOL: &str = "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1";
const SCALAR_TN_M16N16_FRAGMENT: &str = "kernels/gemm_bi_triad/scalar_tn_m16n16.cu";
const SCALAR_NN_M32N64_SPLITK32_SYMBOL: &str = "gemm_bi_nn_splitk32_m32n64_exact_v1";
const SCALAR_NN_M32N64_SPLITK32_FRAGMENT: &str = "kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu";

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

#[test]
fn fixed_tf32_source_is_forward_only_and_self_contained() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gemm_bi_fixed/tf32.cu");
    let source = std::fs::read_to_string(path).expect("standalone Fixed TF32 source");
    assert!(source.contains("gemm_bi_nn_tf32_v1_m128n64_bk32_s2"));
    assert!(source.contains("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"));
    for forbidden in [
        "gemm_bi_tn_",
        "gemm_bi_nt_",
        "backward",
        "atomic",
        "split_k",
    ] {
        assert!(
            !source.to_ascii_lowercase().contains(forbidden),
            "Fixed TF32 source contains training-only token {forbidden}"
        );
    }
    let sm120_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels/gemm_bi_fixed/tf32_sm120.cu"
    );
    let sm120 = std::fs::read_to_string(sm120_path).expect("standalone Fixed SM120 TF32 source");
    assert!(sm120.contains("gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2"));
    for forbidden in [
        "gemm_bi_tn_",
        "gemm_bi_nt_",
        "backward",
        "atomic",
        "split_k",
    ] {
        assert!(
            !sm120.to_ascii_lowercase().contains(forbidden),
            "Fixed SM120 TF32 source contains training-only token {forbidden}"
        );
    }
    let half_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels/gemm_bi_fixed/sm120_tma.cu"
    );
    let half = std::fs::read_to_string(half_path).expect("standalone Fixed SM120 half source");
    assert!(half.contains("GBF_SM120_HALF_DEFINE_PAIR(64, 64, 64, 2)"));
    assert!(half.contains("GBF_SM120_HALF_DEFINE_PAIR(64, 128, 64, 2)"));
    assert!(half.contains("GBF_SM120_HALF_DEFINE_PAIR(128, 64, 32, 3)"));
    assert!(half.contains("GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 2)"));
    assert!(half.contains("GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 3)"));
    assert!(half.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
    assert!(half.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    for forbidden in ["gemm_bi_tn_", "gemm_bi_nt_", "backward", "split_k"] {
        assert!(
            !half.to_ascii_lowercase().contains(forbidden),
            "Fixed SM120 half source contains training-only token {forbidden}"
        );
    }
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
        include_str!("../kernels/gemm_bi_fixed/ffma.cu"),
        include_str!("../kernels/gemm_bi_fixed/tf32.cu"),
        include_str!("../kernels/gemm_bi_fixed/tf32_sm120.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm120_tma.cu"),
        include_str!("../kernels/gemm_bi_fixed/wmma_legacy.cu"),
        include_str!("../kernels/gemm_bi_fixed/matvec.cu"),
        include_str!("../kernels/gemm_bi_fixed/mma16.cu"),
        include_str!("../kernels/gemm_bi_fixed/tcw64.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm90_wgmma.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm100_tcgen05.cu"),
    ])
}

fn fixed_blob_for(arch: &str) -> String {
    let mut source = fixed_blob();
    if arch == "sm_89" {
        source.push('\n');
        source.push_str(include_str!(
            "../kernels/gemm_bi_fixed/sm89_half_pipeline.cu"
        ));
        source.push('\n');
        source.push_str(include_str!(
            "../kernels/gemm_bi_fixed/sm89_f32_n64_copyplan.cu"
        ));
    }
    source
}

fn scalar_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/scalar.cu"),
        include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
        include_str!("../kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu"),
        include_str!("../kernels/gemm_bi_triad/scalar_nt_m2n16.cu"),
        include_str!("../kernels/gemm_bi_triad/scalar_tn_m16n16.cu"),
    ])
}

#[test]
fn scalar_nn_m32n64_splitk32_has_one_exact_source_owner_abi_and_resource_contract() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fragment_path = manifest.join(SCALAR_NN_M32N64_SPLITK32_FRAGMENT);
    let fragment = std::fs::read_to_string(&fragment_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fragment_path.display()));
    assert_eq!(
        fragment.matches(SCALAR_NN_M32N64_SPLITK32_SYMBOL).count(),
        1,
        "the production fragment must own exactly one M32N64 Split-K entry"
    );
    let compact: String = fragment
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(compact.contains(&format!(
        "extern\"C\"__global____launch_bounds__(NN_SPLITK_EXACT_THREADS,4)void{SCALAR_NN_M32N64_SPLITK32_SYMBOL}(float*__restrict__partial,constfloat*__restrict__A,constfloat*__restrict__B,intM,intN,intK_CHUNKS,intlda)"
    )));
    for required in [
        "#define NN_SPLITK_EXACT_M 128",
        "#define NN_SPLITK_EXACT_K 8192",
        "#define NN_SPLITK_EXACT_N 128",
        "#define NN_SPLITK_EXACT_THREADS 128",
        "#define NN_SPLITK_EXACT_WARP_SIZE 32",
        "#define NN_SPLITK_EXACT_A_PAD 4",
        "#define NN_SPLITK_EXACT_B_PAD 4",
        "TOTAL_BLOCKS == 2048",
        "sizeof(As) + sizeof(Bs) == 13312",
        "for (int dot = 0; dot < NN_SPLITK_EXACT_BK; ++dot)",
        "results[index] = __fmaf_rn(",
    ] {
        assert!(
            fragment.contains(required),
            "M32N64 source omitted {required}"
        );
    }
    for forbidden in [
        "GEMM_BI_SCALAR_SMEM_A_PAD",
        "GEMM_BI_SCALAR_SMEM_B_PAD",
        "GEMM_BI_SCALAR_WARP_SIZE",
    ] {
        assert!(
            !fragment.contains(forbidden),
            "M32N64 source depends on undefined production macro {forbidden}"
        );
    }
    assert!(!fragment.to_ascii_lowercase().contains("atomic"));

    let scalar_directory = manifest.join("kernels/gemm_bi_triad");
    let mut owners = Vec::new();
    for entry in std::fs::read_dir(&scalar_directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", scalar_directory.display()))
    {
        let path = entry.expect("scalar source directory entry").path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("cu")
            || path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains(SCALAR_NN_M32N64_SPLITK32_SYMBOL) {
            owners.push(
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .expect("UTF-8 scalar source name")
                    .to_owned(),
            );
        }
    }
    assert_eq!(owners, ["scalar_nn_splitk_m32n64.cu"]);

    let modules_path = manifest.join("src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    let modules = std::fs::read_to_string(&modules_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", modules_path.display()));
    let scalar_fragments = modules
        .split_once("const SCALAR_SOURCE_FRAGMENTS: &[SourceFragment] = &[")
        .and_then(|(_, tail)| tail.split_once("const SM80_SOURCE_FRAGMENTS"))
        .map(|(body, _)| body)
        .expect("production scalar source-fragment inventory");
    assert_eq!(
        scalar_fragments
            .matches(SCALAR_NN_M32N64_SPLITK32_FRAGMENT)
            .count(),
        2
    );
    let scalar_symbols = modules
        .split_once("pub(super) const SCALAR_SYMBOLS: &[&str] = &[")
        .and_then(|(_, tail)| tail.split_once("pub(super) const SM80_SYMBOLS"))
        .map(|(body, _)| body)
        .expect("production scalar symbol inventory");
    assert_eq!(
        scalar_symbols
            .matches(SCALAR_NN_M32N64_SPLITK32_SYMBOL)
            .count(),
        1
    );
    assert_eq!(
        scalar_blob()
            .matches(SCALAR_NN_M32N64_SPLITK32_SYMBOL)
            .count(),
        1
    );
}

#[test]
fn scalar_nt_m2n16_has_one_production_source_and_scalar_owner() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fragment_path = manifest.join(SCALAR_NT_M2N16_FRAGMENT);
    let fragment = std::fs::read_to_string(&fragment_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fragment_path.display()));
    assert_eq!(
        fragment.matches(SCALAR_NT_M2N16_SYMBOL).count(),
        1,
        "the production fragment must own exactly one exported entry"
    );

    let mut owners = Vec::new();
    let scalar_directory = manifest.join("kernels/gemm_bi_triad");
    for entry in std::fs::read_dir(&scalar_directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", scalar_directory.display()))
    {
        let path = entry.expect("scalar source directory entry").path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("cu") {
            continue;
        }
        if path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains(SCALAR_NT_M2N16_SYMBOL) {
            owners.push(
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .expect("UTF-8 scalar source name")
                    .to_owned(),
            );
        }
    }
    assert_eq!(owners, ["scalar_nt_m2n16.cu"]);

    let modules_path = manifest.join("src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    let modules = std::fs::read_to_string(&modules_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", modules_path.display()));
    let scalar_fragments = modules
        .split_once("const SCALAR_SOURCE_FRAGMENTS: &[SourceFragment] = &[")
        .and_then(|(_, tail)| tail.split_once("const SM80_SOURCE_FRAGMENTS"))
        .map(|(body, _)| body)
        .expect("production scalar source-fragment inventory");
    assert_eq!(
        scalar_fragments.matches(SCALAR_NT_M2N16_FRAGMENT).count(),
        2,
        "the scalar fragment must have one logical name and one include"
    );
    let scalar_symbols = modules
        .split_once("pub(super) const SCALAR_SYMBOLS: &[&str] = &[")
        .and_then(|(_, tail)| tail.split_once("pub(super) const SM80_SYMBOLS"))
        .map(|(body, _)| body)
        .expect("production scalar symbol inventory");
    assert_eq!(
        scalar_symbols.matches(SCALAR_NT_M2N16_SYMBOL).count(),
        1,
        "the specialized entry must be owned exactly once by TriadScalar"
    );
}

#[test]
fn scalar_tn_m16n16_has_one_exact_source_owner_abi_and_resource_contract() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fragment_path = manifest.join(SCALAR_TN_M16N16_FRAGMENT);
    let fragment = std::fs::read_to_string(&fragment_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fragment_path.display()));
    assert_eq!(
        fragment.matches(SCALAR_TN_M16N16_SYMBOL).count(),
        1,
        "the production fragment must own exactly one M16N16 entry"
    );
    let compact: String = fragment
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(compact.contains(&format!(
        "extern\"C\"__global____launch_bounds__(64,4)void{SCALAR_TN_M16N16_SYMBOL}(float*output,constfloat*a,constfloat*b,floatalpha,intm,intk,intn)"
    )));
    for required in [
        "SHARED_BYTES == 4096",
        "__fmaf_rn(",
        "__dadd_rn(",
        "__dmul_rn(",
        "__fadd_rn(",
    ] {
        assert!(
            fragment.contains(required),
            "M16N16 source omitted {required}"
        );
    }
    assert!(!fragment.to_ascii_lowercase().contains("atomic"));

    let scalar_directory = manifest.join("kernels/gemm_bi_triad");
    let mut owners = Vec::new();
    for entry in std::fs::read_dir(&scalar_directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", scalar_directory.display()))
    {
        let path = entry.expect("scalar source directory entry").path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("cu")
            || path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains(SCALAR_TN_M16N16_SYMBOL) {
            owners.push(
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .expect("UTF-8 scalar source name")
                    .to_owned(),
            );
        }
    }
    assert_eq!(owners, ["scalar_tn_m16n16.cu"]);

    let modules_path = manifest.join("src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    let modules = std::fs::read_to_string(&modules_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", modules_path.display()));
    let scalar_fragments = modules
        .split_once("const SCALAR_SOURCE_FRAGMENTS: &[SourceFragment] = &[")
        .and_then(|(_, tail)| tail.split_once("const SM80_SOURCE_FRAGMENTS"))
        .map(|(body, _)| body)
        .expect("production scalar source-fragment inventory");
    assert_eq!(
        scalar_fragments.matches(SCALAR_TN_M16N16_FRAGMENT).count(),
        2
    );
    let scalar_symbols = modules
        .split_once("pub(super) const SCALAR_SYMBOLS: &[&str] = &[")
        .and_then(|(_, tail)| tail.split_once("pub(super) const SM80_SYMBOLS"))
        .map(|(body, _)| body)
        .expect("production scalar symbol inventory");
    assert_eq!(scalar_symbols.matches(SCALAR_TN_M16N16_SYMBOL).count(), 1);
}

#[test]
fn scalar_nt_m2n16_source_freezes_abi_shape_and_resources() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SCALAR_NT_M2N16_FRAGMENT);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let signature = format!(
        "extern\"C\"__global____launch_bounds__(64,4)void{SCALAR_NT_M2N16_SYMBOL}(float*output,constfloat*a,constfloat*b,floatalpha,intm,intn,intk_out)"
    );
    assert!(
        compact.contains(&signature),
        "production kernel ABI drifted"
    );
    assert!(source.contains("static_assert(GemmBiNtM2N16Kernel::SHARED_BYTES == 17984"));
    assert!(source.contains("if (m != 512 || n != 2048 || k_out != 16) return;"));
    assert!(source.contains("cp.async.ca.shared.global"));
    assert!(source.contains("__fmaf_rn("));
    assert!(source.contains("__fadd_rn("));
    assert!(!source.to_ascii_lowercase().contains("atomic"));
    assert_eq!(source.matches("extern \"C\" __global__").count(), 1);

    let run = source
        .split_once("static __device__ __forceinline__ void run(")
        .map(|(_, tail)| tail)
        .expect("M2N16 device entry");
    let guard = run
        .find("if (m != 512 || n != 2048 || k_out != 16) return;")
        .expect("exact-shape guard");
    let shared = run
        .find("extern __shared__")
        .expect("dynamic shared declaration");
    assert!(
        guard < shared,
        "exact-shape guard must precede shared-memory work"
    );

    assert_eq!(
        scalar_blob().matches(SCALAR_NT_M2N16_SYMBOL).count(),
        1,
        "the external TriadScalar compile fixture must include the production entry"
    );
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

/// The portable module as every non-CC-12 target composes it: sm80.cu plus
/// the extension fragments (the tc64 TN stream-K twin and the wide TF32 tile).
fn sm80_streamk_blob() -> String {
    compose(&[
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm80.cu"),
        include_str!("../kernels/gemm_bi_triad/sm80_streamk.cu"),
        include_str!("../kernels/gemm_bi_triad/sm80_tf32_wide.cu"),
        include_str!("../kernels/gemm_bi_triad/sm80_tn_splitk.cu"),
    ])
}

/// The exports the extension fragments add to the portable module.
const SM80_EXTENSION_SYMBOLS: [&str; 7] = [
    "gemm_bi_tn_tc64_streamk_bf16",
    "gemm_bi_tn_tc64_streamk_f16",
    "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s2",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s3",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
];

/// The existing extension target set, limited by the active toolkit's
/// explicit SM110 support floor. Compilation errors remain fatal.
fn sm80_extension_targets(version: (i32, i32)) -> Vec<&'static str> {
    let mut targets = vec!["sm_80", "sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a"];
    if version >= (13, 2) {
        targets.push("sm_110a");
    }
    targets
}

#[test]
fn sm80_extension_targets_respect_nvrtc_version() {
    for version in [(12, 8), (12, 9), (13, 0), (13, 1)] {
        assert_eq!(
            sm80_extension_targets(version),
            ["sm_80", "sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a"],
            "unsupported sm_110a must not be requested with NVRTC {version:?}"
        );
    }
    for version in [(13, 2), (13, 3), (14, 0)] {
        assert_eq!(
            sm80_extension_targets(version),
            [
                "sm_80", "sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a", "sm_110a"
            ],
            "supported extension target disappeared with NVRTC {version:?}"
        );
    }
}

/// Every supported target composes each extension export exactly once;
/// the sm_80 image additionally passes the strict assembler resource census.
#[test]
fn sm80_extension_kernels_compile_for_every_portable_target() {
    for arch in sm80_extension_targets(nvrtc_version()) {
        let ptx = compile_module_for("TriadSm80+extensions", sm80_streamk_blob(), arch);
        for symbol in SM80_EXTENSION_SYMBOLS {
            assert_eq!(
                ptx.matches(&format!(".entry {symbol}(")).count(),
                1,
                "{arch} must export {symbol} exactly once"
            );
        }
    }
    // The assembler pass runs on the sm_80 image: the helper assembles for
    // sm_80, and the register file the spill census measures is the same on
    // every sm80-family part.
    let ptx = compile_module_for("TriadSm80+extensions", sm80_streamk_blob(), "sm_80");
    let (report, _) = assemble_and_disassemble_sm80_scalar(&ptx);
    let mut current: Option<&str> = None;
    let mut seen = std::collections::BTreeSet::new();
    for line in report.lines() {
        if let Some(rest) = line.split_once("Compiling entry function '") {
            current = SM80_EXTENSION_SYMBOLS
                .iter()
                .copied()
                .find(|symbol| rest.1.starts_with(symbol));
        }
        if let Some(symbol) = current
            && (line.contains(" bytes spill ") || line.contains(" bytes stack frame"))
        {
            for marker in [
                " bytes stack frame",
                " bytes spill stores",
                " bytes spill loads",
            ] {
                if let Some(head) = line.split(marker).next()
                    && let Some(value) = head
                        .split_whitespace()
                        .last()
                        .and_then(|v| v.parse::<u64>().ok())
                    && line.contains(marker)
                {
                    assert_eq!(value, 0, "{symbol} uses local resources: {line}");
                }
            }
            seen.insert(symbol);
        }
    }
    assert_eq!(
        seen.len(),
        SM80_EXTENSION_SYMBOLS.len(),
        "ptxas reported resources for {seen:?}"
    );
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
        include_str!("../kernels/gemm_bi_triad/sm120_exact.cu"),
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
                            "gemm_bi_{op}_sm100_tcgen_{tile}_bk64_{stages}_{schedule}_{dtype}"
                        ));
                    }
                }
            }
        }
    }
    symbols
}

fn sm90a_symbols() -> Vec<String> {
    [
        "gemm_bi_nn_sm90a_wgmma_wg1_bf16",
        "gemm_bi_nn_sm90a_wgmma_wg1_f16",
        "gemm_bi_tn_sm90a_wgmma_wg1_bf16",
        "gemm_bi_tn_sm90a_wgmma_wg1_f16",
        "gemm_bi_nt_sm90a_wgmma_wg1_bf16",
        "gemm_bi_nt_sm90a_wgmma_wg1_f16",
        "gemm_bi_nn_sm90a_wgmma_wg2_bf16",
        "gemm_bi_nn_sm90a_wgmma_wg2_f16",
        "gemm_bi_tn_sm90a_wgmma_wg2_bf16",
        "gemm_bi_tn_sm90a_wgmma_wg2_f16",
        "gemm_bi_nt_sm90a_wgmma_wg2_bf16",
        "gemm_bi_nt_sm90a_wgmma_wg2_f16",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn sm80_tf32_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for route in [
            "m128n64_bk32_s2",
            "m128n64_bk32_s3",
            "m64n64_bk32_s2",
            "m64n64_bk32_s3",
            "m16n32_bk32_s4",
            "m16n16_bk32_s4",
        ] {
            symbols.push(format!("gemm_bi_{op}_sm80_mma_tf32_v1_{route}"));
        }
    }
    symbols
}

fn sm90a_tf32_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for warpgroup in [1, 2] {
            symbols.push(format!(
                "gemm_bi_{op}_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg{warpgroup}"
            ));
        }
    }
    symbols
}

#[test]
fn sm90a_tf32_tma_calls_bind_every_prepared_subview_origin() {
    let source = include_str!("../kernels/gemm_bi_triad/sm90a.cu");
    let producer = source
        .split_once("void sm90a_tf32_produce_stage(")
        .and_then(|(_, tail)| {
            tail.split_once("void sm90a_tf32_wgmma_k8(")
                .map(|(body, _)| body)
        })
        .expect("SM90a TF32 stage producer");
    let compact: String = producer
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert_eq!(
        compact.matches("sm90a_tf32_tma_copy(").count(),
        6,
        "SM90a TF32 must issue one source-level TMA load for each NN/TN/NT operand"
    );
    assert_eq!(
        compact.matches("params.a_x,params.a_y").count(),
        3,
        "every SM90a TF32 A load must bind both prepared A origins"
    );
    assert_eq!(
        compact.matches("params.b_x,params.b_y").count(),
        3,
        "every SM90a TF32 B load must bind both prepared B origins"
    );
}

#[test]
fn tf32_positive_ceil_divisions_use_subtract_before_addition() {
    for (name, source, column_divisor, column_sites, reduction_sites) in [
        (
            "SM90a",
            include_str!("../kernels/gemm_bi_triad/sm90a.cu"),
            "128",
            2,
            1,
        ),
        (
            "SM100",
            include_str!("../kernels/gemm_bi_triad/sm100.cu"),
            "Columns",
            2,
            1,
        ),
        (
            "SM120",
            include_str!("../kernels/gemm_bi_triad/sm120.cu"),
            "N",
            6,
            3,
        ),
    ] {
        let compact: String = source
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let safe_columns = format!("1+(columns-1)/{column_divisor}");
        assert_eq!(
            compact.matches(&safe_columns).count(),
            column_sites,
            "{name} TF32 output tiling must avoid overflowing positive columns"
        );
        assert_eq!(
            compact.matches("1+(reduction-1)/32").count(),
            reduction_sites,
            "{name} TF32 reduction tiling must avoid overflowing a positive reduction"
        );
    }
}

fn sm100_tf32_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for columns in [64, 128] {
            for stages in [2, 3, 4] {
                for schedule in ["c4", "p8"] {
                    symbols.push(format!(
                        "gemm_bi_{op}_sm100_tcgen_tf32_v1_m128n{columns}_bk32_s{stages}_{schedule}"
                    ));
                }
            }
        }
    }
    symbols
}

fn sm120_tf32_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["m128n64", "m64n128"] {
            for stages in [2, 3] {
                symbols.push(format!(
                    "gemm_bi_{op}_sm120_tma_mma_tf32_v1_{tile}_bk32_s{stages}"
                ));
            }
        }
        symbols.push(format!("gemm_bi_{op}_sm120_tma_mma_tf32_v1_m64n64_bk32_s2"));
    }
    symbols.push("gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair".to_string());
    symbols.push("gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk".to_string());
    symbols.push("gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2".to_string());
    for (op, tiles) in [
        ("nn", &["m128n64", "m64n128", "m64n64"][..]),
        ("tn", &["m128n64", "m64n128", "m64n64"][..]),
        ("nt", &["m128n64", "m64n128", "m64n64"][..]),
    ] {
        for tile in tiles {
            symbols.push(format!("gemm_bi_{op}_sm120_tma_fma_v1_{tile}_bk16_s2"));
        }
    }
    for tile in ["m128n64", "m64n128", "m64n64"] {
        symbols.push(format!("gemm_bi_nt_sm120_tma_fma_v1_{tile}_bk16_s2_kvec"));
    }
    symbols
}

#[test]
fn sm120_tf32_rect_wide_source_contract() {
    let source = include_str!("../kernels/gemm_bi_triad/sm120.cu");
    let symbol = "gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2";
    assert!(source.contains(symbol));
    assert!(source.contains("sm120_tf32_rect_wide_entry(output, a_map, b_map, bias, params)"));
    assert!(source.contains("sm120_tf32_rect_wide_kernel<80, 32, 32, 64, 2>"));
    assert!(!source.contains("gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32"));
}

fn sm120_symbols() -> Vec<String> {
    let mut symbols = Vec::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["64x64", "128x64", "64x128", "128x128"] {
            for bk in ["bk32", "bk64"] {
                for stages in ["s2", "s3"] {
                    for dtype in ["bf16", "f16"] {
                        symbols.push(format!(
                            "gemm_bi_{op}_sm120_tma_{tile}_{bk}_{stages}_{dtype}"
                        ));
                    }
                }
            }
        }
    }
    // The stream-K TN bodies over the training batch's tile.
    for dtype in ["bf16", "f16"] {
        symbols.push(format!(
            "gemm_bi_tn_sm120_tma_64x64_bk64_s3_streamk_{dtype}"
        ));
    }
    symbols
}

#[derive(Clone, Copy)]
struct CompileGatePtxToken<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct CompileGatePtxEntry {
    symbol: String,
    text: String,
    body: String,
}

#[derive(Debug)]
struct CompileGatePtxFunction {
    symbol: String,
    body: Option<String>,
}

#[derive(Debug)]
struct CompileGatePtx {
    target: String,
    entries: Vec<CompileGatePtxEntry>,
    functions: Vec<CompileGatePtxFunction>,
}

impl CompileGatePtx {
    fn entry(&self, symbol: &str) -> &CompileGatePtxEntry {
        self.entries
            .iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap_or_else(|| panic!("missing parsed PTX entry {symbol}"))
    }
}

fn compile_gate_has_exact_token(ptx: &str, expected: &str) -> bool {
    compile_gate_ptx_tokens(ptx)
        .into_iter()
        .filter(|token| !token.text.starts_with('"'))
        .any(|token| token.text == expected)
}

fn assert_compile_gate_entry_tokens(label: &str, entry: &CompileGatePtxEntry, required: &[&str]) {
    for &instruction in required {
        assert!(
            compile_gate_has_exact_token(&entry.body, instruction),
            "{label}/{} is missing {instruction}",
            entry.symbol
        );
    }
}

fn assert_compile_gate_entry_excludes(
    label: &str,
    entry: &CompileGatePtxEntry,
    forbidden: &[&str],
) {
    for &instruction in forbidden {
        assert!(
            !compile_gate_has_exact_token(&entry.body, instruction),
            "{label}/{} contains incompatible core {instruction}",
            entry.symbol
        );
    }
}

fn strip_compile_gate_ptx_comments(ptx: &str) -> Result<String, String> {
    let mut stripped = ptx.as_bytes().to_vec();
    let mut cursor = 0;
    while cursor < stripped.len() {
        if stripped[cursor] == b'"' {
            cursor += 1;
            let mut closed = false;
            while cursor < stripped.len() {
                match stripped[cursor] {
                    b'\\' => cursor = cursor.saturating_add(2),
                    b'"' => {
                        cursor += 1;
                        closed = true;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
            if !closed {
                return Err("unterminated PTX string literal".into());
            }
            continue;
        }
        if stripped.get(cursor..cursor + 2) == Some(b"//") {
            while cursor < stripped.len() && stripped[cursor] != b'\n' {
                stripped[cursor] = b' ';
                cursor += 1;
            }
            continue;
        }
        if stripped.get(cursor..cursor + 2) == Some(b"/*") {
            stripped[cursor] = b' ';
            stripped[cursor + 1] = b' ';
            cursor += 2;
            let mut closed = false;
            while cursor < stripped.len() {
                if stripped.get(cursor..cursor + 2) == Some(b"*/") {
                    stripped[cursor] = b' ';
                    stripped[cursor + 1] = b' ';
                    cursor += 2;
                    closed = true;
                    break;
                }
                if stripped[cursor] != b'\n' {
                    stripped[cursor] = b' ';
                }
                cursor += 1;
            }
            if !closed {
                return Err("unterminated PTX block comment".into());
            }
            continue;
        }
        cursor += 1;
    }
    String::from_utf8(stripped).map_err(|_| "comment-stripped PTX is not UTF-8".to_string())
}

fn compile_gate_ptx_tokens(ptx: &str) -> Vec<CompileGatePtxToken<'_>> {
    let bytes = ptx.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        let start = cursor;
        if bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor = cursor.saturating_add(2),
                    b'"' => {
                        cursor += 1;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
        } else if matches!(
            bytes[cursor],
            b'(' | b')' | b'{' | b'}' | b'[' | b']' | b',' | b';'
        ) {
            cursor += 1;
        } else {
            cursor += 1;
            while cursor < bytes.len()
                && !bytes[cursor].is_ascii_whitespace()
                && !matches!(
                    bytes[cursor],
                    b'(' | b')' | b'{' | b'}' | b'[' | b']' | b',' | b';' | b'"'
                )
            {
                cursor += 1;
            }
        }
        tokens.push(CompileGatePtxToken {
            text: &ptx[start..cursor],
            start,
            end: cursor,
        });
    }
    tokens
}

fn is_compile_gate_ptx_symbol(token: &str) -> bool {
    let mut bytes = token.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
}

fn matching_compile_gate_ptx_token(
    tokens: &[CompileGatePtxToken<'_>],
    start: usize,
    open: &str,
    close: &str,
) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        if token.text == open {
            depth += 1;
        } else if token.text == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn compile_gate_entry_signature_end(
    tokens: &[CompileGatePtxToken<'_>],
    entry: usize,
    symbol: &str,
) -> Result<usize, String> {
    let after_symbol = entry + 2;
    match tokens.get(after_symbol).map(|token| token.text) {
        Some("(") => matching_compile_gate_ptx_token(tokens, after_symbol, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX entry {symbol} has an unclosed parameter list")),
        Some("{") => Ok(after_symbol),
        Some(directive) if directive.starts_with('.') => Ok(after_symbol),
        Some(_) => Err(format!("PTX entry {symbol} has a malformed signature")),
        None => Err(format!("PTX entry {symbol} has no body")),
    }
}

fn compile_gate_entry_body_open(
    tokens: &[CompileGatePtxToken<'_>],
    mut cursor: usize,
    symbol: &str,
) -> Result<usize, String> {
    loop {
        let token = tokens
            .get(cursor)
            .ok_or_else(|| format!("PTX entry {symbol} has no body"))?;
        if token.text == "{" {
            return Ok(cursor);
        }
        if matches!(
            token.text,
            ";" | "}" | ".entry" | ".extern" | ".func" | ".target"
        ) || !token.text.starts_with('.')
        {
            return Err(format!("PTX entry {symbol} has no body"));
        }

        if token.text == ".pragma" {
            tokens
                .get(cursor + 1)
                .filter(|value| value.text.starts_with('"') && value.text.ends_with('"'))
                .ok_or_else(|| format!("PTX entry {symbol} has a malformed pragma"))?;
            tokens
                .get(cursor + 2)
                .filter(|terminator| terminator.text == ";")
                .ok_or_else(|| format!("PTX entry {symbol} has a malformed pragma"))?;
            cursor += 3;
            continue;
        }

        cursor += 1;
        while let Some(operand) = tokens.get(cursor) {
            if operand.text == "{" || operand.text.starts_with('.') {
                break;
            }
            if operand.text == "}" {
                return Err(format!("PTX entry {symbol} has no body"));
            }
            cursor += 1;
            if operand.text == ";" {
                break;
            }
        }
    }
}

fn reject_nested_compile_gate_directives(
    tokens: &[CompileGatePtxToken<'_>],
    body_open: usize,
    body_close: usize,
    owner: &str,
) -> Result<(), String> {
    if let Some(directive) = tokens[body_open + 1..body_close]
        .iter()
        .find(|token| matches!(token.text, ".entry" | ".func" | ".target"))
    {
        return Err(format!(
            "PTX {owner} contains nested module directive {}",
            directive.text
        ));
    }
    Ok(())
}

fn parse_compile_gate_function(
    ptx: &str,
    tokens: &[CompileGatePtxToken<'_>],
    function: usize,
) -> Result<(CompileGatePtxFunction, usize), String> {
    let mut cursor = function + 1;
    if tokens.get(cursor).is_some_and(|token| token.text == "(") {
        cursor = matching_compile_gate_ptx_token(tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| "PTX function has an unclosed return parameter list".to_string())?;
    }
    let symbol = tokens
        .get(cursor)
        .filter(|token| is_compile_gate_ptx_symbol(token.text))
        .ok_or_else(|| "PTX function has no valid symbol".to_string())?;
    cursor += 1;
    if tokens.get(cursor).is_some_and(|token| token.text == "(") {
        cursor = matching_compile_gate_ptx_token(tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX function {} has unclosed parameters", symbol.text))?;
    }

    let mut modifier = function;
    let mut is_extern = false;
    while modifier > 0 && tokens[modifier - 1].text.starts_with('.') {
        modifier -= 1;
        is_extern |= tokens[modifier].text == ".extern";
    }
    while let Some(token) = tokens.get(cursor) {
        match token.text {
            "{" => {
                if is_extern {
                    return Err(format!("extern PTX function {} has a body", symbol.text));
                }
                let body_close = matching_compile_gate_ptx_token(tokens, cursor, "{", "}")
                    .ok_or_else(|| format!("PTX function {} has an unclosed body", symbol.text))?;
                reject_nested_compile_gate_directives(tokens, cursor, body_close, "function")?;
                return Ok((
                    CompileGatePtxFunction {
                        symbol: symbol.text.to_owned(),
                        body: Some(ptx[tokens[cursor].end..tokens[body_close].start].to_owned()),
                    },
                    body_close + 1,
                ));
            }
            ";" => {
                return Ok((
                    CompileGatePtxFunction {
                        symbol: symbol.text.to_owned(),
                        body: None,
                    },
                    cursor + 1,
                ));
            }
            "}" | ".entry" | ".func" | ".target" => {
                return Err(format!("PTX function {} has no valid body", symbol.text));
            }
            _ => cursor += 1,
        }
    }
    Err(format!(
        "PTX function {} has no body or declaration terminator",
        symbol.text
    ))
}

fn parse_compile_gate_ptx(ptx: &str) -> Result<CompileGatePtx, String> {
    let stripped = strip_compile_gate_ptx_comments(ptx)?;
    let tokens = compile_gate_ptx_tokens(&stripped);
    let mut target = None;
    let mut entries = Vec::new();
    let mut functions = Vec::new();
    let mut cursor = 0;
    while cursor < tokens.len() {
        match tokens[cursor].text {
            ".target" => {
                let value = tokens
                    .get(cursor + 1)
                    .filter(|token| is_compile_gate_ptx_symbol(token.text))
                    .ok_or_else(|| "malformed PTX target directive".to_string())?;
                if target.replace(value.text.to_owned()).is_some() {
                    return Err("duplicate PTX target directives".into());
                }
                cursor += 2;
                continue;
            }
            ".func" => {
                let (function, next) = parse_compile_gate_function(&stripped, &tokens, cursor)?;
                functions.push(function);
                cursor = next;
                continue;
            }
            "{" => {
                let close = matching_compile_gate_ptx_token(&tokens, cursor, "{", "}")
                    .ok_or_else(|| "PTX contains an unclosed module-scope brace".to_string())?;
                reject_nested_compile_gate_directives(&tokens, cursor, close, "module scope")?;
                cursor = close + 1;
                continue;
            }
            "}" => return Err("PTX contains an unmatched module-scope closing brace".into()),
            ".entry" => {}
            _ => {
                cursor += 1;
                continue;
            }
        }
        let mut modifier = cursor;
        while modifier > 0 && tokens[modifier - 1].text.starts_with('.') {
            modifier -= 1;
            if tokens[modifier].text == ".extern" {
                return Err("extern PTX entry directive".into());
            }
        }
        let symbol = tokens
            .get(cursor + 1)
            .filter(|token| is_compile_gate_ptx_symbol(token.text))
            .ok_or_else(|| "PTX entry has no valid symbol".to_string())?;
        let signature_end = compile_gate_entry_signature_end(&tokens, cursor, symbol.text)?;
        let body_open = compile_gate_entry_body_open(&tokens, signature_end, symbol.text)?;
        let body_open_token = tokens
            .get(body_open)
            .ok_or_else(|| format!("PTX entry {} has no body", symbol.text))?;
        let body_close = matching_compile_gate_ptx_token(&tokens, body_open, "{", "}")
            .ok_or_else(|| format!("PTX entry {} has an unclosed body", symbol.text))?;
        reject_nested_compile_gate_directives(
            &tokens,
            body_open,
            body_close,
            &format!("entry {}", symbol.text),
        )?;
        entries.push(CompileGatePtxEntry {
            symbol: symbol.text.to_owned(),
            text: stripped[tokens[cursor].start..tokens[body_close].end].to_owned(),
            body: stripped[body_open_token.end..tokens[body_close].start].to_owned(),
        });
        cursor = body_close + 1;
    }
    let target = target.ok_or_else(|| "missing PTX target directive".to_string())?;
    Ok(CompileGatePtx {
        target,
        entries,
        functions,
    })
}

fn compile_gate_function_ref<'a>(
    parsed: &'a CompileGatePtx,
    symbol: &str,
) -> Result<&'a CompileGatePtxFunction, String> {
    let mut matches = parsed
        .functions
        .iter()
        .filter(|function| function.symbol == symbol);
    let function = matches
        .next()
        .ok_or_else(|| format!("PTX is missing function {symbol}"))?;
    if matches.next().is_some() {
        return Err(format!("PTX contains ambiguous function {symbol}"));
    }
    Ok(function)
}

fn compile_gate_direct_call_targets(body: &str) -> Result<Vec<String>, String> {
    let tokens = compile_gate_ptx_tokens(body);
    let mut targets = Vec::new();
    for (call, token) in tokens.iter().enumerate() {
        if token.text != "call" && !token.text.starts_with("call.") {
            continue;
        }
        if token.text != "call.uni" {
            return Err(format!(
                "PTX contains unsupported call opcode {}",
                token.text
            ));
        }
        let mut cursor = call + 1;
        if tokens.get(cursor).is_some_and(|token| token.text == "(") {
            cursor = matching_compile_gate_ptx_token(&tokens, cursor, "(", ")")
                .map(|close| close + 1)
                .ok_or_else(|| "PTX call has unclosed return arguments".to_string())?;
            if !tokens.get(cursor).is_some_and(|token| token.text == ",") {
                return Err("PTX call has no target separator".into());
            }
            cursor += 1;
        }
        let target = tokens
            .get(cursor)
            .filter(|target| is_compile_gate_ptx_symbol(target.text))
            .ok_or_else(|| "PTX call has no direct target".to_string())?;
        cursor += 1;
        if !tokens.get(cursor).is_some_and(|token| token.text == ",") {
            return Err(format!(
                "PTX call to {} has no argument separator",
                target.text
            ));
        }
        cursor += 1;
        if !tokens.get(cursor).is_some_and(|token| token.text == "(") {
            return Err(format!("PTX call to {} has no argument list", target.text));
        }
        cursor = matching_compile_gate_ptx_token(&tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX call to {} has unclosed arguments", target.text))?;
        if !tokens.get(cursor).is_some_and(|token| token.text == ";") {
            return Err(format!("PTX call to {} has no terminator", target.text));
        }
        targets.push(target.text.to_owned());
    }
    Ok(targets)
}

fn validate_compile_gate_sm90a_wg2_producers(
    parsed: &CompileGatePtx,
    symbols: &[String],
) -> Result<(), String> {
    const PRODUCER: &[&str] = &[
        "setmaxnreg.dec.sync.aligned.u32",
        "bar.sync",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "ret",
    ];
    if symbols.len() != 12 {
        return Err(format!(
            "SM90a typed metadata has {} entries, expected 12",
            symbols.len()
        ));
    }

    let mut group_targets = Vec::new();
    for pair in symbols[6..].as_chunks::<2>().0 {
        let mut pair_target = None;
        for symbol in pair {
            let entry = parsed.entry(symbol);
            let calls = compile_gate_direct_call_targets(&entry.body)
                .map_err(|error| format!("SM90a/{symbol}: {error}"))?;
            if calls.len() != 1 {
                return Err(format!(
                    "SM90a/{symbol} has {} direct calls, expected one producer call",
                    calls.len()
                ));
            }
            let target = &calls[0];
            let function = compile_gate_function_ref(parsed, target)?;
            let body = function
                .body
                .as_deref()
                .ok_or_else(|| format!("SM90a producer {target} has no body"))?;
            for &instruction in PRODUCER {
                if !compile_gate_has_exact_token(body, instruction) {
                    return Err(format!("SM90a producer {target} is missing {instruction}"));
                }
            }
            if let Some(expected) = &pair_target
                && expected != target
            {
                return Err(format!(
                    "SM90a WG2 pair {pair:?} calls different producers {expected} and {target}"
                ));
            }
            pair_target = Some(target.clone());
        }
        group_targets.push(
            pair_target.ok_or_else(|| "SM90a WG2 metadata contains an empty pair".to_string())?,
        );
    }
    if group_targets
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != group_targets.len()
    {
        return Err("SM90a WG2 operation groups must call three distinct producers".into());
    }
    Ok(())
}

fn validate_exact_ptx_exports(
    label: &str,
    ptx: &str,
    expected_target: &str,
    expected: &[String],
    expected_count: usize,
) -> Result<CompileGatePtx, String> {
    let parsed = parse_compile_gate_ptx(ptx)
        .map_err(|error| format!("{label} PTX parse failed: {error}"))?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .collect();

    let mut expected_counts = std::collections::BTreeMap::new();
    for symbol in expected {
        *expected_counts.entry(symbol.as_str()).or_insert(0_usize) += 1;
    }
    let expected_duplicates: Vec<_> = expected_counts
        .iter()
        .filter_map(|(&symbol, &count)| (count > 1).then_some(symbol))
        .collect();
    let expected_set: std::collections::BTreeSet<_> = expected_counts.into_keys().collect();

    let mut actual_counts = std::collections::BTreeMap::new();
    for &symbol in &actual {
        *actual_counts.entry(symbol).or_insert(0_usize) += 1;
    }
    let actual_duplicates: Vec<_> = actual_counts
        .iter()
        .filter_map(|(&symbol, &count)| (count > 1).then_some(symbol))
        .collect();
    let actual_set: std::collections::BTreeSet<_> = actual_counts.into_keys().collect();
    let missing: Vec<_> = expected_set.difference(&actual_set).copied().collect();
    let foreign: Vec<_> = actual_set.difference(&expected_set).copied().collect();

    if expected.len() == expected_count
        && actual.len() == expected_count
        && parsed.target == expected_target
        && expected_duplicates.is_empty()
        && actual_duplicates.is_empty()
        && missing.is_empty()
        && foreign.is_empty()
    {
        return Ok(parsed);
    }
    Err(format!(
        "{label} PTX mismatch: expected_target={expected_target}; actual_target={}; expected_count={expected_count}; expected_entries={}; actual_entries={}; expected_duplicates={expected_duplicates:?}; actual_duplicates={actual_duplicates:?}; missing={missing:?}; foreign={foreign:?}",
        parsed.target,
        expected.len(),
        actual.len()
    ))
}

#[test]
fn architecture_export_census_accepts_comments_as_whitespace_and_ignores_spoofs() {
    let expected = vec!["only_export".to_string()];
    let ptx = ".version 9.0\n.target sm_90a\n.visible .entry\t/* gap */\nonly_export() {}\n// .entry line_spoof() {}\n/* .entry block_spoof() {} */\n";
    validate_exact_ptx_exports("architecture fixture", ptx, "sm_90a", &expected, 1).unwrap();
}

#[test]
fn architecture_export_census_rejects_comment_obfuscated_duplicates() {
    let expected = vec!["only_export".to_string()];
    let ptx =
        ".version 9.0\n.target sm_90a\n.entry only_export() {}\n.entry/* gap */only_export() {}\n";
    validate_exact_ptx_exports("architecture fixture", ptx, "sm_90a", &expected, 1)
        .expect_err("obfuscated duplicate export must fail");
}

#[test]
fn architecture_export_census_accepts_parameterless_and_entry_pragma_forms() {
    let expected = vec!["only_export".to_string()];
    for ptx in [
        ".version 9.0\n.target sm_90a\n.visible .func helper() { ret; }\n.entry only_export { ret; }\n",
        ".version 9.0\n.target sm_90a\n.entry only_export() .pragma \"nounroll\"; { ret; }\n.visible .func helper() { ret; }\n",
    ] {
        validate_exact_ptx_exports("architecture fixture", ptx, "sm_90a", &expected, 1)
            .expect("valid PTX entry form");
    }
}

#[test]
fn architecture_export_census_rejects_nested_module_directives_and_unbalanced_scopes() {
    let expected = vec!["only_export".to_string()];
    for ptx in [
        ".version 9.0\n.target sm_90a\n.visible .func helper() { .entry only_export() { ret; } }\n",
        ".version 9.0\n.visible .func helper() { .target sm_90a; ret; }\n.entry only_export() { ret; }\n",
        ".version 9.0\n.target sm_90a\n.visible .func helper() { ret;\n.entry only_export() { ret; }\n",
        ".version 9.0\n.target sm_90a\n.entry only_export() { ret; }\n}\n",
        ".version 9.0\n.target sm_90a\n.entry only_export() { .target sm_90a; ret; }\n",
        ".version 9.0\n.target sm_90a\n.visible .func outer() { .func nested() { ret; } }\n.entry only_export() { ret; }\n",
    ] {
        validate_exact_ptx_exports(
            "architecture nested-scope fixture",
            ptx,
            "sm_90a",
            &expected,
            1,
        )
        .expect_err("nested directives and unbalanced scopes must fail closed");
    }
}

#[test]
fn architecture_export_census_retains_only_the_brace_bounded_entry_body() {
    let expected = vec!["only_export".to_string()];
    let ptx = ".version 9.0\n.target sm_90a\n.entry only_export() { mov.b32 {%r1, %r2}, {%r3, %r4}; }\n.visible .func unused() { wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32; }\n";
    let parsed =
        validate_exact_ptx_exports("architecture body fixture", ptx, "sm_90a", &expected, 1)
            .unwrap();
    let body = &parsed.entry("only_export").body;
    assert!(compile_gate_has_exact_token(body, "mov.b32"));
    assert!(!compile_gate_has_exact_token(
        body,
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32"
    ));
}

fn architecture_sm90a_wg2_producer_function(target: &str) -> String {
    format!(
        ".visible .func {target}() {{\n    setmaxnreg.dec.sync.aligned.u32;\n    bar.sync;\n    mbarrier.try_wait.parity.acquire.cta.shared::cta.b64;\n    mbarrier.arrive.expect_tx.release.cta.shared::cta.b64;\n    cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;\n    ret;\n}}\n"
    )
}

fn architecture_sm90a_wg2_producer_fixture() -> String {
    let symbols = sm90a_symbols();
    let targets = ["producer_nn", "producer_tn", "producer_nt"];
    let mut ptx = ".version 9.0\n.target sm_90a\n".to_string();
    for (index, symbol) in symbols.iter().enumerate() {
        ptx.push_str(&format!(".entry {symbol}() {{\n"));
        if index >= 6 {
            ptx.push_str(&format!("    call.uni {}, ();\n", targets[(index - 6) / 2]));
        }
        ptx.push_str("    ret;\n}\n");
    }
    for target in targets {
        ptx.push_str(&architecture_sm90a_wg2_producer_function(target));
    }
    ptx
}

#[test]
fn architecture_sm90a_wg2_producer_calls_resolve_to_unique_complete_functions() {
    let ptx = architecture_sm90a_wg2_producer_fixture();
    let parsed = parse_compile_gate_ptx(&ptx).unwrap();
    assert_eq!(parsed.functions.len(), 3);
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols()).unwrap();

    let missing_call = ptx.replacen("    call.uni producer_nn, ();\n", "", 1);
    let parsed = parse_compile_gate_ptx(&missing_call).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("missing WG2 producer call must fail");

    let wrong_target = ptx.replacen(
        "    call.uni producer_nn, ();\n",
        "    call.uni producer_tn, ();\n",
        1,
    );
    let parsed = parse_compile_gate_ptx(&wrong_target).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("BF16 and F16 in one operation group must call the same producer");

    let producer_nn = architecture_sm90a_wg2_producer_function("producer_nn");
    let missing_helper = ptx.replacen(&producer_nn, "", 1);
    let parsed = parse_compile_gate_ptx(&missing_helper).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("missing called producer helper must fail");

    let duplicate_helper = format!("{ptx}{producer_nn}");
    let parsed = parse_compile_gate_ptx(&duplicate_helper).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("ambiguous called producer helper must fail");

    let stub_helper = ptx.replacen(&producer_nn, ".visible .func producer_nn() { ret; }\n", 1);
    let parsed = parse_compile_gate_ptx(&stub_helper).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("a stubbed called helper must not borrow protocol from unrelated helpers");

    let declaration_only = ptx.replacen(&producer_nn, ".extern .func producer_nn();\n", 1);
    let parsed = parse_compile_gate_ptx(&declaration_only).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("a declaration-only called producer must fail");

    let duplicate_call = ptx.replacen(
        "    call.uni producer_nn, ();\n",
        "    call.uni producer_nn, ();\n    call.uni producer_nn, ();\n",
        1,
    );
    let parsed = parse_compile_gate_ptx(&duplicate_call).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &sm90a_symbols())
        .expect_err("a WG2 entry with multiple direct calls must fail");
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

fn ptxas_path() -> Option<std::path::PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(path) = std::env::var_os(variable) {
            let candidate = std::path::PathBuf::from(path).join("bin/ptxas");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let candidate = entry.path().join("bin/ptxas");
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    if let Ok(output) = std::process::Command::new("which").arg("ptxas").output() {
        let candidate = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !candidate.is_empty() {
            candidates.push(candidate.into());
        }
    }
    let release = |path: &std::path::PathBuf| -> (u32, u32) {
        let Ok(output) = std::process::Command::new(path).arg("--version").output() else {
            return (0, 0);
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let Some(position) = text.find("release ") else {
            return (0, 0);
        };
        let version: String = text[position + "release ".len()..]
            .chars()
            .take_while(|character| character.is_ascii_digit() || *character == '.')
            .collect();
        let mut components = version.split('.');
        (
            components
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            components
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        )
    };
    candidates.into_iter().max_by_key(release)
}

fn cuda_tool(name: &str) -> std::path::PathBuf {
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(path) = std::env::var_os(variable) {
            let candidate = std::path::PathBuf::from(path).join("bin").join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    let standard = std::path::PathBuf::from("/usr/local/cuda/bin").join(name);
    if standard.is_file() {
        standard
    } else {
        name.into()
    }
}

fn ptxas() -> std::path::PathBuf {
    ptxas_path().unwrap_or_else(|| "ptxas".into())
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

fn assemble_and_disassemble_sm120(ptx: &str) -> (String, String) {
    let directory = tempfile::tempdir().expect("SM120 ptxas tempdir");
    let input = directory.path().join("fixed-sm120.ptx");
    let output = directory.path().join("fixed-sm120.cubin");
    std::fs::write(&input, ptx).expect("write SM120 PTX");
    let assembly = std::process::Command::new(ptxas())
        .arg("-arch=sm_120")
        .arg("-v")
        .arg("-lineinfo")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run ptxas for Fixed SM120 module");
    assert!(
        assembly.status.success(),
        "ptxas failed for the Fixed SM120 module: {}",
        String::from_utf8_lossy(&assembly.stderr)
    );
    let disassembly = std::process::Command::new(cuda_tool("nvdisasm"))
        .arg("-g")
        .arg("-hex")
        .arg(&output)
        .output()
        .expect("run nvdisasm for Fixed SM120 module");
    assert!(
        disassembly.status.success(),
        "nvdisasm failed for the Fixed SM120 module: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    (
        String::from_utf8(assembly.stderr).expect("SM120 ptxas report must be UTF-8"),
        String::from_utf8(disassembly.stdout).expect("SM120 SASS must be UTF-8"),
    )
}

fn assemble_and_disassemble_sm89_scalar(ptx: &str) -> (String, String) {
    let directory = tempfile::tempdir().expect("SM89 scalar ptxas tempdir");
    let input = directory.path().join("triad-scalar-sm89.ptx");
    let output = directory.path().join("triad-scalar-sm89.cubin");
    std::fs::write(&input, ptx).expect("write SM89 scalar PTX");
    let assembly = std::process::Command::new(ptxas())
        .arg("-arch=sm_89")
        .arg("-v")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run ptxas for SM89 scalar module");
    assert!(
        assembly.status.success(),
        "ptxas failed for the SM89 scalar module: {}",
        String::from_utf8_lossy(&assembly.stderr)
    );
    let disassembly = std::process::Command::new(cuda_tool("nvdisasm"))
        .arg("--separate-functions")
        .arg(&output)
        .output()
        .expect("run nvdisasm for SM89 scalar module");
    assert!(
        disassembly.status.success(),
        "nvdisasm failed for the SM89 scalar module: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    (
        String::from_utf8(assembly.stderr).expect("SM89 ptxas report must be UTF-8"),
        String::from_utf8(disassembly.stdout).expect("SM89 SASS must be UTF-8"),
    )
}

fn assemble_and_disassemble_sm80_scalar(ptx: &str) -> (String, String) {
    let directory = tempfile::tempdir().expect("SM80 scalar ptxas tempdir");
    let input = directory.path().join("triad-scalar-sm80.ptx");
    let output = directory.path().join("triad-scalar-sm80.cubin");
    std::fs::write(&input, ptx).expect("write SM80 scalar PTX");
    let assembly = std::process::Command::new(ptxas())
        .arg("-arch=sm_80")
        .arg("-v")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run ptxas for SM80 scalar module");
    assert!(
        assembly.status.success(),
        "ptxas failed for the SM80 scalar module: {}",
        String::from_utf8_lossy(&assembly.stderr)
    );
    let disassembly = std::process::Command::new(cuda_tool("nvdisasm"))
        .arg("--separate-functions")
        .arg(&output)
        .output()
        .expect("run nvdisasm for SM80 scalar module");
    assert!(
        disassembly.status.success(),
        "nvdisasm failed for the SM80 scalar module: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    (
        String::from_utf8(assembly.stderr).expect("SM80 ptxas report must be UTF-8"),
        String::from_utf8(disassembly.stdout).expect("SM80 SASS must be UTF-8"),
    )
}

fn ptx_entry(ptx: &str, symbol: &str) -> String {
    parse_compile_gate_ptx(ptx)
        .unwrap_or_else(|error| panic!("parse PTX entry {symbol}: {error}"))
        .entry(symbol)
        .text
        .clone()
}

fn ptx_parameters<'a>(entry: &'a str, symbol: &str) -> &'a str {
    let marker = format!(".entry {symbol}(");
    entry
        .split_once(&marker)
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(parameters, _)| parameters))
        .unwrap_or_else(|| panic!("parameter list for {symbol}"))
}

fn ptx_parameter_offset(
    reference: &str,
    bundle: &str,
    aliases: &std::collections::BTreeMap<String, usize>,
) -> Option<usize> {
    let reference: String = reference
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let reference = reference.strip_prefix('[')?.strip_suffix(']')?;
    let (base, offset) = if let Some((base, offset)) = reference.split_once('+') {
        (base, offset.parse().ok()?)
    } else {
        (reference, 0)
    };
    let base_offset = if base == bundle {
        0
    } else {
        *aliases.get(base)?
    };
    base_offset.checked_add(offset)
}

fn ptx_u32_parameter_load_offsets(entry: &str, bundle: &str) -> std::collections::BTreeSet<usize> {
    let mut aliases = std::collections::BTreeMap::<String, usize>::new();
    let mut offsets = std::collections::BTreeSet::new();
    for line in entry.lines() {
        let line = line.trim();
        if line
            .strip_suffix(':')
            .is_some_and(|label| !label.is_empty() && !label.contains(char::is_whitespace))
        {
            aliases.clear();
            continue;
        }
        let (predicated, instruction) = if line.starts_with('@') {
            let Some((_, instruction)) = line.split_once(char::is_whitespace) else {
                continue;
            };
            (true, instruction.trim_start())
        } else {
            (false, line)
        };
        let Some((opcode, operands)) = instruction.split_once(char::is_whitespace) else {
            continue;
        };
        let destination = operands
            .split_once(',')
            .map(|(destination, _)| destination.trim());
        if predicated {
            if let Some(destination) =
                destination.filter(|destination| destination.starts_with("%rd"))
            {
                aliases.remove(destination);
            }
            continue;
        }
        if matches!(opcode, "mov.b64" | "mov.u64") {
            let Some((destination, source)) = operands.trim_end_matches(';').split_once(',') else {
                continue;
            };
            let destination = destination.trim();
            let source = source.trim();
            let source_offset = if source == bundle {
                Some(0)
            } else {
                aliases.get(source).copied()
            };
            if let Some(source_offset) = source_offset {
                aliases.insert(destination.to_owned(), source_offset);
            } else {
                aliases.remove(destination);
            }
            continue;
        }
        if let Some(destination) = destination.filter(|destination| destination.starts_with("%rd"))
        {
            aliases.remove(destination);
        }
        if opcode.starts_with("ld.param.")
            && opcode
                .split('.')
                .any(|part| matches!(part, "u32" | "s32" | "b32"))
            && let Some((_, reference)) = operands.split_once(',')
            && let Some(offset) =
                ptx_parameter_offset(reference.trim_end_matches(';'), bundle, &aliases)
        {
            offsets.insert(offset);
        }
    }
    offsets
}

#[test]
fn ptx_origin_load_oracle_accepts_direct_parameter_references() {
    let bundle = "kernel_param_4";
    let entry = r#"
        ld.param.u32 %r1, [kernel_param_4];
        ld.param.s32 %r2, [kernel_param_4+4];
        ld.param.b32 %r3, [kernel_param_4+8];
        ld.param.u32 %r4, [kernel_param_4+12];
    "#;
    assert_eq!(
        ptx_u32_parameter_load_offsets(entry, bundle),
        [0, 4, 8, 12].into_iter().collect()
    );
}

#[test]
fn ptx_origin_load_oracle_accepts_aliased_parameter_base() {
    let bundle = "kernel_param_4";
    let entry = r#"
        mov.b64 %rd11, kernel_param_4;
        ld.param.u32 %r7, [%rd11];
        ld.param.u32 %r8, [%rd11+4];
        ld.param.u32 %r9, [%rd11+8];
        ld.param.u32 %r10, [%rd11+12];
    "#;
    assert_eq!(
        ptx_u32_parameter_load_offsets(entry, bundle),
        [0, 4, 8, 12].into_iter().collect()
    );
}

#[test]
fn ptx_origin_load_oracle_rejects_wrong_and_overwritten_aliases() {
    let bundle = "kernel_param_4";
    let wrong_bundle = r#"
        mov.b64 %rd11, other_kernel_param_4;
        ld.param.u32 %r7, [%rd11];
    "#;
    assert!(ptx_u32_parameter_load_offsets(wrong_bundle, bundle).is_empty());

    let overwritten = r#"
        mov.b64 %rd11, kernel_param_4;
        add.u64 %rd11, %rd12, 4;
        ld.param.u32 %r7, [%rd11];
    "#;
    assert!(ptx_u32_parameter_load_offsets(overwritten, bundle).is_empty());

    let predicated_overwrite = r#"
        mov.b64 %rd11, kernel_param_4;
        @%p1 mov.b64 %rd11, other_kernel_param_4;
        ld.param.u32 %r7, [%rd11];
    "#;
    assert!(ptx_u32_parameter_load_offsets(predicated_overwrite, bundle).is_empty());
}

#[test]
fn ptx_origin_load_oracle_rejects_aliases_across_control_flow_joins() {
    let bundle = "kernel_param_4";
    let entry = r#"
        @%p1 bra $join;
        mov.b64 %rd11, kernel_param_4;
    $join:
        ld.param.u32 %r7, [%rd11];
    "#;
    assert!(ptx_u32_parameter_load_offsets(entry, bundle).is_empty());
}

fn has_exact_maxntid(entry: &str, threads: u32) -> bool {
    let canonical = format!(".maxntid {threads}");
    let explicit = format!(".maxntid {threads}, 1, 1");
    entry
        .lines()
        .map(str::trim)
        .any(|line| line == canonical || line == explicit)
}

fn has_exact_minnctapersm(entry: &str, blocks: u32) -> bool {
    let expected = format!(".minnctapersm {blocks}");
    entry.lines().map(str::trim).any(|line| line == expected)
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

fn line_offset(source: &str, mut predicate: impl FnMut(&str) -> bool) -> Option<usize> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);
        if predicate(content) {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn ptxas_function_symbol(line: &str) -> Option<&str> {
    let (_, symbol) = line.split_once("Function properties for ")?;
    let symbol = symbol.trim();
    let symbol = if let Some(quoted) = symbol.strip_prefix('\'') {
        quoted.strip_suffix('\'')?
    } else {
        symbol
    };
    (!symbol.is_empty() && !symbol.contains(char::is_whitespace)).then_some(symbol)
}

fn nvdisasm_function_symbol(line: &str) -> Option<&str> {
    let line = line.trim();
    let line = line.strip_prefix("//").map(str::trim).unwrap_or(line);
    let symbol = line.strip_prefix("Function : ")?;
    (!symbol.is_empty() && !symbol.contains(char::is_whitespace)).then_some(symbol)
}

fn function_resource_report<'a>(report: &'a str, symbol: &str) -> &'a str {
    let start = line_offset(report, |line| ptxas_function_symbol(line) == Some(symbol))
        .unwrap_or_else(|| panic!("ptxas report omitted {symbol}"));
    let tail = &report[start..];
    let body_start = tail
        .find('\n')
        .map(|offset| offset + 1)
        .unwrap_or(tail.len());
    let end = line_offset(&tail[body_start..], |line| {
        ptxas_function_symbol(line).is_some()
    })
    .map(|offset| body_start + offset)
    .unwrap_or(tail.len());
    &tail[..end]
}

fn sass_entry<'a>(sass: &'a str, symbol: &str) -> &'a str {
    if let Some(start) = line_offset(sass, |line| nvdisasm_function_symbol(line) == Some(symbol)) {
        let tail = &sass[start..];
        let body_start = tail
            .find('\n')
            .map(|offset| offset + 1)
            .unwrap_or(tail.len());
        let end = line_offset(&tail[body_start..], |line| {
            nvdisasm_function_symbol(line).is_some()
        })
        .map(|offset| body_start + offset)
        .unwrap_or(tail.len());
        return &tail[..end];
    }
    let label = format!("\n{symbol}:\n");
    let start = sass
        .find(&label)
        .map(|offset| offset + 1)
        .unwrap_or_else(|| panic!("SASS omitted {symbol}"));
    let tail = &sass[start..];
    let end = tail
        .find("\n//--------------------- .text.")
        .or_else(|| tail.find("\n\t.section\t.text."))
        .unwrap_or(tail.len());
    &tail[..end]
}

fn sass_stores_attributed_to_line(sass: &str, source_line: usize) -> Vec<&str> {
    let marker = format!(", line {source_line}");
    let mut attributed = false;
    let mut stores = Vec::new();
    for line in sass.lines() {
        if line.contains("//## File ") {
            attributed = line.contains(&marker);
        } else if attributed && line.contains("STG") && line.contains(';') {
            stores.push(line);
        }
    }
    stores
}

#[test]
fn ptxas_resource_parser_ignores_suffix_symbols_before_the_exact_header() {
    let symbol = "gemm_bi_nn";
    for report in [
        "ptxas info : Function properties for 'gemm_bi_nn_suffix'\n\
         7 bytes stack frame\n\
         ptxas info : Function properties for 'gemm_bi_nn'\n\
         0 bytes stack frame\n\
         ptxas info : Function properties for 'gemm_bi_tn'\n\
         9 bytes stack frame\n",
        "ptxas info : Function properties for gemm_bi_nn_suffix\n\
         7 bytes stack frame\n\
         ptxas info : Function properties for gemm_bi_nn\n\
         0 bytes stack frame\n\
         ptxas info : Function properties for gemm_bi_tn\n\
         9 bytes stack frame\n",
    ] {
        let resources = function_resource_report(report, symbol);
        assert!(resources.contains("0 bytes stack frame"));
        assert!(!resources.contains("7 bytes stack frame"));
        assert!(!resources.contains("9 bytes stack frame"));
    }
}

#[test]
fn nvdisasm_parser_ignores_suffix_symbols_before_the_exact_header() {
    let sass = "// Function : gemm_bi_nn_suffix\n\
                /*0000*/ LDL R0, [R1] ;\n\
                \tFunction : gemm_bi_nn\n\
                /*0010*/ FFMA R0, R1, R2, R3 ;\n\
                \tFunction : gemm_bi_tn\n\
                /*0020*/ STL [R1], R0 ;\n";
    let entry = sass_entry(sass, "gemm_bi_nn");
    assert!(entry.contains("FFMA"));
    assert!(!entry.contains("LDL"));
    assert!(!entry.contains("STL"));
}

fn cp_async_sizes(entry: &str) -> std::collections::BTreeSet<u64> {
    entry
        .lines()
        .filter(|line| line.contains("cp.async.ca.shared.global"))
        .filter_map(|line| {
            let compact: String = line
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect();
            [4_u64, 16]
                .into_iter()
                .find(|bytes| compact.contains(&format!(",{bytes},")))
        })
        .collect()
}

fn assert_zero_local_resources(report: &str, target: &str) {
    // Below CUDA 12.9 the assembler spills kernels the contract toolkits keep
    // in registers; the loader excludes such a symbol on that toolkit, so
    // the gate reports the spill there instead of failing on it.
    let toolkit_variance = nvrtc_version() < (12, 9);
    let mut function = "unknown";
    for line in report.lines() {
        if let Some(symbol) = line.split("Function properties for ").nth(1) {
            function = symbol.trim();
        }
        for marker in [
            " bytes stack frame",
            " bytes spill stores",
            " bytes spill loads",
        ] {
            if let Some(value) = metric_before(line, marker) {
                if value != 0 && toolkit_variance {
                    eprintln!("{target}/{function} uses local resources on this toolkit: {line}");
                    continue;
                }
                assert_eq!(value, 0, "{target}/{function} uses local resources: {line}");
            }
        }
    }
}

#[test]
fn scalar_big_nn_and_tn_epilogues_have_no_terminal_cta_barrier() {
    let source = include_str!("../kernels/gemm_bi_triad/scalar.cu");
    let mut terminal_barriers = Vec::new();
    for (symbol, epilogue_marker, next_symbol) in [
        ("gemm_bi_nn", "#undef ISSUE_TILE", "gemm_bi_tn_impl"),
        ("gemm_bi_tn_impl", "#undef ISSUE_TILE_TN", "gemm_bi_tn"),
    ] {
        let start_marker = format!("void {symbol}(");
        let start = source
            .find(&start_marker)
            .unwrap_or_else(|| panic!("missing {symbol} entry"));
        let end_marker = format!("void {next_symbol}(");
        let end = source[start..]
            .find(&end_marker)
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("missing entry after {symbol}"));
        let kernel = &source[start..end];
        let (pipeline, epilogue) = kernel
            .split_once(epilogue_marker)
            .unwrap_or_else(|| panic!("missing {symbol} epilogue boundary"));

        assert!(
            pipeline.contains("__syncthreads();"),
            "{symbol} must retain its pipeline barriers"
        );
        let count = epilogue.matches("__syncthreads();").count();
        if count != 0 {
            terminal_barriers.push((symbol, count));
        }
    }
    assert!(
        terminal_barriers.is_empty(),
        "Big NN/TN epilogues must not synchronize after their final global stores: {terminal_barriers:?}"
    );
}

/// One NVRTC compile per (module, target, source): the gates ask for the
/// same PTX many times over, and a compile of the Fixed blob costs seconds.
fn compile_module_for(kind: &str, source: String, arch: &'static str) -> String {
    type CompileMemo = std::collections::HashMap<(String, &'static str, String), String>;
    static MEMO: std::sync::OnceLock<std::sync::Mutex<CompileMemo>> = std::sync::OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    let key = (kind.to_string(), arch, source);
    if let Some(ptx) = memo.lock().unwrap().get(&key) {
        return ptx.clone();
    }
    let ptx = compile_module_uncached(kind, key.2.clone(), arch);
    memo.lock().unwrap().insert(key, ptx.clone());
    ptx
}

fn compile_module_uncached(kind: &str, source: String, arch: &'static str) -> String {
    let group_m = if matches!(arch, "sm_80" | "sm_86" | "sm_87") {
        8
    } else {
        16
    };
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        options: vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            "--generate-line-info".to_string(),
            "-DNDEBUG".to_string(),
            format!("-DGEMM_BI_GROUP_M={group_m}"),
            "-DMAMBA_RS_STATE_CAP=256".to_string(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    match cudarc::nvrtc::compile_ptx_with_opts(source, opts) {
        Ok(ptx) => ptx.to_src(),
        Err(error) => panic!("{kind} kernel module does not compile for {arch}: {error}"),
    }
}

fn module_sources(arch: &str) -> [(&'static str, String); 4] {
    [
        ("Fixed", fixed_blob_for(arch)),
        ("TriadScalar", scalar_blob()),
        ("TriadSm80", sm80_blob()),
        ("TriadSm80+extensions", sm80_streamk_blob()),
    ]
}

fn compile_for(arch: &'static str) {
    for (kind, source) in module_sources(arch) {
        let ptx = compile_module_for(kind, source, arch);
        if kind == "Fixed" {
            assert_fixed_tf32_ptx(arch, &ptx);
        }
    }
}

fn compile_fixed_for(arch: &'static str) -> String {
    compile_module_for("Fixed", fixed_blob_for(arch), arch)
}

fn normalized_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn fixed_sm120_tf32_pair_store_production_source_contract() {
    const CANDIDATE: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store";
    let source = include_str!("../kernels/gemm_bi_fixed/tf32_sm120.cu");
    assert_eq!(source.matches(CANDIDATE).count(), 1, "pair-store export");
    assert!(normalized_whitespace(source).contains(
        "template <int M, int N, int Stages, int WarpN, bool PairStore, bool ProducerWarp>"
    ));
    for required in [
        "if constexpr (PairStore)",
        "params.m >= M",
        "output_row <= params.m - M",
        "params.n >= N",
        "output_column <= params.n - N",
        "reinterpret_cast<unsigned long long>(output) & 7ULL",
        "(params.ldc & 1) == 0",
        "for (int element = 0; element < 4; element += 2)",
        "float2 pair = {",
        "*reinterpret_cast<float2*>",
    ] {
        assert!(
            source.contains(required),
            "pair-store source is missing {required}"
        );
    }
    let lower = source.to_ascii_lowercase();
    for forbidden in ["split_k", "splitk", "--use_fast_math"] {
        assert!(
            !lower.contains(forbidden),
            "pair-store source contains forbidden token {forbidden}"
        );
    }
    let bias_initialization = source
        .find("accumulator[m_atom][n_atom][element] =")
        .expect("bias initialization");
    let mainloop = source
        .find("for (int tile = 0; tile < tile_count; ++tile)")
        .expect("ascending tile mainloop");
    let pair_epilogue = source
        .find("if constexpr (PairStore)")
        .expect("pair-store epilogue");
    let scalar_epilogue = source
        .rfind("output[(long long)row * params.ldc + column] =")
        .expect("scalar fallback epilogue");
    assert!(
        bias_initialization < mainloop
            && mainloop < pair_epilogue
            && pair_epilogue < scalar_epilogue,
        "pair-store epilogue must execute after bias initialization and the complete TMA/MMA mainloop, immediately before the scalar fallback"
    );

    let fixed_tile = include_str!("../src/mamba_ssm/gpu/gemm_bi_fixed.rs");
    assert!(
        !fixed_tile.contains("Tf32Sm120M64S2PairStore"),
        "the private production schedule must not widen FixedTile"
    );
    let inventory = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert_eq!(
        inventory.matches(CANDIDATE).count(),
        1,
        "runtime inventory must contain the private pair-store symbol exactly once"
    );
}

#[test]
fn fixed_sm120_tf32_producer_warp_candidate_source_contract() {
    const M64_CANDIDATE: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp";
    let source = include_str!("../kernels/gemm_bi_fixed/tf32_sm120.cu");
    assert_eq!(
        source.matches(M64_CANDIDATE).count(),
        1,
        "M64 candidate export"
    );
    assert!(normalized_whitespace(source).contains(
        "template <int M, int N, int Stages, int WarpN, bool PairStore, bool ProducerWarp>"
    ));
    for required in [
        "constexpr int compute_warps = GbfSm120Tf32Warps<M, N, WarpN>::threads / 32",
        "int compute_warp = ProducerWarp ? warp - 1 : warp",
        "if constexpr (ProducerWarp)",
        "int consumed = refill - Stages",
        "gbf_sm120_wait_barrier(empty_base + stage * 8, generation & 1U)",
    ] {
        assert!(
            source.contains(required),
            "producer source is missing {required}"
        );
    }

    let fixed = include_str!("../src/mamba_ssm/gpu/gemm_bi_fixed.rs");
    assert!(
        fixed.contains("Tf32Sm120M64S2ProducerWarp"),
        "forced M64 candidate tile"
    );
    assert!(
        fixed.contains("(&kernels.m64n64_s2_producer_warp, 64, 64, 160, 32_896)"),
        "M64 candidate launch geometry"
    );

    let holders = include_str!("../src/mamba_ssm/gpu/kernels.rs");
    assert!(holders.contains("pub m64n64_s2_producer_warp: CudaFunction"));
    assert!(holders.contains("\"gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp\""));
    assert!(holders.contains("(&kernels.m64n64_s2_producer_warp, 32_896)"));
}

#[test]
fn fixed_sm120_tf32_pair_store_production_target_matrix() {
    const INCUMBENT: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2";
    const PAIR_STORE: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store";
    let targets: &[&str] = if nvrtc_version() >= (12, 9) {
        &["compute_120", "sm_120", "compute_121", "sm_121"]
    } else {
        &["compute_120", "sm_120"]
    };
    let artifact_dir =
        std::env::var_os("GEMM_BI_TF32_PAIR_STORE_ARTIFACT_DIR").map(std::path::PathBuf::from);
    if let Some(directory) = &artifact_dir {
        std::fs::create_dir_all(directory).expect("create pair-store artifact directory");
    }
    for &target in targets {
        let ptx = compile_fixed_for(target);
        assert_fixed_tf32_ptx(target, &ptx);
        let parsed = parse_compile_gate_ptx(&ptx).expect("parse pair-store target PTX");
        let incumbent = parsed.entry(INCUMBENT);
        let pair_store = parsed.entry(PAIR_STORE);
        let is_eight_byte_store = |opcode: &str| {
            opcode.starts_with("st.global.v2.")
                || matches!(opcode, "st.global.u64" | "st.global.b64" | "st.global.f64")
        };
        assert_eq!(
            compile_gate_ptx_tokens(&incumbent.body)
                .iter()
                .filter(|token| is_eight_byte_store(token.text))
                .count(),
            0,
            "{target} incumbent pair-store inventory"
        );
        assert_eq!(
            compile_gate_ptx_tokens(&pair_store.body)
                .iter()
                .filter(|token| is_eight_byte_store(token.text))
                .count(),
            16,
            "{target} pair-store inventory"
        );
        if let Some(directory) = &artifact_dir {
            std::fs::write(directory.join(format!("fixed-{target}.ptx")), &ptx)
                .expect("write pair-store target PTX");
        }
    }
}

#[test]
fn fixed_sm120_tf32_pair_store_production_codegen_contract() {
    const INCUMBENT: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2";
    const CANDIDATE: &str = "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store";
    let source = fixed_blob();
    let fast_store_line = source
        .lines()
        .position(|line| line.contains("output + (long long)row * params.ldc + column) = pair;"))
        .map(|line| line + 1)
        .expect("pair-store source line");
    let fallback_store_line = source
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("output[(long long)row * params.ldc + column] ="))
        .map(|(line, _)| line + 1)
        .last()
        .expect("scalar-fallback source line");
    let ptx = compile_module_for("Fixed", source, "sm_120");
    let parsed = parse_compile_gate_ptx(&ptx).expect("parse Fixed SM120 PTX");
    let incumbent = parsed.entry(INCUMBENT);
    let candidate = parsed.entry(CANDIDATE);
    assert!(has_exact_maxntid(&candidate.text, 128), "candidate maxntid");
    assert_eq!(
        ptx_parameters(&candidate.text, CANDIDATE)
            .matches(".param")
            .count(),
        5,
        "candidate ABI"
    );
    assert_compile_gate_entry_tokens(
        "Fixed SM120 TF32 pair-store production schedule",
        candidate,
        &[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "cvt.rna.tf32.f32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ],
    );
    let vector_store = |opcode: &str| {
        opcode.starts_with("st.global.v2.")
            || matches!(opcode, "st.global.u64" | "st.global.b64" | "st.global.f64")
    };
    let candidate_tokens = compile_gate_ptx_tokens(&candidate.body);
    let incumbent_tokens = compile_gate_ptx_tokens(&incumbent.body);
    assert_eq!(
        candidate_tokens
            .iter()
            .filter(|token| vector_store(token.text))
            .count(),
        16,
        "candidate must emit one eight-byte PTX store per unrolled accumulator pair"
    );
    assert_eq!(
        incumbent_tokens
            .iter()
            .filter(|token| vector_store(token.text))
            .count(),
        0,
        "incumbent must remain scalar"
    );
    for token in &candidate_tokens {
        assert!(
            !token.text.starts_with("atom.")
                && !token.text.starts_with("atom::")
                && !token.text.starts_with("red.")
                && !token.text.starts_with("red::")
                && !token.text.starts_with("redux.")
                && !token.text.starts_with("ld.local")
                && !token.text.starts_with("st.local"),
            "candidate contains forbidden PTX opcode {}",
            token.text
        );
        if token.text.contains("mma") {
            assert_eq!(
                token.text, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "candidate contains a foreign MMA family"
            );
        }
    }

    let (report, sass) = assemble_and_disassemble_sm120(&ptx);
    let incumbent_resources = function_resource_report(&report, INCUMBENT);
    assert_zero_local_resources(incumbent_resources, "Fixed SM120 TF32 pair-store incumbent");
    let resources = function_resource_report(&report, CANDIDATE);
    assert_zero_local_resources(resources, "Fixed SM120 TF32 pair-store production schedule");
    let incumbent_registers = incumbent_resources
        .lines()
        .find_map(|line| metric_before(line, " registers"))
        .expect("incumbent register usage");
    let registers = resources
        .lines()
        .find_map(|line| metric_before(line, " registers"))
        .expect("candidate register usage");
    assert!(registers <= 128, "pair-store uses {registers} registers");
    if nvrtc_version() == (12, 8) {
        assert!(
            registers <= incumbent_registers + 4,
            "pair-store uses {registers} registers, incumbent uses {incumbent_registers}"
        );
    }
    if nvrtc_version() == (13, 0) {
        assert!(registers <= 96, "candidate uses {registers} registers");
    }
    let candidate_sass = sass_entry(&sass, CANDIDATE);
    let fast_stores = sass_stores_attributed_to_line(candidate_sass, fast_store_line);
    assert_eq!(
        fast_stores.len(),
        16,
        "candidate fast source line must lower to one SASS store per unrolled pair"
    );
    assert!(
        fast_stores.iter().all(|line| line.contains("STG.E.64")),
        "candidate fast source line must lower only to eight-byte SASS stores: {fast_stores:?}"
    );
    let fallback_stores = sass_stores_attributed_to_line(candidate_sass, fallback_store_line);
    assert_eq!(
        fallback_stores.len(),
        32,
        "candidate scalar fallback source line must retain one SASS store per accumulator"
    );
    assert!(
        fallback_stores
            .iter()
            .all(|line| line.contains("STG.E ") && !line.contains("STG.E.64")),
        "candidate fallback source line must lower only to four-byte SASS stores: {fallback_stores:?}"
    );
    for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX"] {
        assert!(
            !contains_opcode_prefix(candidate_sass, forbidden),
            "candidate SASS contains forbidden opcode {forbidden}"
        );
    }
}

#[test]
fn fixed_f32_n128_s2_source_and_ptx_contract() {
    const SYMBOL: &str = "gemm_bi_f32_f32_n128_s2";
    const BEGIN: &str = "// BEGIN exact-f32 N128 S2 candidate";
    const END: &str = "// END exact-f32 N128 S2 candidate";
    let source = include_str!("../kernels/gemm_bi_fixed/ffma.cu");
    assert_eq!(source.matches(BEGIN).count(), 1, "candidate BEGIN marker");
    assert_eq!(source.matches(END).count(), 1, "candidate END marker");
    assert_eq!(source.matches(SYMBOL).count(), 1, "candidate source export");
    let candidate = source
        .split_once(BEGIN)
        .and_then(|(_, tail)| tail.split_once(END).map(|(body, _)| body))
        .expect("exact-f32 N128 S2 candidate source/export");
    assert_eq!(candidate.matches(SYMBOL).count(), 1, "candidate definition");

    for required in [
        SYMBOL,
        "#define GBF_F32_N128_BM 64",
        "#define GBF_F32_N128_BN 128",
        "#define GBF_F32_N128_BK 32",
        "#define GBF_F32_N128_STAGES 2",
        "#define GBF_F32_N128_THREADS 256",
        "__launch_bounds__(GBF_F32_N128_THREADS, 2)",
        "cp.async.cg.shared.global",
        "cp.async.wait_group 0",
        "__fmaf_rn",
        "#pragma unroll 8",
    ] {
        assert!(
            candidate.contains(required),
            "exact-f32 N128 S2 source is missing {required}"
        );
    }
    let signature = candidate
        .split_once(&format!("{SYMBOL}("))
        .and_then(|(_, tail)| tail.split_once(") {").map(|(parameters, _)| parameters))
        .expect("exact-f32 N128 S2 source signature");
    assert_eq!(signature.matches(',').count() + 1, 12, "raw CUDA ABI");

    let lower = candidate.to_ascii_lowercase();
    for forbidden in ["mma", "wmma", "split_k", "splitk", "--use_fast_math"] {
        assert!(
            !lower.contains(forbidden),
            "exact-f32 N128 S2 source contains forbidden token {forbidden}"
        );
    }
    for token in candidate.split_ascii_whitespace() {
        assert!(
            !token.starts_with("atom.")
                && !token.starts_with("atom::")
                && !token.starts_with("red.")
                && !token.starts_with("red::")
                && !token.starts_with("redux."),
            "exact-f32 N128 S2 source contains forbidden instruction token {token}"
        );
    }
    let compiler = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert!(compiler.contains("\"--fmad=true\".to_string()"));
    assert!(!compiler.contains("--use_fast_math"));

    let version = nvrtc_version();
    let mut architectures = vec![
        "sm_80", "sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a", "sm_120",
    ];
    if version.0 == 12 && version >= (12, 8) {
        architectures.push("sm_101a");
    }
    if version >= (12, 9) {
        architectures.push("sm_103a");
        architectures.push("sm_121");
    }
    if version >= (13, 2) {
        architectures.push("sm_110");
        architectures.push("sm_110a");
    }
    for arch in architectures {
        let ptx = compile_fixed_for(arch);
        let parsed = parse_compile_gate_ptx(&ptx).expect("parse Fixed PTX");
        assert_eq!(
            parsed
                .entries
                .iter()
                .filter(|entry| entry.symbol == SYMBOL)
                .count(),
            1,
            "{arch} candidate PTX entry count"
        );
        let entry = parsed.entry(SYMBOL);
        assert!(has_exact_maxntid(&entry.text, 256), "{arch} maxntid");
        assert!(
            has_exact_minnctapersm(&entry.text, 2),
            "{arch} minnctapersm"
        );
        assert_eq!(
            ptx_parameters(&entry.text, SYMBOL)
                .matches(".param")
                .count(),
            12,
            "{arch} raw ABI"
        );
        assert_compile_gate_entry_tokens(
            "exact-f32 N128 S2",
            entry,
            &["cp.async.cg.shared.global", "fma.rn.f32"],
        );
        for token in compile_gate_ptx_tokens(&entry.body) {
            assert!(
                !token.text.starts_with("atom.")
                    && !token.text.starts_with("atom::")
                    && !token.text.starts_with("red.")
                    && !token.text.starts_with("red::")
                    && !token.text.starts_with("redux.")
                    && !token.text.contains("mma"),
                "{arch}/{SYMBOL} contains forbidden PTX token {}",
                token.text
            );
        }
    }
}

const FIXED_SM89_EXACT_N64_COPYPLAN: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";

fn assert_fixed_sm89_exact_n64_copyplan_ptx(arch: &str, ptx: &str) {
    let parsed = parse_compile_gate_ptx(ptx).expect("parse Fixed exact N64 copy-plan PTX");
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| {
            entry
                .symbol
                .starts_with("gemm_bi_nn_fixed_sm89_f32_n64_copyplan")
        })
        .map(|entry| entry.symbol.as_str())
        .collect();
    let expected = if arch == "sm_89" {
        vec![FIXED_SM89_EXACT_N64_COPYPLAN]
    } else {
        vec![]
    };
    assert_eq!(actual, expected, "{arch} Fixed exact N64 inventory");
    if arch != "sm_89" {
        return;
    }
    let entry = parsed.entry(FIXED_SM89_EXACT_N64_COPYPLAN);
    assert!(
        has_exact_maxntid(&entry.text, 128),
        "{arch} exact N64 maxntid"
    );
    assert!(
        has_exact_minnctapersm(&entry.text, 2),
        "{arch} exact N64 min CTAs"
    );
    let parameters = ptx_parameters(&entry.text, FIXED_SM89_EXACT_N64_COPYPLAN);
    let declarations: Vec<_> = parameters
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect();
    assert_eq!(declarations.len(), 5, "exact N64 five-argument ABI");
    assert!(
        declarations[..4]
            .iter()
            .all(|line| line.starts_with(".param .u64 "))
    );
    assert!(
        declarations[4].starts_with(".param .align 4 .b8 ") && declarations[4].contains("[32]"),
        "exact N64 compact parameter ABI"
    );
    assert_compile_gate_entry_tokens(
        "Fixed SM89 exact N64 copy-plan",
        entry,
        &[
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "fma.rn.f32",
        ],
    );
    for token in compile_gate_ptx_tokens(&entry.body) {
        assert!(
            token.text != ".local"
                && !token.text.starts_with("ld.local")
                && !token.text.starts_with("st.local")
                && !token.text.starts_with("mma.")
                && !token.text.starts_with("wmma.")
                && !token
                    .text
                    .split('.')
                    .any(|part| part == "tf32" || part == "ftz")
                && !token.text.starts_with("atom.")
                && !token.text.starts_with("atom::")
                && !token.text.starts_with("red.")
                && !token.text.starts_with("red::")
                && !token.text.starts_with("redux."),
            "exact N64 forbidden PTX token {}",
            token.text,
        );
    }
}

#[test]
fn fixed_sm89_exact_n64_copyplan_source_contract_and_target_boundary() {
    let candidate = include_str!("../kernels/gemm_bi_fixed/sm89_f32_n64_copyplan.cu");
    assert_eq!(candidate.matches(FIXED_SM89_EXACT_N64_COPYPLAN).count(), 1);
    let parameters = candidate
        .split_once(&format!("{FIXED_SM89_EXACT_N64_COPYPLAN}("))
        .and_then(|(_, tail)| tail.split_once(") {").map(|(parameters, _)| parameters))
        .unwrap();
    assert_eq!(
        parameters.matches(',').count() + 1,
        5,
        "production maximum-seven ABI"
    );
    for required in [
        "sizeof(FixedSm89ExactF32Params) == 32",
        "alignof(FixedSm89ExactF32Params) == 4",
        "float alpha, beta;",
        "int m, n, k, lda, ldb, ldc;",
        "__launch_bounds__(SM89_EXACT_N64_CP_THREADS, 2)",
        "SM89_EXACT_N64_CP_BM == 64 && SM89_EXACT_N64_CP_BN == 64 && SM89_EXACT_N64_CP_BK == 32",
        "smem_a[2][SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK]",
        "smem_b[2][SM89_EXACT_N64_CP_BK * SM89_EXACT_N64_CP_BN]",
        "const float alpha = params.alpha, beta = params.beta;",
        "const int m = params.m, n = params.n, k = params.k;",
        "const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;",
    ] {
        assert!(
            candidate.contains(required),
            "exact N64 source omitted {required}"
        );
    }
    let base = fixed_blob();
    for arch in [
        "sm_80",
        "sm_86",
        "sm_87",
        "compute_89",
        "sm_90a",
        "sm_100a",
        "sm_103a",
        "sm_110a",
        "sm_120",
        "sm_121",
        "compute_120",
        "compute_121",
    ] {
        assert_eq!(fixed_blob_for(arch), base, "{arch} Fixed base bytes");
    }
    let mut retained = base;
    retained.push('\n');
    retained.push_str(include_str!(
        "../kernels/gemm_bi_fixed/sm89_half_pipeline.cu"
    ));
    let ada = fixed_blob_for("sm_89");
    assert_eq!(
        ada.strip_prefix(&retained).unwrap(),
        format!("\n{candidate}")
    );
}

#[test]
fn fixed_sm89_exact_n64_copyplan_nvrtc_ptx_and_zero_spill_resources() {
    let ptx = compile_fixed_for("sm_89");
    assert_fixed_sm89_exact_n64_copyplan_ptx("sm_89", &ptx);
    // Existing assembler utility accepts a whole PTX module; its temporary names
    // do not affect the Fixed artifact or the exact per-symbol resource census.
    let (report, sass) = assemble_and_disassemble_sm89_scalar(&ptx);
    let resources = function_resource_report(&report, FIXED_SM89_EXACT_N64_COPYPLAN);
    assert_zero_local_resources(resources, "SM89 Fixed exact N64 copy-plan");
    for marker in [
        " bytes stack frame",
        " bytes spill stores",
        " bytes spill loads",
    ] {
        let values: Vec<_> = resources
            .lines()
            .filter_map(|line| metric_before(line, marker))
            .collect();
        assert_eq!(values, [0], "exact N64 requires explicit zero{marker}");
    }
    let registers = resources
        .lines()
        .find_map(|line| metric_before(line, " registers"))
        .unwrap();
    assert!(
        (1..=160).contains(&registers),
        "exact N64 registers {registers} exceed 160"
    );
    let shared = resources
        .lines()
        .find_map(|line| metric_before(line, " bytes smem"))
        .unwrap();
    assert_eq!(shared, 32_768, "exact N64 static shared contract");
    let entry = sass_entry(&sass, FIXED_SM89_EXACT_N64_COPYPLAN);
    for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX", "HMMA", "IMMA", "DMMA"] {
        assert!(
            !contains_opcode_prefix(entry, forbidden),
            "exact N64 SASS contains {forbidden}"
        );
    }
    assert!(
        contains_opcode_prefix(entry, "FFMA"),
        "exact N64 SASS omitted FMA"
    );
    assert!(
        contains_opcode_prefix(entry, "LDGSTS"),
        "exact N64 SASS omitted asynchronous copy"
    );
    println!(
        "SM89 Fixed exact N64 copy-plan: registers={registers} static_shared={shared} stack=0 spills=0"
    );
}

fn assert_fixed_sm89_half_pipeline_ptx(arch: &str, ptx: &str) {
    const SYMBOLS: [&str; 2] = [
        "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_bf16",
        "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_f16",
    ];
    let parsed = parse_compile_gate_ptx(ptx).expect("parse Fixed half pipeline PTX");
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| {
            entry
                .symbol
                .starts_with("gemm_bi_nn_fixed_sm89_tc128_pipeline")
        })
        .map(|entry| entry.symbol.as_str())
        .collect();
    let unique: std::collections::BTreeSet<_> = actual.iter().copied().collect();
    let expected = if arch == "sm_89" {
        SYMBOLS.into_iter().collect()
    } else {
        std::collections::BTreeSet::new()
    };
    assert_eq!(
        actual.len(),
        unique.len(),
        "{arch} duplicated Fixed half pipeline export"
    );
    assert_eq!(
        unique, expected,
        "{arch} Fixed half pipeline export inventory"
    );
    for symbol in expected {
        let entry = parsed.entry(symbol);
        let mma = if symbol.ends_with("_bf16") {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        assert_compile_gate_entry_tokens(
            "Fixed SM89 half pipeline",
            entry,
            &[
                mma,
                "cp.async.cg.shared.global",
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
        );
        let parameters = ptx_parameters(&entry.text, symbol);
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        assert_eq!(declarations.len(), 5, "{arch}/{symbol} five-argument ABI");
        assert!(
            declarations[..4]
                .iter()
                .all(|line| line.starts_with(".param .u64 ")),
            "{arch}/{symbol} pointer ABI"
        );
        assert!(
            declarations[4].starts_with(".param .align 4 .b8 ") && declarations[4].contains("[32]"),
            "{arch}/{symbol} parameter-bundle ABI"
        );
        assert!(
            !compile_gate_ptx_tokens(&entry.body)
                .into_iter()
                .any(|token| {
                    token.text.starts_with("atom.")
                        || token.text.starts_with("atom::")
                        || token.text.starts_with("red.")
                        || token.text.starts_with("red::")
                        || token.text.starts_with("redux.")
                }),
            "{arch}/{symbol} contains a numeric atomic or reduction"
        );
    }
}

fn assert_fixed_tf32_ptx(arch: &str, ptx: &str) {
    assert_fixed_sm89_half_pipeline_ptx(arch, ptx);
    assert_fixed_sm89_exact_n64_copyplan_ptx(arch, ptx);
    const PORTABLE: [&str; 5] = [
        "gemm_bi_nn_tf32_v1_m128n64_bk32_s2",
        "gemm_bi_nn_tf32_v1_m128n64_bk32_s3",
        "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
        "gemm_bi_nn_tf32_v1_m64n64_bk32_s3",
        "gemm_bi_nn_tf32_v1_m16n32_bk32_s4",
    ];
    const SM120: [&str; 7] = [
        "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2",
        "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s3",
        "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s2",
        "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s3",
        "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp",
        "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2",
        "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store",
    ];
    const SM120_HALF_BASES: [&str; 5] = [
        "gemm_bi_nn_sm120_tma_64x64_bk64_s2",
        "gemm_bi_nn_sm120_tma_64x128_bk64_s2",
        "gemm_bi_nn_sm120_tma_128x64_bk32_s3",
        "gemm_bi_nn_sm120_tma_128x128_bk32_s2",
        "gemm_bi_nn_sm120_tma_128x128_bk32_s3",
    ];
    let parsed = parse_compile_gate_ptx(ptx).expect("parse Fixed PTX");
    let owns_sm120 = matches!(arch, "sm_120" | "sm_121" | "compute_120" | "compute_121");
    let mut expected = PORTABLE.to_vec();
    if owns_sm120 {
        expected.extend(SM120);
    }
    let actual = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.contains("_tf32_v1_"))
        .map(|entry| entry.symbol.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        actual,
        expected.iter().copied().collect(),
        "{arch} Fixed TF32 export inventory"
    );
    for symbol in expected {
        let entry = parsed.entry(symbol);
        assert_compile_gate_entry_tokens(
            "Fixed TF32",
            entry,
            &[
                "cvt.rna.tf32.f32",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            ],
        );
        assert!(
            !compile_gate_ptx_tokens(&entry.body)
                .into_iter()
                .any(|token| {
                    token.text.starts_with("atom.")
                        || token.text.starts_with("atom::")
                        || token.text.starts_with("red.")
                        || token.text.starts_with("red::")
                        || token.text.starts_with("redux.")
                }),
            "{arch}/{symbol} contains a reduction instruction"
        );
        let parameters = ptx_parameters(&entry.text, symbol);
        assert_eq!(
            parameters.matches(".param").count(),
            5,
            "{arch}/{symbol} ABI"
        );
    }
    if owns_sm120 {
        for symbol in SM120 {
            assert_compile_gate_entry_tokens(
                "Fixed SM120 TF32",
                parsed.entry(symbol),
                &["cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
            );
        }
    }
    let expected_half = if owns_sm120 {
        SM120_HALF_BASES
            .iter()
            .flat_map(|base| {
                [
                    format!("{base}_bf16"),
                    format!("{base}_f16"),
                    format!("{base}_f32out_bf16"),
                    format!("{base}_f32out_f16"),
                ]
            })
            .collect::<std::collections::BTreeSet<_>>()
    } else {
        std::collections::BTreeSet::new()
    };
    let actual_half = parsed
        .entries
        .iter()
        .filter(|entry| {
            entry.symbol.starts_with("gemm_bi_nn_sm120_tma_")
                && !entry.symbol.contains("_tf32_v1_")
                && (entry.symbol.ends_with("_bf16") || entry.symbol.ends_with("_f16"))
        })
        .map(|entry| entry.symbol.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        actual_half, expected_half,
        "{arch} Fixed SM120 half export inventory"
    );
    for symbol in expected_half {
        let entry = parsed.entry(&symbol);
        let mma = if symbol.ends_with("_bf16") {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        assert_compile_gate_entry_tokens(
            "Fixed SM120 half",
            entry,
            &[
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                mma,
            ],
        );
        assert_eq!(
            ptx_parameters(&entry.text, &symbol)
                .matches(".param")
                .count(),
            5,
            "{arch}/{symbol} ABI"
        );
        assert!(
            !compile_gate_ptx_tokens(&entry.body)
                .into_iter()
                .any(|token| {
                    token.text.starts_with("atom.")
                        || token.text.starts_with("atom::")
                        || token.text.starts_with("red.")
                        || token.text.starts_with("red::")
                        || token.text.starts_with("redux.")
                }),
            "{arch}/{symbol} contains a reduction instruction"
        );
    }
}

#[test]
fn compiles_for_sm80() {
    compile_for("sm_80");

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_80"),
        options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(sm80_blob(), opts)
        .expect("TriadSm80 TF32 module must compile for sm_80");
    let ptx = std::str::from_utf8(image.as_bytes().expect("SM80 PTX image"))
        .expect("SM80 PTX must be UTF-8");
    assert_eq!(sm80_tf32_symbols().len(), 18);
    for symbol in sm80_tf32_symbols() {
        assert_eq!(ptx.matches(&format!(".entry {symbol}(")).count(), 1);
    }
    for instruction in [
        "cvt.rna.tf32.f32",
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    ] {
        assert!(
            ptx.contains(instruction),
            "SM80 PTX is missing {instruction}"
        );
    }
}

#[test]
fn scalar_nt_m2n16_compiles_for_compute80_with_exact_abi_and_bounded_resources() {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_80"),
        options: vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            "-DNDEBUG".to_string(),
            "-DGEMM_BI_GROUP_M=8".to_string(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts)
        .expect("TriadScalar M2N16 must compile for compute_80");
    let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
        image.as_bytes().expect("SM80 scalar PTX image"),
    )
    .expect("SM80 scalar PTX must be canonical UTF-8");
    let parsed = parse_compile_gate_ptx(&ptx).expect("parse compute_80 TriadScalar PTX");
    assert_eq!(parsed.target, "sm_80");
    assert_eq!(
        parsed
            .entries
            .iter()
            .filter(|entry| entry.symbol == SCALAR_NT_M2N16_SYMBOL)
            .count(),
        1,
        "compute_80 TriadScalar must export the specialized entry exactly once"
    );

    let entry = ptx_entry(&ptx, SCALAR_NT_M2N16_SYMBOL);
    assert_eq!(
        ptx_parameters(&entry, SCALAR_NT_M2N16_SYMBOL)
            .matches(".param")
            .count(),
        7,
        "M2N16 packed ABI"
    );
    assert!(has_exact_maxntid(&entry, 64));
    assert!(has_exact_minnctapersm(&entry, 4));
    assert!(ptx.contains(".extern .shared"));
    assert!(entry.contains("cp.async.ca.shared.global"));
    assert!(entry.contains("fma.rn.f32"));
    for forbidden in ["atom.", "atom::", "red.", "red::", "redux."] {
        assert!(
            !contains_opcode_prefix(&entry, forbidden),
            "SM80 M2N16 PTX contains forbidden opcode {forbidden}"
        );
    }

    let (report, sass) = assemble_and_disassemble_sm80_scalar(&ptx);
    let resources = function_resource_report(&report, SCALAR_NT_M2N16_SYMBOL);
    assert_zero_local_resources(resources, "SM80 scalar M2N16");
    for marker in [
        " bytes stack frame",
        " bytes spill stores",
        " bytes spill loads",
    ] {
        let values = resources
            .lines()
            .filter_map(|line| metric_before(line, marker))
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            [0],
            "SM80 M2N16 requires one explicit zero{marker} record"
        );
    }
    let registers = resources
        .lines()
        .find_map(|line| metric_before(line, " registers"))
        .expect("SM80 M2N16 register usage");
    assert!(
        registers <= 112,
        "SM80 M2N16 uses {registers} registers; cap is 112"
    );

    let sass = sass_entry(&sass, SCALAR_NT_M2N16_SYMBOL);
    assert!(
        sass.lines()
            .any(|line| contains_opcode_prefix(line, "LDGSTS")),
        "SM80 M2N16 SASS omitted asynchronous global-to-shared copies"
    );
    assert!(
        sass.lines()
            .any(|line| contains_opcode_prefix(line, "FFMA")),
        "SM80 M2N16 SASS omitted deterministic FMA work"
    );
    for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX"] {
        assert!(
            !contains_opcode_prefix(sass, forbidden),
            "SM80 M2N16 SASS contains forbidden opcode {forbidden}"
        );
    }
}

#[test]
fn scalar_nn_m32n64_splitk32_compiles_for_compute80_and_sm120_with_exact_contract() {
    let compile = |target: &'static str, group_m: u32| {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(target),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                "-DNDEBUG".to_string(),
                format!("-DGEMM_BI_GROUP_M={group_m}"),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let image =
            cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts).unwrap_or_else(|error| {
                panic!("TriadScalar NN M32N64 Split-K must compile for {target}: {error:?}")
            });
        mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
            image.as_bytes().expect("scalar M32N64 Split-K PTX image"),
        )
        .expect("scalar M32N64 Split-K PTX must be canonical UTF-8")
    };

    for (target, group_m, emitted) in [("compute_80", 8, "sm_80"), ("compute_120", 16, "sm_120")] {
        let ptx = compile(target, group_m);
        let parsed = parse_compile_gate_ptx(&ptx).expect("parse TriadScalar M32N64 Split-K PTX");
        assert_eq!(parsed.target, emitted);
        assert_eq!(
            parsed
                .entries
                .iter()
                .filter(|entry| entry.symbol == SCALAR_NN_M32N64_SPLITK32_SYMBOL)
                .count(),
            1
        );
        let entry = ptx_entry(&ptx, SCALAR_NN_M32N64_SPLITK32_SYMBOL);
        assert_eq!(
            ptx_parameters(&entry, SCALAR_NN_M32N64_SPLITK32_SYMBOL)
                .matches(".param")
                .count(),
            7
        );
        assert!(has_exact_maxntid(&entry, 128));
        assert!(has_exact_minnctapersm(&entry, 4));
        assert!(entry.contains("cp.async.ca.shared.global"));
        assert!(entry.contains("fma.rn.f32"));
        for forbidden in [
            "atom.", "atom::", "red.", "red::", "redux.", "mma.", "call.",
        ] {
            assert!(
                !contains_opcode_prefix(&entry, forbidden),
                "{target} NN M32N64 Split-K contains forbidden opcode {forbidden}"
            );
        }
        assert!(!entry.contains(".ftz"));

        let (report, sass) = if target == "compute_120" {
            assemble_and_disassemble_sm120(&ptx)
        } else {
            assemble_and_disassemble_sm80_scalar(&ptx)
        };
        let resources = function_resource_report(&report, SCALAR_NN_M32N64_SPLITK32_SYMBOL);
        assert_zero_local_resources(resources, "scalar NN M32N64 Split-K");
        let registers = resources
            .lines()
            .find_map(|line| metric_before(line, " registers"))
            .expect("NN M32N64 Split-K register usage");
        assert!(
            registers <= 64,
            "{target} NN M32N64 Split-K uses {registers} registers"
        );
        let sass = sass_entry(&sass, SCALAR_NN_M32N64_SPLITK32_SYMBOL);
        for required in ["LDGSTS", "FFMA"] {
            assert!(
                sass.lines()
                    .any(|line| contains_opcode_prefix(line, required)),
                "{target} NN M32N64 Split-K SASS omitted {required}"
            );
        }
        for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX", "HMMA"] {
            assert!(
                !contains_opcode_prefix(sass, forbidden),
                "{target} NN M32N64 Split-K SASS contains {forbidden}"
            );
        }
    }
}

#[test]
fn scalar_tn_m16n16_compiles_for_compute80_and_sm120_with_exact_numeric_contract() {
    let compile = |target: &'static str, group_m: u32| {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(target),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                "-DNDEBUG".to_string(),
                format!("-DGEMM_BI_GROUP_M={group_m}"),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let image =
            cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts).unwrap_or_else(|error| {
                panic!("TriadScalar TN M16N16 must compile for {target}: {error:?}")
            });
        mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
            image.as_bytes().expect("scalar TN M16N16 PTX image"),
        )
        .expect("scalar TN M16N16 PTX must be canonical UTF-8")
    };

    for (target, group_m, emitted) in [("compute_80", 8, "sm_80"), ("compute_120", 16, "sm_120")] {
        let ptx = compile(target, group_m);
        let parsed = parse_compile_gate_ptx(&ptx).expect("parse TriadScalar TN M16N16 PTX");
        assert_eq!(parsed.target, emitted);
        assert_eq!(
            parsed
                .entries
                .iter()
                .filter(|entry| entry.symbol == SCALAR_TN_M16N16_SYMBOL)
                .count(),
            1
        );
        let entry = ptx_entry(&ptx, SCALAR_TN_M16N16_SYMBOL);
        assert_eq!(
            ptx_parameters(&entry, SCALAR_TN_M16N16_SYMBOL)
                .matches(".param")
                .count(),
            7
        );
        assert!(has_exact_maxntid(&entry, 64));
        assert!(has_exact_minnctapersm(&entry, 4));
        let mut previous = 0;
        for opcode in ["fma.rn.f32", "add.rn.f64", "mul.rn.f64", "add.rn.f32"] {
            let offset = entry[previous..]
                .find(opcode)
                .map(|offset| previous + offset)
                .unwrap_or_else(|| panic!("{target} TN M16N16 omitted ordered {opcode}"));
            previous = offset + opcode.len();
        }
        for forbidden in [
            "atom.", "atom::", "red.", "red::", "redux.", "mma.", "call.",
        ] {
            assert!(
                !contains_opcode_prefix(&entry, forbidden),
                "{target} TN M16N16 contains forbidden opcode {forbidden}"
            );
        }
        assert!(
            !entry.contains(".ftz"),
            "{target} TN M16N16 contains a flush-to-zero modifier"
        );

        if target == "compute_120" {
            let (report, sass) = assemble_and_disassemble_sm120(&ptx);
            let resources = function_resource_report(&report, SCALAR_TN_M16N16_SYMBOL);
            assert_zero_local_resources(resources, "SM120 scalar TN M16N16");
            for marker in [
                " bytes stack frame",
                " bytes spill stores",
                " bytes spill loads",
            ] {
                assert_eq!(
                    resources
                        .lines()
                        .filter_map(|line| metric_before(line, marker))
                        .collect::<Vec<_>>(),
                    [0]
                );
            }
            let registers = resources
                .lines()
                .find_map(|line| metric_before(line, " registers"))
                .expect("SM120 TN M16N16 register usage");
            assert!(
                registers <= 112,
                "SM120 TN M16N16 uses {registers} registers"
            );
            let sass = sass_entry(&sass, SCALAR_TN_M16N16_SYMBOL);
            for required in ["LDGSTS", "FFMA", "DADD", "DMUL", "FADD"] {
                assert!(
                    sass.lines()
                        .any(|line| contains_opcode_prefix(line, required)),
                    "SM120 TN M16N16 SASS omitted {required}"
                );
            }
            for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX"] {
                assert!(
                    !contains_opcode_prefix(sass, forbidden),
                    "SM120 TN M16N16 SASS contains {forbidden}"
                );
            }
        }
    }
}

#[test]
fn compiles_for_sm89() {
    compile_for("sm_89");
}

#[test]
fn scalar_sm89_main_and_splitk_have_bounded_resources_without_reduction_instructions() {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_89"),
        options: vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            "-DNDEBUG".to_string(),
            "-DGEMM_BI_GROUP_M=16".to_string(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts)
        .expect("TriadScalar must compile for compute_89");
    let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
        image.as_bytes().expect("SM89 scalar PTX image"),
    )
    .expect("SM89 scalar PTX must be canonical UTF-8");
    let resource_contracts = [
        ("gemm_bi_nn", 256, 2, 128),
        ("gemm_bi_nn_m64n64_bk16_s2_v1", 128, 3, 128),
        ("gemm_bi_tn", 256, 2, 128),
        ("gemm_bi_tn_aligned", 256, 2, 128),
        ("gemm_bi_nt", 256, 2, 128),
        (SCALAR_NT_M2N16_SYMBOL, 64, 4, 112),
        ("gemm_bi_nn_splitk32_partial", 128, 4, 128),
        ("gemm_bi_splitk_reduce", 256, 8, 64),
    ];
    for (symbol, threads, requested_min_ctas, _) in resource_contracts {
        let entry = ptx_entry(&ptx, symbol);
        assert!(
            has_exact_maxntid(&entry, threads),
            "SM89 {symbol} must retain its {threads}-thread launch bound"
        );
        assert!(
            has_exact_minnctapersm(&entry, requested_min_ctas),
            "SM89 {symbol} must retain its requested {requested_min_ctas}-CTA launch hint"
        );
        assert!(
            !entry
                .lines()
                .map(str::trim_start)
                .any(|line| line.starts_with(".local ")),
            "SM89 {symbol} PTX contains local storage"
        );
    }

    let ptx_nt = ptx_entry(&ptx, "gemm_bi_nt");
    assert_eq!(
        cp_async_sizes(&ptx_nt),
        [4_u64, 16].into_iter().collect(),
        "Big NT must retain both scalar-tail and aligned-vector asynchronous copies"
    );
    assert!(
        ptx_nt.contains("ld.shared.v4.u32"),
        "Big NT must retain the vector shared-memory staging load"
    );

    let (report, sass) = assemble_and_disassemble_sm89_scalar(&ptx);
    for (symbol, _, _, register_limit) in resource_contracts {
        let resources = function_resource_report(&report, symbol);
        assert_zero_local_resources(resources, &format!("SM89 {symbol}"));
        for marker in [
            " bytes stack frame",
            " bytes spill stores",
            " bytes spill loads",
        ] {
            let values = resources
                .lines()
                .filter_map(|line| metric_before(line, marker))
                .collect::<Vec<_>>();
            assert_eq!(
                values,
                [0],
                "SM89 {symbol} requires one explicit zero{marker} record"
            );
        }
        let registers = resources
            .lines()
            .find_map(|line| metric_before(line, " registers"))
            .unwrap_or_else(|| panic!("SM89 {symbol} register usage"));
        println!("SM89 {symbol}: {registers} registers; zero stack/spills/local");
        assert!(
            registers <= register_limit,
            "SM89 {symbol} uses {registers} registers; limit is {register_limit}"
        );

        let sass_entry = sass_entry(&sass, symbol);
        for forbidden in ["LDL", "STL", "ATOM", "RED", "REDUX"] {
            assert!(
                !contains_opcode_prefix(sass_entry, forbidden),
                "SM89 {symbol} SASS contains forbidden opcode {forbidden}"
            );
        }
        assert!(
            sass_entry
                .lines()
                .any(|line| contains_opcode_prefix(line, "FFMA")),
            "SM89 {symbol} SASS omitted its deterministic FMA work"
        );
    }

    let sass_nt = sass_entry(&sass, "gemm_bi_nt");
    let asynchronous_copies: Vec<_> = sass_nt
        .lines()
        .filter(|line| contains_opcode_prefix(line, "LDGSTS"))
        .collect();
    assert!(
        asynchronous_copies.iter().any(|line| line.contains(".128")),
        "SM89 gemm_bi_nt SASS omitted 16-byte asynchronous copies"
    );
    assert!(
        asynchronous_copies
            .iter()
            .any(|line| !line.contains(".128")),
        "SM89 gemm_bi_nt SASS omitted 4-byte asynchronous copies"
    );
    assert!(
        sass_nt
            .lines()
            .any(|line| contains_opcode_prefix(line, "LDS.128")),
        "SM89 gemm_bi_nt SASS omitted vector shared-memory staging"
    );
}

#[test]
fn scalar_sm89_tn_narrow_splitm_candidates_have_exact_protocols_and_bounds() {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_89"),
        options: vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            "-DNDEBUG".to_string(),
            "-DGEMM_BI_GROUP_M=16".to_string(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts)
        .expect("TriadScalar TN narrow split-M candidates must compile for compute_89");
    let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
        image.as_bytes().expect("SM89 scalar PTX image"),
    )
    .expect("SM89 scalar PTX must be canonical UTF-8");

    for symbol in [
        "gemm_bi_tn_narrow_splitm_partial",
        "gemm_bi_tn_narrow_splitm_partial_aligned",
    ] {
        let entry = ptx_entry(&ptx, symbol);
        assert!(has_exact_maxntid(&entry, 128));
        assert!(has_exact_minnctapersm(&entry, 4));
        assert_eq!(entry.matches(".param ").count(), 7);
        assert!(entry.contains("fma.rn.f32"));
        assert!(!contains_opcode_prefix(&entry, "atom."));
        assert!(!contains_opcode_prefix(&entry, "red."));
        assert!(!contains_opcode_prefix(&entry, "div."));
        assert!(!contains_opcode_prefix(&entry, "rem."));
    }

    let (report, sass) = assemble_and_disassemble_sm89_scalar(&ptx);
    for symbol in [
        "gemm_bi_tn_narrow_splitm_partial",
        "gemm_bi_tn_narrow_splitm_partial_aligned",
    ] {
        let resources = function_resource_report(&report, symbol);
        assert_zero_local_resources(resources, &format!("SM89 {symbol}"));
        let registers = resources
            .lines()
            .find_map(|line| metric_before(line, " registers"))
            .unwrap_or_else(|| panic!("SM89 {symbol} register usage"));
        println!("SM89 {symbol}: {registers} registers; zero stack/spills/local");
        assert!(
            registers <= 128,
            "SM89 {symbol} uses {registers} registers; limit is 128"
        );

        let sass_entry = sass_entry(&sass, symbol);
        for forbidden in ["LDL", "STL", "RED", "REDUX"] {
            assert!(
                !contains_opcode_prefix(sass_entry, forbidden),
                "SM89 {symbol} SASS contains forbidden opcode {forbidden}"
            );
        }
        assert!(
            !contains_opcode_prefix(sass_entry, "ATOM"),
            "SM89 {symbol} SASS contains an atomic opcode"
        );
    }
}

#[test]
fn scalar_sm89_splitm_reducer_uses_a_portable_launch_bound() {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_89"),
        options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(scalar_blob(), opts)
        .expect("TriadScalar split-M reducer must compile for compute_89");
    let ptx = mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
        image.as_bytes().expect("SM89 scalar PTX image"),
    )
    .expect("SM89 scalar PTX must be canonical UTF-8");
    let entry = ptx_entry(&ptx, "gemm_bi_splitm_reduce");
    assert!(has_exact_maxntid(&entry, 256));
    assert!(has_exact_minnctapersm(&entry, 4));
    assert!(!has_exact_minnctapersm(&entry, 8));
    assert_eq!(entry.matches(".param ").count(), 6);
    assert!(entry.contains("add.f64"));
    assert!(entry.contains("mul.f64"));
    assert!(entry.contains("cvt.rn.f32.f64"));
}

#[test]
fn compiles_for_sm90a() {
    const BF16_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16";
    const F16_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16";
    const TF32_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32";
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
    let typed_symbols = sm90a_symbols();
    let mut expected = typed_symbols.clone();
    expected.extend(sm90a_tf32_symbols());
    let parsed =
        validate_exact_ptx_exports("TriadSm90a/sm_90a", ptx, "sm_90a", &expected, 18).unwrap();
    validate_compile_gate_sm90a_wg2_producers(&parsed, &typed_symbols).unwrap();
    for symbol in typed_symbols {
        let entry = parsed.entry(&symbol);
        assert_compile_gate_entry_tokens(
            "SM90a",
            entry,
            &[
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "wgmma.fence.sync.aligned",
                "wgmma.commit_group.sync.aligned",
                "wgmma.wait_group.sync.aligned",
            ],
        );
        let (core, forbidden) = if symbol.ends_with("_bf16") {
            (BF16_CORE, [F16_CORE, TF32_CORE])
        } else {
            (F16_CORE, [BF16_CORE, TF32_CORE])
        };
        assert_compile_gate_entry_tokens("SM90a", entry, &[core]);
        assert_compile_gate_entry_excludes("SM90a", entry, &forbidden);
        if symbol.contains("_wg1_") {
            assert_compile_gate_entry_tokens(
                "SM90a",
                entry,
                &[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                ],
            );
        } else {
            assert_compile_gate_entry_tokens("SM90a", entry, &["setmaxnreg.inc.sync.aligned.u32"]);
        }
    }
    assert_eq!(sm90a_tf32_symbols().len(), 6);
    let map_alignment = if nvrtc_version().0 >= 13 { 128 } else { 64 };
    for symbol in sm90a_tf32_symbols() {
        let entry = parsed.entry(&symbol);
        assert_compile_gate_entry_tokens(
            "SM90a TF32",
            entry,
            &[
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "wgmma.fence.sync.aligned",
                "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32",
                "wgmma.commit_group.sync.aligned",
                "wgmma.wait_group.sync.aligned",
            ],
        );
        assert_compile_gate_entry_excludes("SM90a TF32", entry, &[BF16_CORE, F16_CORE]);
        if symbol.ends_with("_wg2") {
            assert_compile_gate_entry_tokens(
                "SM90a TF32",
                entry,
                &[
                    "setmaxnreg.inc.sync.aligned.u32",
                    "setmaxnreg.dec.sync.aligned.u32",
                ],
            );
        }
        let parameters = ptx_parameters(&entry.text, &symbol);
        assert_eq!(parameters.matches(".param").count(), 5);
        assert_eq!(
            parameters
                .matches(&format!(".param .align {map_alignment} .b8"))
                .count(),
            2
        );
        assert_eq!(parameters.matches("[128]").count(), 2);
        let bundle = format!("{symbol}_param_4");
        let loaded_offsets = ptx_u32_parameter_load_offsets(&entry.text, &bundle);
        for (field, offset) in [("a_x", 0), ("a_y", 4), ("b_x", 8), ("b_y", 12)] {
            assert!(
                loaded_offsets.contains(&offset),
                "SM90a TF32 {symbol} does not consume {field} from parameter bundle offset {offset}"
            );
        }
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
fn compiles_for_sm101a_when_the_active_nvrtc_supports_it() {
    let version = nvrtc_version();
    if version < (12, 8) || version.0 >= 13 {
        return;
    }
    compile_for("sm_101a");
}

#[test]
fn compiles_for_sm86() {
    compile_for("sm_86");
}

#[test]
fn compiles_for_sm87() {
    compile_for("sm_87");
}

#[test]
fn compiles_for_sm103a() {
    if nvrtc_version() < (12, 9) {
        return;
    }
    compile_for("sm_103a");
}

#[test]
fn family_targets_assemble_under_ptxas() {
    let Some(ptxas) = ptxas_path() else {
        println!("no ptxas on this box; assembly gate skipped (NVRTC gates still ran)");
        return;
    };
    let version = nvrtc_version();
    let mut architectures = vec!["sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a", "sm_120"];
    if version.0 == 12 && version >= (12, 8) {
        architectures.push("sm_101a");
    }
    if version >= (12, 9) {
        architectures.push("sm_103a");
    }
    if version >= (13, 2) {
        architectures.push("sm_110");
        architectures.push("sm_110a");
    }
    for arch in architectures {
        for (kind, source) in module_sources(arch) {
            let ptx = compile_module_for(kind, source, arch);
            let directory = tempfile::tempdir().expect("architecture-gate ptxas tempdir");
            let input = directory.path().join(format!("{kind}-{arch}.ptx"));
            let output = directory.path().join(format!("{kind}-{arch}.cubin"));
            std::fs::write(&input, ptx).expect("write architecture-gate PTX");
            let assembly = std::process::Command::new(&ptxas)
                .arg(format!("-arch={arch}"))
                .arg(&input)
                .arg("-o")
                .arg(&output)
                .output()
                .expect("run ptxas");
            let stderr = String::from_utf8_lossy(&assembly.stderr);
            if !assembly.status.success() && stderr.contains("Unsupported .version") {
                println!(
                    "ptxas predates the loaded NVRTC PTX version; assembly gate skipped ({})",
                    stderr.lines().next().unwrap_or("")
                );
                return;
            }
            assert!(
                assembly.status.success(),
                "ptxas rejected the {kind} module for {arch}:\n{stderr}"
            );
        }
    }
}

#[test]
fn fixed_tcgen05_only_in_blackwell_family_ptx() {
    let version = nvrtc_version();
    let mut tcgen_architectures = vec!["sm_100a"];
    if version.0 == 12 && version >= (12, 8) {
        tcgen_architectures.push("sm_101a");
    }
    if version >= (12, 9) {
        tcgen_architectures.push("sm_103a");
    }
    if version >= (13, 2) {
        tcgen_architectures.push("sm_110a");
    }
    for arch in tcgen_architectures {
        let ptx = compile_fixed_for(arch);
        assert!(
            ptx.contains("tcgen05.mma") && ptx.contains("tcgen05.alloc"),
            "{arch} Fixed PTX lost the tcgen05 rung"
        );
    }
    let mut baseline_architectures = vec!["sm_80", "sm_89", "sm_90a", "sm_120"];
    if version >= (13, 2) {
        baseline_architectures.push("sm_110");
    }
    for arch in baseline_architectures {
        let ptx = compile_fixed_for(arch);
        assert!(
            !ptx.contains("tcgen05"),
            "{arch} Fixed PTX must not contain tcgen05 instructions"
        );
    }
}

#[test]
fn compiles_for_sm110() {
    if nvrtc_version() < (13, 2) {
        return;
    }
    compile_for("sm_110");
}

#[test]
fn compiles_exact_sm100_family_triad_modules() {
    const F16_CORE: &str = "tcgen05.mma.cta_group::1.kind::f16";
    const TF32_CORE: &str = "tcgen05.mma.cta_group::1.kind::tf32";
    let version = nvrtc_version();
    let mut targets = vec![("compute_100a", "sm_100a")];
    if version >= (12, 9) {
        targets = vec![
            ("compute_100f", "sm_100f"),
            ("compute_100a", "sm_100a"),
            ("compute_103f", "sm_103f"),
            ("compute_103a", "sm_103a"),
        ];
    }
    if version >= (13, 2) {
        targets.extend([("compute_110f", "sm_110f"), ("compute_110a", "sm_110a")]);
    }
    for (requested, emitted) in targets {
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
        let mut expected = sm100_symbols();
        expected.extend(sm100_tf32_symbols());
        let parsed = validate_exact_ptx_exports(
            &format!("TriadSm100/{requested}"),
            &ptx,
            emitted,
            &expected,
            108,
        )
        .unwrap();
        let common = [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
            "tcgen05.fence::before_thread_sync",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
        ];
        for symbol in sm100_symbols() {
            let entry = parsed.entry(&symbol);
            assert_compile_gate_entry_tokens(requested, entry, &common);
            assert_compile_gate_entry_tokens(requested, entry, &[F16_CORE]);
            assert_compile_gate_entry_excludes(requested, entry, &[TF32_CORE]);
            if symbol.contains("_nn_") {
                assert_compile_gate_entry_tokens(
                    requested,
                    entry,
                    &[
                        "tcgen05.st.sync.aligned.32x32b.x8.b32",
                        "tcgen05.wait::st.sync.aligned",
                    ],
                );
            }
            let parameters = ptx_parameters(&entry.text, &symbol);
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
        assert_eq!(sm100_tf32_symbols().len(), 36);
        let map_alignment = if nvrtc_version().0 >= 13 { 128 } else { 64 };
        for symbol in sm100_tf32_symbols() {
            let entry = parsed.entry(&symbol);
            assert_compile_gate_entry_tokens(requested, entry, &common);
            assert_compile_gate_entry_tokens(requested, entry, &[TF32_CORE]);
            assert_compile_gate_entry_excludes(requested, entry, &[F16_CORE]);
            if symbol.contains("_nn_") {
                assert_compile_gate_entry_tokens(
                    requested,
                    entry,
                    &[
                        "tcgen05.st.sync.aligned.32x32b.x8.b32",
                        "tcgen05.wait::st.sync.aligned",
                    ],
                );
            }
            let parameters = ptx_parameters(&entry.text, &symbol);
            assert_eq!(parameters.matches(".param").count(), 5);
            assert_eq!(
                parameters
                    .matches(&format!(".param .align {map_alignment} .b8"))
                    .count(),
                2
            );
            assert_eq!(parameters.matches("[128]").count(), 2);
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
    let version = nvrtc_version();
    let mut targets = vec![("compute_100", "sm_100")];
    if version >= (12, 9) {
        targets.push(("compute_103", "sm_103"));
    }
    if version >= (13, 2) {
        targets.push(("compute_110", "sm_110"));
    }
    for (requested, emitted) in targets {
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
    const BF16_CORE: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
    const F16_CORE: &str = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
    const TF32_CORE: &str = "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32";
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
        assert!(!ptx.contains(".target sm_120a"));
        assert!(!ptx.contains(".target sm_120f"));
        assert!(!ptx.contains(".target sm_121a"));
        assert!(!ptx.contains(".target sm_121f"));

        let mut expected = sm120_symbols();
        expected.extend(sm120_tf32_symbols());
        let parsed = validate_exact_ptx_exports(
            &format!("TriadSm120/{requested}"),
            &ptx,
            emitted,
            &expected,
            128,
        )
        .unwrap();

        let map_alignment = if version.0 >= 13 { 128 } else { 64 };
        for symbol in sm120_symbols() {
            let parsed_entry = parsed.entry(&symbol);
            let entry = &parsed_entry.text;
            let parameters = ptx_parameters(entry, &symbol);
            // The stream-K bodies take the partial-slab and flag pointers
            // between the output and the tensor maps.
            let expected_parameters = if symbol.contains("_streamk_") { 7 } else { 5 };
            assert_eq!(
                parameters.matches(".param").count(),
                expected_parameters,
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
            let bundle_index = expected_parameters - 1;
            assert!(
                parameters.contains(&format!(
                    ".param .align 4 .b8 {symbol}_param_{bundle_index}[40]"
                )),
                "{requested} 40-byte parameter bundle for {symbol}: {parameters}"
            );

            let threads = if symbol.contains("_64x64_") {
                128
            } else if symbol.contains("_128x128_") && symbol.contains("_bk64_") {
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
            assert_eq!(
                entry.matches("bar.warp.sync").count(),
                3,
                "{requested} warp synchronization census for {symbol}"
            );
            if symbol.contains("_tn_") && symbol.contains("_bk32_") && symbol.contains("_128x64_") {
                let producer_sites = if symbol.contains("_s2_") { 3 } else { 4 };
                assert_eq!(
                    entry
                        .matches(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                        )
                        .count(),
                    3 * producer_sites,
                    "{requested} wide TN/BK32 TMA census for {symbol}",
                );
            }
            let (dtype, forbidden) = if symbol.ends_with("_bf16") {
                ("bf16", [F16_CORE, TF32_CORE])
            } else {
                ("f16", [BF16_CORE, TF32_CORE])
            };
            let mma = format!("mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32");
            assert!(
                compile_gate_has_exact_token(entry, &mma),
                "{requested}/{symbol} is missing exact core opcode {mma}"
            );
            assert_compile_gate_entry_excludes(requested, parsed_entry, &forbidden);
            let (mma_count, a_loads, b_loads) = if symbol.contains("_128x128_bk32_") {
                (32, 8, 8)
            } else if symbol.contains("_bk32_") {
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
        assert_eq!(sm120_tf32_symbols().len(), 30);
        for symbol in sm120_tf32_symbols() {
            let entry = parsed.entry(&symbol);
            assert_compile_gate_entry_tokens(
                requested,
                entry,
                &[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                    "mbarrier.init.shared::cta.b64",
                    "fence.mbarrier_init.release.cluster",
                    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                ],
            );
            // The exact-F32 routes multiply in scalar FMA and never round an
            // operand; the TF32 routes convert and run the tensor core.
            if symbol.contains("_tma_fma_v1_") {
                assert_compile_gate_entry_tokens(requested, entry, &["fma.rn.f32"]);
                assert_compile_gate_entry_excludes(
                    requested,
                    entry,
                    &[
                        BF16_CORE,
                        F16_CORE,
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                );
            } else {
                assert_compile_gate_entry_tokens(
                    requested,
                    entry,
                    &[
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                );
                assert_compile_gate_entry_excludes(requested, entry, &[BF16_CORE, F16_CORE]);
            }
            assert!(
                compile_gate_ptx_tokens(&entry.body)
                    .into_iter()
                    .any(|token| token.text.starts_with("st.global.")),
                "{requested}/{symbol} is missing st.global opcode"
            );
            let parameters = ptx_parameters(&entry.text, &symbol);
            // The stream-K and exact-F32 kernels carry their slab and flag
            // buffers ahead of the tensor maps; every other route keeps the
            // five-argument ABI.
            let expected_parameters =
                if symbol.ends_with("_pair_streamk") || symbol.contains("_tma_fma_v1_") {
                    7
                } else {
                    5
                };
            assert_eq!(parameters.matches(".param").count(), expected_parameters);
            assert_eq!(
                parameters
                    .matches(&format!(".param .align {map_alignment} .b8"))
                    .count(),
                2
            );
            assert_eq!(parameters.matches("[128]").count(), 2);
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

/// Diagnostic: compiles the scalar module with the production options and
/// writes the PTX under `MAMBA_RS_PTX_PROBE_DIR`, so repeated processes can
/// compare their images byte for byte./// The scalar kernels load their register fragments from shared memory as
/// explicit vectors. A scalar element loop (`regM[... + i] = As[... + i]`)
/// left the merging to the compiler's vectoriser, which chose differently
/// from one compile to the next and moved the module's artifact identity
/// (two PTX images of `gemm_bi_nt_slim` under one compile key).
#[test]
fn scalar_fragment_loads_are_explicit_vectors() {
    let source = include_str!("../kernels/gemm_bi_triad/scalar.cu");
    let mut offenders = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let body = line.trim_end_matches('\\').trim();
        if (body.starts_with("regM") || body.starts_with("regN"))
            && body.contains("] =")
            && body.ends_with("+ i];")
        {
            offenders.push(index + 1);
        }
        if body.ends_with("+ i];") && !body.contains("gemm_bi_scalar_load_fragment") {
            let previous = source
                .lines()
                .nth(index.saturating_sub(1))
                .unwrap_or_default()
                .trim();
            if previous.contains("regM[")
                || previous.contains("regN[")
                || previous.contains("_next[")
            {
                offenders.push(index + 1);
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "scalar.cu loads register fragments element by element at lines {offenders:?}"
    );
    assert!(
        source.matches("gemm_bi_scalar_load_fragment<").count() >= 40,
        "the explicit fragment loader must serve every scalar kernel"
    );
}
