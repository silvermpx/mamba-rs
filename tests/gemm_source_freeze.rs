//! The GEMM modules' sources, pinned file by file, with no GPU and no CUDA
//! toolkit. The GEMM fast routes are admitted only when a module's whole
//! compiled identity equals a recorded, qualified cohort, so a byte changed
//! in any file the GEMM modules compose closes those routes on the qualified
//! boards while every other test stays green. This test names the file
//! instead. A change to the table is legitimate only together with a
//! requalification (identity_mint on CUDA 12.8, 13.0 and 13.2 and the
//! evidence runs). The Mamba-1 kernels and the kernels the families share
//! sit in their own modules and are free to change; the second test keeps
//! them out of the Inference module.

use std::path::{Path, PathBuf};

/// Every file under the GEMM source trees and the prelude they compose:
/// path, FNV-1a 64 of the bytes, length.
const GEMM_SOURCES: &[(&str, u64, usize)] = &[
    ("kernels/_typed_prelude.cuh", 0x672e6c6786769881, 2234),
    (
        "kernels/gemm_bi_inference/casts.cu",
        0x3bd3fd9fb5ee0497,
        2171,
    ),
    (
        "kernels/gemm_bi_inference/common.cuh",
        0x227cd7fb0826e3c7,
        6235,
    ),
    (
        "kernels/gemm_bi_inference/ffma.cu",
        0x9fb4b616d1c6f11f,
        27446,
    ),
    (
        "kernels/gemm_bi_inference/matvec.cu",
        0xe6885006d16447a0,
        14647,
    ),
    (
        "kernels/gemm_bi_inference/mma16.cu",
        0x393ce6d9d8c2ef3f,
        61656,
    ),
    (
        "kernels/gemm_bi_inference/sm100/tcgen05.cu",
        0x540c26ec94167896,
        22628,
    ),
    (
        "kernels/gemm_bi_inference/sm120/f32_n64_copyplan.cu",
        0x9c8d90942a662e46,
        51929,
    ),
    (
        "kernels/gemm_bi_inference/sm120/f32_n64_sliced.cu",
        0xd68940d7cc8cd02f,
        23169,
    ),
    (
        "kernels/gemm_bi_inference/sm120/f32_postbias.cu",
        0xe9f6ab49d401ecae,
        28069,
    ),
    (
        "kernels/gemm_bi_inference/sm120/tf32.cu",
        0xe16f1efd83ab0c87,
        17574,
    ),
    (
        "kernels/gemm_bi_inference/sm120/tma.cu",
        0x57f600f0349222ba,
        31685,
    ),
    (
        "kernels/gemm_bi_inference/sm80/cells_common.cuh",
        0xe48e7073b86a06a0,
        4653,
    ),
    (
        "kernels/gemm_bi_inference/sm80/cells_f32_simt.cu",
        0x07f1b02da4e8c39f,
        11119,
    ),
    (
        "kernels/gemm_bi_inference/sm80/cells_half_mma.cu",
        0xde8a91ce4e240505,
        16399,
    ),
    (
        "kernels/gemm_bi_inference/sm80/cells_tf32_mma.cu",
        0xf851f50facadf486,
        11773,
    ),
    (
        "kernels/gemm_bi_inference/sm80/f32_m128n64_tail_copyplan.cu",
        0x510dfb97f5c5954d,
        17713,
    ),
    (
        "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu",
        0x176e376786e6a5d9,
        17060,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_f32out_s3.cu",
        0x54afef3f1463d02e,
        10891,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_n64.cu",
        0x6ccbaa2e7602f457,
        12685,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_pipeline.cu",
        0xe312c737dda81acc,
        17552,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_s3.cu",
        0x433908f5d46c5d9a,
        7984,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_swizzle.cu",
        0x884df51d64550af4,
        16737,
    ),
    (
        "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh",
        0xa99d85eb09e44507,
        1891,
    ),
    (
        "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu",
        0x159ae2d02e730d9b,
        14259,
    ),
    (
        "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu",
        0x2b83a258ef328b4f,
        19223,
    ),
    (
        "kernels/gemm_bi_inference/sm90a/wgmma.cu",
        0x6bdeee2e276b8e1e,
        17502,
    ),
    (
        "kernels/gemm_bi_inference/tcw64.cu",
        0xcee10a1407998221,
        42719,
    ),
    (
        "kernels/gemm_bi_inference/tf32.cu",
        0xc54c3234ee760a3a,
        15639,
    ),
    (
        "kernels/gemm_bi_inference/wmma_legacy.cu",
        0xfa399215c48d93df,
        10972,
    ),
    ("kernels/gemm_bi_triad/common.cuh", 0xd06942d6c5b44868, 1000),
    (
        "kernels/gemm_bi_triad/contract.cuh",
        0xaab241d080b4217d,
        5104,
    ),
    (
        "kernels/gemm_bi_triad/epilogue.cuh",
        0xa9bc47a2e1615149,
        1147,
    ),
    ("kernels/gemm_bi_triad/mma16.cuh", 0x5407785dd217d044, 1097),
    (
        "kernels/gemm_bi_triad/scalar.cu",
        0xab0b7c5248d47c5e,
        289900,
    ),
    (
        "kernels/gemm_bi_triad/scalar_nn_m64n64.cu",
        0x0653a15ce38b0a16,
        32825,
    ),
    (
        "kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu",
        0x32d9fd3695d9cd6d,
        8227,
    ),
    (
        "kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu",
        0x953a1a12cbf5ed2f,
        1234,
    ),
    (
        "kernels/gemm_bi_triad/scalar_nt_m2n16.cu",
        0x0d96e4a2964c1e81,
        4268,
    ),
    (
        "kernels/gemm_bi_triad/scalar_tn_m16n16.cu",
        0xe2407ce1a016096a,
        6876,
    ),
    (
        "kernels/gemm_bi_triad/sm100/tcgen05.cu",
        0x6a7169e31e278a61,
        54978,
    ),
    (
        "kernels/gemm_bi_triad/sm120/exact.cu",
        0x5dece8d8ffede529,
        26501,
    ),
    (
        "kernels/gemm_bi_triad/sm120/tma.cu",
        0xa87b7aba7276dd2f,
        116423,
    ),
    (
        "kernels/gemm_bi_triad/sm80/exact_f32.cu",
        0x6e94fc50d1464174,
        30173,
    ),
    (
        "kernels/gemm_bi_triad/sm80/exact_f32_d128.cu",
        0xc4274641cd08ee8,
        15975,
    ),
    (
        "kernels/gemm_bi_triad/sm80/half.cu",
        0x6cd9fece9b715ec7,
        88944,
    ),
    (
        "kernels/gemm_bi_triad/sm80/half_d128.cu",
        0x96fdd311ed300461,
        12050,
    ),
    (
        "kernels/gemm_bi_triad/sm80/half_relay.cu",
        0x9c41e046b98d5759,
        18657,
    ),
    (
        "kernels/gemm_bi_triad/sm80/half_small.cu",
        0x736b602633ef2e9e,
        21186,
    ),
    (
        "kernels/gemm_bi_triad/sm80/half_tn.cu",
        0xb4d99185f5af9250,
        49267,
    ),
    (
        "kernels/gemm_bi_triad/sm80/mma.cu",
        0x7312ebcc8866754d,
        189602,
    ),
    (
        "kernels/gemm_bi_triad/sm80/nt_compact.cuh",
        0x0558f7e6d49c448e,
        354,
    ),
    (
        "kernels/gemm_bi_triad/sm80/streamk.cu",
        0x01dd94862f5620dd,
        28344,
    ),
    (
        "kernels/gemm_bi_triad/sm80/exact_thin.cu",
        0x2c5402b94dc4f606,
        25900,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_joint.cu",
        0x2825166316d1be5e,
        98168,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_joint_primitives.cuh",
        0x0247c603860dc16f,
        828,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_rna.cu",
        0x6184ec1b835c4db4,
        75644,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_wide.cu",
        0x89b6365073412c5d,
        17796,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_wide_nn.cu",
        0x8af93f65b031a90b,
        18724,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_wide_nt.cu",
        0xf572e42283d0f654,
        17239,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_wide_nt_epi.cu",
        0x8302f76792c0f253,
        22165,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tf32_wide_tn.cu",
        0x385301a25ada0bf2,
        15127,
    ),
    (
        "kernels/gemm_bi_triad/sm80/tn_splitk.cu",
        0xd31ad14da759aa19,
        20813,
    ),
    (
        "kernels/gemm_bi_triad/sm90a/wgmma.cu",
        0x568992c0aef42b56,
        36146,
    ),
];

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn files_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read a GEMM source directory") {
        let path = entry.expect("read a directory entry").path();
        if path.is_dir() {
            files_under(root, &path, out);
        } else {
            out.push(
                path.strip_prefix(root)
                    .expect("under the repository")
                    .to_path_buf(),
            );
        }
    }
}

#[test]
fn gemm_module_sources_are_frozen() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found = vec![PathBuf::from("kernels/_typed_prelude.cuh")];
    for dir in ["kernels/gemm_bi_inference", "kernels/gemm_bi_triad"] {
        files_under(root, &root.join(dir), &mut found);
    }
    found.sort();
    let mut problems = Vec::new();
    for path in &found {
        let name = path.to_string_lossy().replace('\\', "/");
        let bytes = std::fs::read(root.join(path)).expect("read a GEMM source");
        let observed = (fnv1a64(&bytes), bytes.len());
        match GEMM_SOURCES.iter().find(|(pinned, _, _)| *pinned == name) {
            None => problems.push(format!(
                "{name} is new: 0x{:016x}, {}",
                observed.0, observed.1
            )),
            Some((_, hash, len)) if (*hash, *len) != observed => problems.push(format!(
                "{name} moved: expected 0x{hash:016x}, {len}; observed 0x{:016x}, {}",
                observed.0, observed.1
            )),
            Some(_) => {}
        }
    }
    for (pinned, _, _) in GEMM_SOURCES {
        if !found
            .iter()
            .any(|path| path.to_string_lossy().replace('\\', "/") == *pinned)
        {
            problems.push(format!("{pinned} is gone"));
        }
    }
    assert!(
        problems.is_empty(),
        "GEMM identity moved: the qualified fast routes now decline on the boards whose \
         cohorts compose these files. Revert, or requalify and update the table:\n{}",
        problems.join("\n")
    );
}

/// The include_str! paths of one fragment table in modules.rs, relative to
/// the crate root whatever directory modules.rs sits in.
fn table_files(modules: &str, table: &str) -> Vec<String> {
    let start = modules
        .find(&format!("const {table}: &[SourceFragment] = &["))
        .unwrap_or_else(|| panic!("{table} is defined in modules.rs"));
    let body = &modules[start..];
    let body = &body[..body.find("\n];").expect("the table ends")];
    body.match_indices("include_str!(\"")
        .map(|(at, prefix)| {
            let rest = &body[at + prefix.len()..];
            let mut path = &rest[..rest.find('"').expect("the path ends")];
            while let Some(up) = path.strip_prefix("../") {
                path = up;
            }
            path.to_string()
        })
        .collect()
}

#[test]
fn inference_module_composes_no_model_kernel() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let modules = std::fs::read_to_string(root.join("src/gpu/gemm_bi_triad/modules.rs"))
        .expect("read modules.rs");
    let inference = table_files(&modules, "INFERENCE_SOURCE_FRAGMENTS");
    assert!(
        !inference.is_empty(),
        "INFERENCE_SOURCE_FRAGMENTS composes its kernels"
    );
    for path in inference {
        assert!(
            path.starts_with("kernels/gemm_bi_inference/") || path == "kernels/_typed_prelude.cuh",
            "the Inference module composes {path}, which is not a GEMM source"
        );
    }
    for table in ["MAMBA1_SOURCE_FRAGMENTS", "SHARED_SOURCE_FRAGMENTS"] {
        let files = table_files(&modules, table);
        assert!(!files.is_empty(), "{table} composes its kernels");
        for path in files {
            assert!(
                !path.starts_with("kernels/gemm_bi_"),
                "{table} composes the GEMM source {path}"
            );
        }
    }
}
