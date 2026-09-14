//! The six scalar-NT copy-plan cohorts the dispatcher admits must carry
//! the identities the final six-cohort census measured on the board:
//! compile key, artifact, source, header manifest and NVRTC library domain,
//! per toolkit and state capacity. The receipts are the identity records those
//! runs wrote, kept as a fixture so the check needs neither a GPU nor raw logs.

use serde_json::Value;

const DISPATCH: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
const RECEIPTS: &str =
    include_str!("fixtures/ada_f32_nt_copyplan_siblings_identity_receipts.jsonl");
const COHORTS: [(&str, &str, &str); 6] = [
    (
        "cuda12.8-cap16",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda12.8-cap16.9j7vm0/identities.wwgemm",
        "9811c8f1d0ebef143bb9e8f3261ff7dba162ecc90efbce428938254fff936d7f",
    ),
    (
        "cuda12.8-cap64",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda12.8-cap64.2hGlK4/identities.wwgemm",
        "bcbbdcde0ddd8bbdb6d8550c047e3c632f1c089cd5bea6d53f69bee3c481cab7",
    ),
    (
        "cuda13.0-cap16",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda13.0-cap16.TNAtg8/identities.wwgemm",
        "53d08d7262e2cc6601542a43beb3aacfa0a4aa6b7f91a0f86fb1b7fd6a091228",
    ),
    (
        "cuda13.0-cap64",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda13.0-cap64.D7rL8r/identities.wwgemm",
        "8faa897edb5df9f52ad1e0f768a8c9de7f2e9c7d33c1ebf90cb429103c8e6754",
    ),
    (
        "cuda13.2-cap16",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda13.2-cap16.JZx9G2/identities.wwgemm",
        "e7fab57006cb2a070c4cac38bb75e021ebf570d6148e09185114f622d4944318",
    ),
    (
        "cuda13.2-cap64",
        "scratchpad/mamba-audit/combined-gemm-initial-cuda13.2-cap64.c3UFSh/identities.wwgemm",
        "8afe6f6cfb3983b8b39c132e8e1b52f54a534fc9f0ec4e7ead7e30ec3a636801",
    ),
];
const FIELDS: [&str; 5] = [
    "compile_key",
    "artifact_digest",
    "source_digest",
    "header_manifest_digest",
    "nvrtc_library_domain",
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    version: (i32, i32),
    fields: [[u8; 32]; 5],
}

#[derive(Clone, Debug)]
struct Composed {
    scalar: Identity,
    fixed: Identity,
}

#[derive(Clone, Debug)]
struct Measured {
    cohort: String,
    receipt_path: String,
    receipt_frame_sha256: String,
    composed: Composed,
}

fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    text.split_once(start).unwrap().1.split_once(end).unwrap().0
}

fn bytes(text: &str, marker: &str) -> [u8; 32] {
    let tail = text.split_once(marker).unwrap().1;
    let body = between(tail, "[", "]");
    body.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u8>().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn version(text: &str) -> (i32, i32) {
    let body = between(text.split_once("nvrtc_version:").unwrap().1, "(", ")");
    let (major, minor) = body.split_once(',').unwrap();
    (major.trim().parse().unwrap(), minor.trim().parse().unwrap())
}

fn json_bytes(record: &Value, field: &str) -> [u8; 32] {
    record[field]
        .as_array()
        .unwrap()
        .iter()
        .map(|byte| u8::try_from(byte.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn measured_identity(record: &Value) -> Identity {
    let version = record["nvrtc_version"].as_array().unwrap();
    Identity {
        version: (
            i32::try_from(version[0].as_i64().unwrap()).unwrap(),
            i32::try_from(version[1].as_i64().unwrap()).unwrap(),
        ),
        fields: [
            json_bytes(record, "compile_key"),
            json_bytes(record, "artifact_digest"),
            json_bytes(record, "source_digest"),
            json_bytes(record, "header_manifest_digest"),
            json_bytes(record, "nvrtc_library_domain"),
        ],
    }
}

fn measured(receipt: &str) -> Measured {
    let record = serde_json::from_str::<Value>(receipt).unwrap();
    assert_eq!(record["schema"], "MambaBiNtCopyPlanSiblingsFinalIdentityV2");
    Measured {
        cohort: record["cohort"].as_str().unwrap().to_owned(),
        receipt_path: record["receipt_path"].as_str().unwrap().to_owned(),
        receipt_frame_sha256: record["receipt_frame_sha256"].as_str().unwrap().to_owned(),
        composed: Composed {
            scalar: measured_identity(&record["scalar"]),
            fixed: measured_identity(&record["fixed"]),
        },
    }
}

fn frozen() -> Vec<Composed> {
    let fixed_source_cap16 = bytes(
        DISPATCH
            .split_once("const FIXED_COPYPLAN_SOURCE_DIGEST_CAP16: [u8; 32] =")
            .unwrap()
            .1,
        "",
    );
    let fixed_source_cap64 = bytes(
        DISPATCH
            .split_once("const FIXED_COPYPLAN_SOURCE_DIGEST_CAP64: [u8; 32] =")
            .unwrap()
            .1,
        "",
    );
    let scalar_source = bytes(
        DISPATCH
            .split_once("const TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST: [u8; 32] =")
            .unwrap()
            .1,
        "",
    );
    let fixed = between(
        DISPATCH,
        "const FIXED_COPYPLAN_EVIDENCE_COHORTS:",
        "struct ScalarTransposeQualificationIdentity",
    )
    .split("\n    FixedCopyPlanQualificationIdentity {")
    .skip(1)
    .map(|block| {
        let source_digest = match (
            block.contains("source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP16,"),
            block.contains("source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP64,"),
        ) {
            (true, false) => fixed_source_cap16,
            (false, true) => fixed_source_cap64,
            _ => panic!("Fixed cohort must bind exactly one capacity source digest"),
        };
        Identity {
            version: version(block),
            fields: [
                bytes(block, "compile_key:"),
                bytes(block, "artifact_digest:"),
                source_digest,
                bytes(block, "header_manifest_digest:"),
                bytes(block, "nvrtc_library_domain:"),
            ],
        }
    })
    .collect::<Vec<_>>();
    let scalar = between(
        DISPATCH,
        "const TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8:",
        "/// Frozen whole-pipeline identities",
    )
    .split("ScalarTransposeQualificationIdentity {")
    .skip(1)
    .map(|block| {
        assert!(block.contains("source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,"));
        Identity {
            version: version(block),
            fields: [
                bytes(block, "compile_key:"),
                bytes(block, "artifact_digest:"),
                scalar_source,
                bytes(block, "header_manifest_digest:"),
                bytes(block, "nvrtc_library_domain:"),
            ],
        }
    })
    .collect::<Vec<_>>();
    let composed = between(
        DISPATCH,
        "const NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:",
        "// The six measured complete-module pairs",
    )
    .split("NtFixedCopyPlanComposedQualificationIdentity {")
    .skip(1)
    .collect::<Vec<_>>();
    let aliases = [
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8", 0, 0),
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8", 0, 1),
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0", 1, 2),
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0", 1, 3),
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2", 2, 4),
        ("TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2", 2, 5),
    ];
    assert_eq!((fixed.len(), scalar.len(), composed.len()), (6, 3, 6));
    composed
        .into_iter()
        .zip(aliases)
        .map(|(block, (scalar_name, scalar_index, fixed_index))| {
            assert!(block.contains(&format!("scalar: {scalar_name},")));
            assert!(block.contains(&format!(
                "fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[{fixed_index}]"
            )));
            Composed {
                scalar: scalar[scalar_index].clone(),
                fixed: fixed[fixed_index].clone(),
            }
        })
        .collect()
}

fn mismatch(live: &Identity, frozen: &Identity) -> Option<&'static str> {
    if live.version != frozen.version {
        return Some("nvrtc_version");
    }
    FIELDS
        .into_iter()
        .zip(live.fields.iter().zip(frozen.fields.iter()))
        .find_map(|(name, (live, frozen))| (live != frozen).then_some(name))
}

#[test]
fn admitted_composed_cohorts_match_independent_live_identity_receipts() {
    let receipts = RECEIPTS.lines().map(measured).collect::<Vec<_>>();
    assert_eq!(receipts.len(), COHORTS.len());
    let mut dispatch = frozen();
    for (((cohort, path, frame_sha256), receipt), frozen) in
        COHORTS.into_iter().zip(&receipts).zip(dispatch.iter())
    {
        assert_eq!(receipt.cohort, cohort);
        assert_eq!(receipt.receipt_path, path);
        assert_eq!(receipt.receipt_frame_sha256, frame_sha256);
        assert_eq!(
            mismatch(&receipt.composed.scalar, &frozen.scalar),
            None,
            "{cohort} scalar"
        );
        assert_eq!(
            mismatch(&receipt.composed.fixed, &frozen.fixed),
            None,
            "{cohort} Fixed"
        );
    }

    for index in [0, 2, 4] {
        assert_eq!(dispatch[index].scalar, dispatch[index + 1].scalar);
        assert_ne!(dispatch[index].fixed, dispatch[index + 1].fixed);
        assert_ne!(
            dispatch[index].fixed.fields[2],
            dispatch[index + 1].fixed.fields[2],
            "{} Fixed source must remain capacity-specific",
            COHORTS[index].0
        );
    }

    // Mutation proof for the original CUDA12.8/13.0 admission bug: a
    // Fixed-module header cannot stand in for the scalar-module header.
    for cohort in 0..COHORTS.len() {
        assert_ne!(
            dispatch[cohort].scalar.fields[3],
            dispatch[cohort].fixed.fields[3]
        );
        dispatch[cohort].scalar.fields[3] = dispatch[cohort].fixed.fields[3];
        assert_eq!(
            mismatch(&receipts[cohort].composed.scalar, &dispatch[cohort].scalar),
            Some("header_manifest_digest"),
            "{}",
            COHORTS[cohort].0
        );
    }
}
