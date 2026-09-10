//! The three scalar-NT copy-plan cohorts the dispatcher admits must carry
//! the identities the pre-admission qualification measured on the board:
//! compile key, artifact, source, header manifest and NVRTC library domain,
//! per toolkit. The receipts are the identity records those runs wrote,
//! kept as a fixture so the check needs neither a GPU nor the raw logs.

use serde_json::Value;

const DISPATCH: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
const RECEIPTS: &str =
    include_str!("fixtures/ada_f32_nt_copyplan_siblings_identity_receipts.jsonl");
const TOOLKITS: [&str; 3] = ["CUDA12.8", "CUDA13.0", "CUDA13.2"];
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

fn identity(compiler: &str, artifact: &str) -> Identity {
    let compile_key = bytes(artifact, "compile_key:");
    assert_eq!(compile_key, bytes(compiler, "invocation_digest:"));
    Identity {
        version: version(compiler),
        fields: [
            compile_key,
            bytes(artifact, "artifact_digest:"),
            bytes(compiler, "source_digest:"),
            bytes(compiler, "header_manifest_digest:"),
            bytes(compiler, "nvrtc_library_domain:"),
        ],
    }
}

fn live(receipt: &str) -> Composed {
    let record = serde_json::from_str::<Value>(receipt).unwrap();
    assert_eq!(
        record["schema"],
        "MambaBiNtCopyPlanSiblingsPreAdmissionIdentityV1"
    );
    let artifacts = record["artifacts"].as_str().unwrap();
    Composed {
        scalar: identity(
            record["scalar_compiler"].as_str().unwrap(),
            between(
                artifacts,
                "triad_scalar: ArtifactIdentity {",
                "}, triad_sm80:",
            ),
        ),
        fixed: identity(
            record["fixed_compiler"].as_str().unwrap(),
            between(artifacts, "fixed: ArtifactIdentity {", "}, triad_scalar:"),
        ),
    }
}

fn frozen() -> Vec<Composed> {
    let fixed_source = bytes(
        DISPATCH
            .split_once("const FIXED_COPYPLAN_SOURCE_DIGEST: [u8; 32] =")
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
        assert!(block.contains("source_digest: FIXED_COPYPLAN_SOURCE_DIGEST,"));
        Identity {
            version: version(block),
            fields: [
                bytes(block, "compile_key:"),
                bytes(block, "artifact_digest:"),
                fixed_source,
                bytes(block, "header_manifest_digest:"),
                bytes(block, "nvrtc_library_domain:"),
            ],
        }
    })
    .collect::<Vec<_>>();
    let scalar_blocks = between(
        DISPATCH,
        "const NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:",
        "// The three frozen candidates passed the complete pre-admission qualification",
    )
    .split("scalar: ScalarTransposeQualificationIdentity {")
    .skip(1)
    .collect::<Vec<_>>();
    assert_eq!((fixed.len(), scalar_blocks.len()), (3, 3));
    scalar_blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| {
            assert!(block.contains("source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,"));
            assert!(block.contains(&format!("fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[{index}]")));
            Composed {
                scalar: Identity {
                    version: version(block),
                    fields: [
                        bytes(block, "compile_key:"),
                        bytes(block, "artifact_digest:"),
                        scalar_source,
                        bytes(block, "header_manifest_digest:"),
                        bytes(block, "nvrtc_library_domain:"),
                    ],
                },
                fixed: fixed[index].clone(),
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
    let receipts = RECEIPTS.lines().collect::<Vec<_>>();
    assert_eq!(receipts.len(), TOOLKITS.len());
    let mut dispatch = frozen();
    for ((toolkit, receipt), frozen) in TOOLKITS.into_iter().zip(&receipts).zip(dispatch.iter()) {
        let live = live(receipt);
        assert_eq!(
            mismatch(&live.scalar, &frozen.scalar),
            None,
            "{toolkit} scalar"
        );
        assert_eq!(
            mismatch(&live.fixed, &frozen.fixed),
            None,
            "{toolkit} Fixed"
        );
    }

    // Mutation proof for the original CUDA12.8/13.0 admission bug: a
    // Fixed-module header cannot stand in for the scalar-module header.
    for cohort in [0, 1] {
        dispatch[cohort].scalar.fields[3] = dispatch[cohort].fixed.fields[3];
        assert_eq!(
            mismatch(&live(receipts[cohort]).scalar, &dispatch[cohort].scalar),
            Some("header_manifest_digest")
        );
    }
}
