//! The three scalar-NT copy-plan cohorts the dispatcher admits must carry
//! the identities the module loader computes on Ada for each retained
//! toolkit: compile key, artifact, source, header manifest and NVRTC library
//! domain of both the scalar module and the Inference module. The fixture is
//! the `identity_mint` output for those two modules on sm_89 under CUDA
//! 12.8, 13.0 and 13.2, so the check needs neither a GPU nor raw logs.

use serde_json::Value;

const DISPATCH: &str = include_str!("../src/gpu/gemm_bi_triad/dispatch.rs");
const MINTED: &str = include_str!("fixtures/ada_f32_nt_copyplan_mint_identities.jsonl");
const TOOLKITS: [(i32, i32); 3] = [(12, 8), (13, 0), (13, 2)];
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
    inference: Identity,
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

fn hex_bytes(record: &Value, field: &str) -> [u8; 32] {
    let hex = record[field].as_str().unwrap();
    assert_eq!(hex.len(), 64, "{field}");
    std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
}

/// The minted identity of one module, in the field order the dispatcher pins.
fn minted_identity(record: &Value) -> Identity {
    // The loader keys an artifact by its invocation digest.
    assert_eq!(record["compile_key"], record["invocation"]);
    let version = record["nvrtc"].as_array().unwrap();
    Identity {
        version: (
            i32::try_from(version[0].as_i64().unwrap()).unwrap(),
            i32::try_from(version[1].as_i64().unwrap()).unwrap(),
        ),
        fields: [
            hex_bytes(record, "compile_key"),
            hex_bytes(record, "artifact"),
            hex_bytes(record, "source"),
            hex_bytes(record, "header"),
            hex_bytes(record, "library"),
        ],
    }
}

fn minted() -> Vec<Composed> {
    let records = MINTED
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 2 * TOOLKITS.len());
    let find = |module: &str, toolkit: (i32, i32)| {
        let found = records
            .iter()
            .filter(|record| {
                record["module"] == module
                    && record["target"] == "sm_89"
                    && record["nvrtc"] == serde_json::json!([toolkit.0, toolkit.1])
            })
            .collect::<Vec<_>>();
        assert_eq!(found.len(), 1, "{module} {toolkit:?}");
        minted_identity(found[0])
    };
    TOOLKITS
        .into_iter()
        .map(|toolkit| {
            let inference = find("Inference", toolkit);
            Composed {
                scalar: find("TriadScalar", toolkit),
                inference,
            }
        })
        .collect()
}

fn frozen() -> Vec<Composed> {
    let inference_source = bytes(
        DISPATCH
            .split_once("const INFERENCE_COPYPLAN_SOURCE_DIGEST: [u8; 32] =")
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
    let inference = between(
        DISPATCH,
        "const INFERENCE_COPYPLAN_EVIDENCE_COHORTS:",
        "struct ScalarTransposeQualificationIdentity",
    )
    .split("\n    InferenceCopyPlanQualificationIdentity {")
    .skip(1)
    .map(|block| {
        assert!(block.contains("source_digest: INFERENCE_COPYPLAN_SOURCE_DIGEST,"));
        Identity {
            version: version(block),
            fields: [
                bytes(block, "compile_key:"),
                bytes(block, "artifact_digest:"),
                inference_source,
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
        "const NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:",
        "// The three measured complete-module pairs",
    )
    .split("NtInferenceCopyPlanComposedQualificationIdentity {")
    .skip(1)
    .collect::<Vec<_>>();
    let aliases = [
        "TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8",
        "TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0",
        "TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2",
    ];
    assert_eq!((inference.len(), scalar.len(), composed.len()), (3, 3, 3));
    composed
        .into_iter()
        .zip(aliases)
        .enumerate()
        .map(|(index, (block, scalar_name))| {
            assert!(block.contains(&format!("scalar: {scalar_name},")));
            assert!(block.contains(&format!(
                "inference: INFERENCE_COPYPLAN_EVIDENCE_COHORTS[{index}]"
            )));
            Composed {
                scalar: scalar[index].clone(),
                inference: inference[index].clone(),
            }
        })
        .collect()
}

fn mismatch(minted: &Identity, frozen: &Identity) -> Option<&'static str> {
    if minted.version != frozen.version {
        return Some("nvrtc_version");
    }
    FIELDS
        .into_iter()
        .zip(minted.fields.iter().zip(frozen.fields.iter()))
        .find_map(|(name, (minted, frozen))| (minted != frozen).then_some(name))
}

#[test]
fn admitted_composed_cohorts_match_the_minted_module_identities() {
    let minted = minted();
    let mut dispatch = frozen();
    for ((toolkit, minted), frozen) in TOOLKITS.into_iter().zip(&minted).zip(dispatch.iter()) {
        assert_eq!(
            mismatch(&minted.scalar, &frozen.scalar),
            None,
            "CUDA {toolkit:?} scalar"
        );
        assert_eq!(
            mismatch(&minted.inference, &frozen.inference),
            None,
            "CUDA {toolkit:?} Inference"
        );
    }

    // The Inference module takes no state capacity: one source serves every
    // toolkit, while the compiled artifact is the toolkit's own.
    for pair in dispatch.windows(2) {
        assert_eq!(pair[0].inference.fields[2], pair[1].inference.fields[2]);
        assert_ne!(pair[0].inference.fields[0], pair[1].inference.fields[0]);
    }

    // Mutation proof for the original CUDA12.8/13.0 admission bug: an
    // Inference-module header cannot stand in for the scalar-module header.
    for (index, toolkit) in TOOLKITS.into_iter().enumerate() {
        assert_ne!(
            dispatch[index].scalar.fields[3],
            dispatch[index].inference.fields[3]
        );
        dispatch[index].scalar.fields[3] = dispatch[index].inference.fields[3];
        assert_eq!(
            mismatch(&minted[index].scalar, &dispatch[index].scalar),
            Some("header_manifest_digest"),
            "CUDA {toolkit:?}"
        );
    }
}
