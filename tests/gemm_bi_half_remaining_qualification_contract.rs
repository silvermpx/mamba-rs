#![cfg(not(feature = "cuda"))]

#[path = "support/triad_half_remaining_qualification.rs"]
mod qualification;

use qualification::{HalfDtype, RetainedFamily, TnCase, TournamentFamily};

const CUDA_HARNESS_SOURCE: &str = include_str!("support/triad_half_remaining_qualification.rs");

#[test]
fn remaining_tn_plan_is_the_literal_six_cell_retained_map() {
    assert_eq!(
        qualification::tn_cases(),
        [
            TnCase::new(
                "d768_in_proj",
                (2_048, 768, 3_072),
                HalfDtype::F16,
                RetainedFamily::RegpipeVec2,
                576,
            ),
            TnCase::new(
                "d768_in_proj",
                (2_048, 768, 3_072),
                HalfDtype::Bf16,
                RetainedFamily::RegpipeVec2,
                576,
            ),
            TnCase::new(
                "d768_out_proj",
                (2_048, 1_536, 768),
                HalfDtype::F16,
                RetainedFamily::Compact,
                288,
            ),
            TnCase::new(
                "d768_out_proj",
                (2_048, 1_536, 768),
                HalfDtype::Bf16,
                RetainedFamily::RegpipeVec2,
                288,
            ),
            TnCase::new(
                "prism_in_proj",
                (4_621, 384, 1_928),
                HalfDtype::F16,
                RetainedFamily::Compact,
                186,
            ),
            TnCase::new(
                "prism_in_proj",
                (4_621, 384, 1_928),
                HalfDtype::Bf16,
                RetainedFamily::Compact,
                186,
            ),
        ]
    );
}

#[test]
fn only_d768_in_requests_the_three_way_incremental_tournament() {
    for case in qualification::tn_cases() {
        assert_eq!(
            case.direct_pair,
            case.cell == "d768_in_proj",
            "unexpected loser rerun for {case:?}"
        );
    }
    let nn = qualification::nn_case();
    assert_eq!(nn.dtype, HalfDtype::F16);
    assert_eq!(nn.dims, (2_048, 768, 3_072));
}

#[test]
fn d768_in_tournament_has_no_preselected_winner_and_covers_every_pair() {
    assert_eq!(
        qualification::d768_in_tournament_pairs(),
        [
            (TournamentFamily::Compact, TournamentFamily::Regpipe),
            (TournamentFamily::Compact, TournamentFamily::RegpipeVec2),
            (TournamentFamily::Regpipe, TournamentFamily::RegpipeVec2),
        ]
    );

    let compact_first = qualification::observed_tournament_order([0.90, 0.80, 0.90]).unwrap();
    assert_eq!(
        compact_first.order,
        [
            TournamentFamily::Compact,
            TournamentFamily::Regpipe,
            TournamentFamily::RegpipeVec2,
        ]
    );
    let vec2_first = qualification::observed_tournament_order([1.10, 1.20, 1.20]).unwrap();
    assert_eq!(vec2_first.order[0], TournamentFamily::RegpipeVec2);
    assert!(qualification::observed_tournament_order([1.0, f64::NAN, 1.0]).is_err());
}

#[test]
fn remaining_tn_candidate_resources_are_capped_at_the_frozen_128_registers() {
    assert_eq!(qualification::REMAINING_TN_REGISTER_CAP, 128);
}

#[test]
fn performance_stops_are_aggregated_after_all_cells() {
    assert!(qualification::finish_performance_stops(Vec::new()).is_ok());
    let error = qualification::finish_performance_stops(vec![
        "TN d768-out F16 retained loss".into(),
        "NN d768-in F16 retained loss".into(),
    ])
    .unwrap_err();
    assert!(error.contains("TN d768-out F16 retained loss"));
    assert!(error.contains("NN d768-in F16 retained loss"));
}

#[test]
fn actual_auto_identity_uses_the_supported_half_physical_qualification_path() {
    assert!(!CUDA_HARNESS_SOURCE.contains("record_eager_gemm_trace"));
    for required in [
        "PhysicalQualificationRoute::HalfPolicy",
        "qualify_physical_launch",
        "validate_actual_graph_against_evidence",
        "drop(qualified)",
    ] {
        assert!(
            CUDA_HARNESS_SOURCE.contains(required),
            "missing actual-AUTO evidence seam {required}"
        );
    }
}
