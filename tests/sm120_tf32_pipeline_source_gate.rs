//! Host-only regression gates for the SM120 TF32 consumer pipeline.

const SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120/tma.cu");

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing SM120 TF32 function signature {signature}"));
    let tail = &source[start..];
    let end = tail[1..]
        .find("\ntemplate <")
        .map(|offset| offset + 1)
        .unwrap_or(tail.len());
    &tail[..end]
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect()
}

fn braced_block_from<'a>(source: &'a str, marker: &str) -> Option<&'a str> {
    let start = source.find(marker)?;
    let opening = start + source[start..].find('{')?;
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[opening..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[start..=opening + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn preload_contract_accepts(source: &str) -> bool {
    let load_issue = compact(function_body(source, "void sm120_tf32_load_issue("));
    let issue_stage = compact(function_body(source, "void sm120_tf32_issue_stage("));
    let initial = "sm120_tf32_load_issue<Op,M,N,Stages>(storage,stage,warp_m,warp_n,0,a_fragments[0],b_fragments[0]);";
    let issue_loop = "for(intissue=0;issue<4;++issue){";
    let guard = "if(issue+1<4){";
    let preload = "sm120_tf32_load_issue<Op,M,N,Stages>(storage,stage,warp_m,warp_n,k8+8,a_fragments[next],b_fragments[next]);";
    let m_loop = "for(intm_atom=0;m_atom<2;++m_atom){";
    let n_loop = "for(intn_atom=0;n_atom<4;++n_atom){";
    let mma = "sm120_tf32_mma_m16n8k8(accumulator[m_atom][n_atom],a_fragments[current][m_atom],b_fragments[current][n_atom]);";
    let ordered = issue_stage
        .find(initial)
        .zip(issue_stage.find(issue_loop))
        .zip(issue_stage.find(guard))
        .zip(issue_stage.find(preload))
        .zip(issue_stage.find(m_loop))
        .zip(issue_stage.find(n_loop))
        .zip(issue_stage.find(mma))
        .is_some_and(
            |(
                (
                    (
                        (((initial_position, loop_position), guard_position), preload_position),
                        m_position,
                    ),
                    n_position,
                ),
                mma_position,
            )| {
                initial_position < loop_position
                    && loop_position < guard_position
                    && guard_position < preload_position
                    && preload_position < m_position
                    && m_position < n_position
                    && n_position < mma_position
            },
        );
    let complete_a = (0..4).all(|element| {
        load_issue
            .matches(&format!("a_fragments[m_atom][{element}]="))
            .count()
            == 1
    });
    let complete_b = (0..2).all(|element| {
        load_issue
            .matches(&format!("b_fragments[n_atom][{element}]="))
            .count()
            == 1
    });

    ordered
        && issue_stage.matches(initial).count() == 1
        && issue_stage.matches(issue_loop).count() == 1
        && issue_stage.matches(guard).count() == 1
        && issue_stage.matches(preload).count() == 1
        && issue_stage.matches(mma).count() == 1
        && issue_stage.contains("unsigneda_fragments[2][2][4];")
        && issue_stage.contains("unsignedb_fragments[2][4][2];")
        && issue_stage.contains("intk8=issue*8;")
        && issue_stage.contains("intcurrent=issue&1;")
        && issue_stage.contains("intnext=current^1;")
        && load_issue.matches(m_loop).count() == 1
        && load_issue.matches(n_loop).count() == 1
        && complete_a
        && complete_b
}

fn sync_contract_accepts(source: &str) -> bool {
    let kernel = compact(function_body(source, "void sm120_tf32_kernel("));
    let retained = "ifconstexpr(Stages==2)sm120_sync_warp();";
    let initial_produce = "sm120_tf32_produce_stage<Op,M,N,Stages>(stage_context,tile);";
    let tile_loop = "for(inttile=0;tile<tile_count;++tile){";
    let issue = "sm120_tf32_issue_stage<Op,M,N,Stages>(storage,stage,warp_m,warp_n,accumulator);";
    let arrive = "sm120_arrive_empty(empty_base+stage*8);";
    let refill_produce = "sm120_tf32_produce_stage<Op,M,N,Stages>(stage_context,refill);";
    let Some(tile_loop_position) = kernel.find(tile_loop) else {
        return false;
    };
    let Some(tile_block) = braced_block_from(&kernel, tile_loop) else {
        return false;
    };
    let sync_positions: Vec<_> = kernel
        .match_indices(retained)
        .map(|(index, _)| index)
        .collect();
    if sync_positions.len() != 2 || kernel.matches("sm120_sync_warp();").count() != 2 {
        return false;
    }
    let initial_ordering = kernel.find(initial_produce).is_some_and(|produce| {
        produce < sync_positions[0] && sync_positions[0] < tile_loop_position
    });
    let consumer_ordering = tile_block
        .find(issue)
        .zip(tile_block.find(arrive))
        .zip(tile_block.find(refill_produce))
        .zip(tile_block.find(retained))
        .is_some_and(|(((issue, arrive), refill), sync)| {
            issue < arrive
                && arrive < refill
                && refill < sync
                && !tile_block[issue..arrive].contains("sm120_sync_warp();")
        });

    initial_ordering
        && consumer_ordering
        && kernel[..tile_loop_position].matches(retained).count() == 1
        && tile_block.matches(retained).count() == 1
}

#[test]
fn tf32_preloads_the_next_fragments_before_the_current_mma() {
    assert!(
        preload_contract_accepts(SM120_SOURCE),
        "the TF32 issue-stage preload contract must hold"
    );
}

#[test]
fn tf32_keeps_only_the_two_s2_producer_ordering_syncs() {
    assert!(
        sync_contract_accepts(SM120_SOURCE),
        "the TF32 kernel must retain only its initial and refill S2 ordering syncs"
    );
}

#[test]
fn tf32_tn_s4_pair_store_is_narrow_and_tail_safe() {
    let store_pair = compact(function_body(SM120_SOURCE, "void sm120_tf32_store_pair("));
    let pair_kernel = compact(function_body(SM120_SOURCE, "void sm120_tf32_pair_kernel("));
    let generic_kernel = compact(function_body(SM120_SOURCE, "void sm120_tf32_kernel("));

    assert!(SM120_SOURCE.contains("tn_sm120_tma_mma_tf32_m64n128_bk32_s4_pair"));
    assert!(SM120_SOURCE.contains("SM120_DEFINE_TF32_PAIR_KERNEL(tn_sm120"));
    assert!(pair_kernel.contains("boolfull_tile=output_row+M<=rows"));
    assert!(pair_kernel.contains("sm120_tf32_store_pair<Op>("));
    assert!(!generic_kernel.contains("sm120_tf32_store_pair<Op>("));
    assert!(store_pair.contains("float2old_pair=*reinterpret_cast<constfloat2*>(destination);"));
    assert!(
        store_pair.contains("*reinterpret_cast<float2*>(destination)=make_float2(first,second);")
    );
    assert_eq!(store_pair.matches("sm120_tf32_store<Op>(").count(), 2);
    assert!(!store_pair.contains("atomic"));
}

#[test]
fn tf32_preload_gate_rejects_broken_fragment_schedules() {
    let mutations = [
        (
            "missing initial slot-zero preload",
            SM120_SOURCE.replacen(
                "storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);",
                "storage, stage, warp_m, warp_n, 0, a_fragments[1], b_fragments[1]);",
                1,
            ),
        ),
        (
            "wrong issue count",
            SM120_SOURCE.replacen("issue < 4", "issue < 3", 1),
        ),
        (
            "missing final-preload guard",
            SM120_SOURCE.replacen("issue + 1 < 4", "issue < 4", 1),
        ),
        (
            "preload overwrites the current slot",
            SM120_SOURCE.replacen(
                "a_fragments[next], b_fragments[next]",
                "a_fragments[current], b_fragments[current]",
                1,
            ),
        ),
        (
            "incomplete A fragment",
            SM120_SOURCE.replacen("a_fragments[m_atom][3] =", "a_fragments[m_atom][2] =", 1),
        ),
        (
            "incomplete B fragment",
            SM120_SOURCE.replacen("b_fragments[n_atom][1] =", "b_fragments[n_atom][0] =", 1),
        ),
    ];

    for (name, source) in mutations {
        assert!(
            !preload_contract_accepts(&source),
            "the TF32 preload source gate accepted mutation: {name}"
        );
    }
}

#[test]
fn tf32_sync_gate_rejects_relocated_ordering_syncs() {
    let retained = "if constexpr (Stages == 2) sm120_sync_warp();";
    let without_initial = SM120_SOURCE.replacen(retained, "", 1);
    let initial_produce = "sm120_tf32_produce_stage<Op, M, N, Stages>(stage_context, tile);";
    let before_initial_produce = without_initial.replacen(
        initial_produce,
        &format!("{retained}\n                {initial_produce}"),
        1,
    );

    let (before_refill_sync, after_refill_sync) = SM120_SOURCE
        .rsplit_once(retained)
        .expect("refill S2 ordering sync");
    let without_refill = format!("{before_refill_sync}{after_refill_sync}");
    let issue = "storage, stage, warp_m, warp_n, accumulator);";
    let after_issue = without_refill.replacen(
        "storage, stage, warp_m, warp_n, accumulator);",
        &format!("{issue}\n        {retained}"),
        1,
    );

    for (name, source) in [
        ("before initial production", before_initial_produce),
        ("between issue and empty arrival", after_issue),
    ] {
        assert!(
            !sync_contract_accepts(&source),
            "the TF32 sync source gate accepted a sync {name}"
        );
    }
}
