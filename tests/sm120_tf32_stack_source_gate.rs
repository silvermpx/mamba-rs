//! Host-only regression gate for the SM120 TF32 issue-loop stack frame.

const SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120/tma.cu");

fn issue_stage(source: &str) -> &str {
    let start = source
        .find("void sm120_tf32_issue_stage(")
        .expect("SM120 TF32 issue stage");
    let tail = &source[start..];
    let end = tail[1..]
        .find("\ntemplate <")
        .map(|offset| offset + 1)
        .unwrap_or(tail.len());
    &tail[..end]
}

fn declares_local_int_array(source: &str) -> bool {
    let Some(opening) = source.find('{') else {
        return false;
    };
    let body = &source[opening + 1..];
    let bytes = body.as_bytes();
    let is_identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';

    for offset in 0..bytes.len().saturating_sub(2) {
        if &bytes[offset..offset + 3] != b"int"
            || offset
                .checked_sub(1)
                .is_some_and(|previous| is_identifier(bytes[previous]))
            || bytes
                .get(offset + 3)
                .is_some_and(|next| is_identifier(*next))
        {
            continue;
        }
        let declaration = &body[offset + 3..];
        let end = declaration.find(';').unwrap_or(declaration.len());
        let declaration = &declaration[..end];
        let left_hand_side = declaration
            .split_once('=')
            .map_or(declaration, |(left, _)| left);
        if left_hand_side.contains('[') {
            return true;
        }
    }
    false
}

fn stack_contract_accepts(source: &str) -> bool {
    let issue_stage = issue_stage(source);
    issue_stage.contains("int k8 = issue * 8;") && !declares_local_int_array(issue_stage)
}

#[test]
fn sm120_tf32_issue_loop_computes_k8_without_an_indexed_local_array() {
    assert!(
        stack_contract_accepts(SM120_SOURCE),
        "SM120 TF32 K8 offsets must be direct scalars without a local integer array"
    );
}

#[test]
fn sm120_tf32_stack_gate_rejects_any_local_integer_array_name() {
    let mutation = SM120_SOURCE.replacen(
        "unsigned a_fragments[2][2][4];",
        "int arbitrary_offsets[4];\n    unsigned a_fragments[2][2][4];",
        1,
    );

    assert!(
        !stack_contract_accepts(&mutation),
        "the stack source gate accepted a differently named local integer array"
    );
}
