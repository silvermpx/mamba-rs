#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/gemm_bi_tf32_contract_toolkit.rs"]
mod toolkit;
use toolkit::*;

const SCALAR_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar.cu");
const COMMON_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/common.cuh");
const SM80_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm80/mma.cu");
const SM80_TN_SPLITK_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm80/tn_splitk.cu");
const SM90A_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm90a/wgmma.cu");
const SM100_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm100/tcgen05.cu");
const SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120/tma.cu");
const CONTEXT_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/context.rs");
const DEVICE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/device.rs");
const IDENTITY_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/kernel_identity.rs");
const CONTRACT_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs");
const DISPATCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
const LAUNCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");
const QUALIFICATION_SOURCE: &str =
    include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs");
const BLAS_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/blas.rs");
const GRAPH_CAPTURE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/graph_capture.rs");
const TRAINING_GRAPH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/training_graph.rs");
const TRAINER_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/trainer.rs");
const INFERENCE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/inference.rs");
const PREFILL_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/prefill.rs");
const MAMBA3_TRAINING_GRAPH_SOURCE: &str = include_str!("../src/mamba3_siso/gpu/training_graph.rs");
const MAMBA3_TRAINER_SOURCE: &str = include_str!("../src/mamba3_siso/gpu/trainer.rs");
const MAMBA3_INFERENCE_SOURCE: &str = include_str!("../src/mamba3_siso/gpu/inference.rs");
const MAMBA3_PREFILL_SOURCE: &str = include_str!("../src/mamba3_siso/gpu/prefill.rs");
const ARCH_GATE_SOURCE: &str = include_str!("arch_compile_gates.rs");
const KERNEL_IDENTITY_CUDA_SOURCE: &str = include_str!("kernel_identity_cuda.rs");
const TF32_CONTRACT_TEST_SOURCE: &str = include_str!("gemm_bi_tf32_contract.rs");

fn assert_code_excludes_all(source: &str, forbidden: &[&str], contract: &str) {
    for needle in forbidden {
        assert!(
            !source.contains(needle),
            "{contract} must exclude {needle:?}"
        );
    }
}

#[test]
fn tc128_nn_pair_store_is_decided_per_row_for_odd_ldc() {
    let body = braced_scope_after(SM80_SOURCE, "void nn_tc_##SUFFIX");
    assert!(
        body.contains("bool packed_epilogue = is_aligned_4(C);"),
        "TC128 NN must keep pair stores available for aligned rows of odd-ldc outputs"
    );
    assert!(
        body.contains("packed_epilogue && c0 + 1 < N &&") && body.contains("is_aligned_4(dst)"),
        "TC128 NN must validate each destination row before a pair store"
    );
}

fn braced_scope_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let mask = source_mask(source);
    let marker_offset = mask
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"));
    braced_scope_at(source, marker_offset, marker)
}

fn struct_scope_for_type<'a>(source: &'a str, type_name: &str) -> &'a str {
    let mask = source_mask(source);
    let mut matches = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = mask[cursor..].find("struct") {
        let start = cursor + relative;
        cursor = start + "struct".len();
        if !token_at(&mask, start, "struct") {
            continue;
        }
        let name = skip_ascii_whitespace(&mask, cursor);
        if token_at(&mask, name, type_name) {
            matches.push(start);
        }
    }
    assert_eq!(matches.len(), 1, "expected exactly one struct {type_name}");
    braced_scope_at(source, matches[0], type_name)
}

#[test]
fn struct_scope_parser_distinguishes_exact_type_from_prefix_collision() {
    let source = r#"
        struct Mamba3GpuInferenceMixedScratch { scratch_only: usize }
        struct Mamba3GpuInferenceMixed_extra { suffix_only: usize }
        // struct Mamba3GpuInferenceMixed { comment_only: usize }
        pub struct Mamba3GpuInferenceMixed { captured_plan: Option<Plan> }
        struct Mamba3GpuInferenceMixedOther { other_only: usize }
    "#;
    assert_eq!(
        struct_scope_for_type(source, "Mamba3GpuInferenceMixed"),
        "struct Mamba3GpuInferenceMixed { captured_plan: Option<Plan> }"
    );
    assert_eq!(
        struct_scope_for_type(source, "Mamba3GpuInferenceMixedScratch"),
        "struct Mamba3GpuInferenceMixedScratch { scratch_only: usize }"
    );
    assert!(
        std::panic::catch_unwind(|| {
            struct_scope_for_type(
                "struct Mamba3GpuInferenceMixedScratch { scratch: usize }",
                "Mamba3GpuInferenceMixed",
            )
        })
        .is_err(),
        "a prefix collision cannot substitute for an absent exact type"
    );
}

fn graph_launches_are_guarded(source: &str) -> bool {
    let mask = source_mask(source);
    let mut closures = Vec::new();
    for guard in ["with_validated_launch", "with_validated_gemm_graph_launch"] {
        let mut cursor = 0;
        while let Some(relative) = mask[cursor..].find(guard) {
            let call = cursor + relative;
            cursor = call + guard.len();
            if !token_at(&mask, call, guard) {
                continue;
            }
            let open = skip_ascii_whitespace(&mask, cursor);
            if mask.as_bytes().get(open) != Some(&b'(') {
                continue;
            }
            let Some(close) = matching_delimiter(&mask, open, b'(', b')') else {
                return false;
            };
            let mut closure_cursor = open + 1;
            while let Some(relative) = mask[closure_cursor..close].find("||") {
                let bars = closure_cursor + relative;
                let body = skip_ascii_whitespace(&mask, bars + 2);
                if mask.as_bytes().get(body) == Some(&b'{') {
                    let Some(end) = matching_delimiter(&mask, body, b'{', b'}') else {
                        return false;
                    };
                    if end > close {
                        return false;
                    }
                    closures.push((body, end + 1));
                } else {
                    closures.push((body, closure_expression_end(&mask, body, close)));
                }
                closure_cursor = bars + 2;
            }
            cursor = close + 1;
        }
    }

    let mut launch_cursor = 0;
    while let Some(relative) = mask[launch_cursor..].find(".launch") {
        let launch = launch_cursor + relative;
        launch_cursor = launch + ".launch".len();
        if !token_at(&mask, launch + 1, "launch") {
            continue;
        }
        let open = skip_ascii_whitespace(&mask, launch_cursor);
        if mask.as_bytes().get(open) != Some(&b'(') {
            continue;
        }
        let close = skip_ascii_whitespace(&mask, open + 1);
        if mask.as_bytes().get(close) != Some(&b')') {
            continue;
        }
        if !closures
            .iter()
            .any(|(start, end)| *start <= launch && launch < *end)
        {
            return false;
        }
    }
    true
}

fn closure_expression_end(source: &str, start: usize, call_close: usize) -> usize {
    let mut parens = 0_u32;
    let mut brackets = 0_u32;
    let mut braces = 0_u32;
    for (relative, byte) in source.as_bytes()[start..call_close]
        .iter()
        .copied()
        .enumerate()
    {
        match byte {
            b'(' => parens += 1,
            b')' => parens = parens.saturating_sub(1),
            b'[' => brackets += 1,
            b']' => brackets = brackets.saturating_sub(1),
            b'{' => braces += 1,
            b'}' => braces = braces.saturating_sub(1),
            b',' if parens == 0 && brackets == 0 && braces == 0 => return start + relative,
            _ => {}
        }
    }
    call_close
}

/// Kernel symbols start with the operand layout they compute (`nn_`, `tn_`
/// or `nt_`); everything else in a CUDA source is a helper, a macro or a type.
fn op_kernel_identifiers(source: &str) -> BTreeSet<String> {
    source
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .filter(|token| ["nn_", "tn_", "nt_"].iter().any(|op| token.starts_with(op)))
        .map(str::to_owned)
        .collect()
}

fn module_has_inner_attributes(module: &str) -> bool {
    let mask = source_mask(module);
    let Some(open) = mask.find('{') else {
        return true;
    };
    let hash = skip_ascii_whitespace(&mask, open + 1);
    if mask.as_bytes().get(hash) != Some(&b'#') {
        return false;
    }
    let bang = skip_ascii_whitespace(&mask, hash + 1);
    if mask.as_bytes().get(bang) != Some(&b'!') {
        return false;
    }
    let bracket = skip_ascii_whitespace(&mask, bang + 1);
    mask.as_bytes().get(bracket) == Some(&b'[')
}

fn active_test_module_scope<'a>(source: &'a str, name: &str) -> Result<&'a str, String> {
    let module = unique_named_item_scope_at_depth(source, "mod", name, 0)?;
    let item_start = module.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if attributes != "#[cfg(test)]" {
        return Err(format!(
            "test module {name} must have exactly one adjacent #[cfg(test)] attribute"
        ));
    }
    if module_has_inner_attributes(module) {
        return Err(format!(
            "test module {name} must not have inner module attributes"
        ));
    }
    Ok(module)
}

fn direct_test_function_scope<'a>(
    module: &'a str,
    name: &str,
    expected_attributes: &str,
) -> Result<&'a str, String> {
    let function = unique_named_item_scope_at_depth(module, "fn", name, 1)?;
    let item_start = function.as_ptr() as usize - module.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(module, item_start)));
    if attributes != expected_attributes {
        return Err(format!(
            "test function {name} has attributes {attributes:?}, expected {expected_attributes:?}"
        ));
    }
    Ok(function)
}

fn active_inlined_production_method_scope<'a>(
    source: &'a str,
    type_name: &str,
    method: &str,
) -> Result<&'a str, String> {
    let function = method_scope_for_type(source, type_name, method);
    let item_start = function.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if attributes != "#[inline(always)]" {
        return Err(format!(
            "production method {type_name}::{method} must have exactly one #[inline(always)] attribute"
        ));
    }
    Ok(function)
}

fn validate_nonzero_input_pointer_guard(source: &str) -> Result<(), String> {
    let function = active_production_function_scope(source, "validate_f32_triad_operands")?;
    let marker = "if request.shape.reduction(request.op) != 0";
    let guard_offsets = marker_offsets_at_brace_depth(function, marker, 1);
    let [guard_offset] = guard_offsets.as_slice() else {
        return Err(format!(
            "expected one direct nonzero-reduction input guard, found {}",
            guard_offsets.len()
        ));
    };
    if !compact_code(&source_mask(contiguous_item_attributes(
        function,
        *guard_offset,
    )))
    .is_empty()
    {
        return Err("nonzero-reduction input guard must not have cfg or other attributes".into());
    }
    let guard = braced_scope_at(function, *guard_offset, marker);
    let expected_guard = r#"
        if request.shape.reduction(request.op) != 0 {
            for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                if pointer == 0 || !pointer.is_multiple_of(alignment) {
                    return Err(format!("input pointer {name}"));
                }
            }
        }
    "#;
    if compact_code(&source_mask(guard)) != compact_code(&source_mask(expected_guard)) {
        return Err("nonzero input pointer validation must keep its exact direct shape".into());
    }
    let guard_range = *guard_offset..*guard_offset + guard.len();
    let mask = source_mask(function);
    if token_present(&mask, "r#operands") {
        return Err("operand validator may not use a raw identifier for its operand bundle".into());
    }
    let body_start = function_body_open(&mask, 0, "validate_f32_triad_operands")? + 1;
    let body = &mask[body_start..];
    let mut input_fields = BTreeMap::<&str, Vec<usize>>::new();
    for offset in token_offsets(body, "operands") {
        let field_dot = skip_ascii_whitespace(body, offset + "operands".len());
        if body.as_bytes().get(field_dot) != Some(&b'.') {
            return Err("operand validator may not alias or destructure the operand bundle".into());
        }
        let field_start = skip_ascii_whitespace(body, field_dot + 1);
        let field_end = identifier_end(body, field_start);
        if !matches!(
            &body[field_start..field_end],
            "output" | "bias" | "a" | "b" | "alpha" | "beta"
        ) {
            return Err("operand validator contains an unknown operand field access".into());
        }
        let field = &body[field_start..field_end];
        if matches!(field, "a" | "b") {
            input_fields
                .entry(field)
                .or_default()
                .push(body_start + offset);
        }
    }
    for field in ["a", "b"] {
        let offsets = input_fields
            .get(field)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if offsets.is_empty() {
            return Err(format!("nonzero input guard is missing operands.{field}"));
        }
        if offsets.iter().any(|offset| !guard_range.contains(offset)) {
            return Err(format!(
                "every operands.{field} read must remain inside the direct nonzero-reduction guard"
            ));
        }
    }
    Ok(())
}

fn next_named_function(source: &str, mut cursor: usize) -> Option<(usize, usize)> {
    while let Some(relative) = source[cursor..].find("fn") {
        let start = cursor + relative;
        cursor = start + 2;
        if !token_at(source, start, "fn") {
            continue;
        }
        let name_start = skip_ascii_whitespace(source, cursor);
        let name_end = identifier_end(source, name_start);
        if name_start != name_end {
            return Some((start, name_end));
        }
    }
    None
}

fn skip_function_generics(source: &str, mut cursor: usize) -> usize {
    cursor = skip_ascii_whitespace(source, cursor);
    if source.as_bytes().get(cursor) != Some(&b'<') {
        return cursor;
    }
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[cursor..].iter().copied().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' if offset == 0 || source.as_bytes()[cursor + offset - 1] != b'-' => {
                depth = depth.checked_sub(1).expect("function generic delimiter");
                if depth == 0 {
                    return cursor + offset + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function generic parameter list")
}

fn function_parameter_count(source: &str, marker: &str) -> usize {
    let mask = source_mask(source);
    assert_eq!(marker, "fn", "only Rust fn tokens are supported");
    let (_, name_end) =
        next_named_function(&mask, 0).unwrap_or_else(|| panic!("missing function marker {marker}"));
    let open = skip_ascii_whitespace(&mask, skip_function_generics(&mask, name_end));
    assert_eq!(
        mask.as_bytes().get(open),
        Some(&b'('),
        "missing parameter list after {marker}"
    );
    let mut parens = 0_u32;
    let mut brackets = 0_u32;
    let mut braces = 0_u32;
    let mut angles = 0_u32;
    let mut items = 0_usize;
    let mut item_has_token = false;
    for character in mask[open..].chars() {
        match character {
            '(' => {
                parens += 1;
                if parens > 1 {
                    item_has_token = true;
                }
            }
            ')' => {
                parens -= 1;
                if parens == 0 {
                    return items + usize::from(item_has_token);
                }
            }
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' => braces += 1,
            '}' => braces -= 1,
            '<' => angles += 1,
            '>' if angles > 0 => angles -= 1,
            ',' if parens == 1 && brackets == 0 && braces == 0 && angles == 0 => {
                if item_has_token {
                    items += 1;
                    item_has_token = false;
                }
            }
            character if !character.is_whitespace() && parens == 1 => item_has_token = true,
            _ => {}
        }
    }
    panic!("unterminated parameter list after {marker}")
}

fn assert_no_rust_function_exceeds_seven(source: &str, label: &str) {
    let source = source_mask(source);
    let mut cursor = 0;
    while let Some((start, name_end)) = next_named_function(&source, cursor) {
        let name_start = skip_ascii_whitespace(&source, start + 2);
        let name = &source[name_start..name_end];
        let count = function_parameter_count(&source[start..], "fn");
        assert!(
            count <= 7,
            "{label}::{name} has {count} arguments; limit is 7"
        );
        cursor = name_end;
    }
}

fn compose_cuda(fragments: &[&str]) -> String {
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

fn tf32_cuda_blob(source: &str, mma16: bool) -> String {
    let mut fragments = vec![
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
    ];
    if mma16 {
        fragments.push(include_str!("../kernels/gemm_bi_triad/mma16.cuh"));
    }
    fragments.push(source);
    compose_cuda(&fragments)
}

fn compile_tf32_ptx(source: String, target: &'static str) -> String {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some(target),
        options: vec![
            "--std=c++17".to_owned(),
            "--fmad=true".to_owned(),
            "--generate-line-info".to_owned(),
            "-DNDEBUG".to_owned(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(source, options)
        .unwrap_or_else(|error| panic!("TF32 module must compile for {target}: {error}"));
    mamba_rs::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
        image.as_bytes().expect("compiled TF32 PTX image"),
    )
    .expect("compiled TF32 PTX must be canonical UTF-8")
}

fn ptx_entry<'a>(ptx: &'a str, symbol: &str) -> &'a str {
    let marker = format!(".entry {symbol}(");
    let start = ptx
        .find(&marker)
        .unwrap_or_else(|| panic!("compiled PTX is missing {symbol}"));
    let mut parentheses = 1_u32;
    let mut body_start = None;
    for (offset, byte) in ptx[start + marker.len()..].bytes().enumerate() {
        match byte {
            b'(' => parentheses += 1,
            b')' => parentheses -= 1,
            b'{' if parentheses == 0 => {
                body_start = Some(start + marker.len() + offset);
                break;
            }
            _ => {}
        }
    }
    let body_start = body_start.unwrap_or_else(|| panic!("compiled PTX body for {symbol}"));
    let mut braces = 0_u32;
    for (offset, byte) in ptx[body_start..].bytes().enumerate() {
        match byte {
            b'{' => braces += 1,
            b'}' => {
                braces -= 1;
                if braces == 0 {
                    return &ptx[start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated compiled PTX body for {symbol}")
}

fn ptx_parameters<'a>(entry: &'a str, symbol: &str) -> &'a str {
    let marker = format!(".entry {symbol}(");
    entry
        .split_once(&marker)
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(parameters, _)| parameters))
        .unwrap_or_else(|| panic!("compiled PTX parameter list for {symbol}"))
}

fn ptx_entry_symbols(ptx: &str, family: &str) -> BTreeSet<String> {
    ptx.lines()
        .filter_map(|line| {
            let marker = line.find(".entry ")? + ".entry ".len();
            let tail = &line[marker..];
            let end = tail.find('(')?;
            let symbol = &tail[..end];
            symbol.contains(family).then(|| symbol.to_owned())
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PtxParameter {
    name: String,
    scalar_type: &'static str,
    bytes: usize,
    alignment: Option<usize>,
}

fn parse_ptx_parameters(parameters: &str) -> Vec<PtxParameter> {
    parameters
        .split(".param")
        .skip(1)
        .map(|declaration| {
            let declaration = declaration
                .split_once(',')
                .map_or(declaration, |(head, _)| head);
            if declaration.contains(".u64") {
                PtxParameter {
                    name: declaration
                        .split_ascii_whitespace()
                        .last()
                        .expect("PTX scalar parameter name")
                        .to_owned(),
                    scalar_type: "u64",
                    bytes: 8,
                    alignment: None,
                }
            } else if declaration.contains(".b8") {
                let bytes = declaration
                    .split_once('[')
                    .and_then(|(_, tail)| tail.split_once(']'))
                    .and_then(|(width, _)| width.trim().parse().ok())
                    .unwrap_or_else(|| panic!("missing PTX byte-array width: {declaration}"));
                let alignment = declaration
                    .split_once(".align")
                    .and_then(|(_, tail)| tail.split_ascii_whitespace().next())
                    .and_then(|value| value.parse().ok());
                PtxParameter {
                    name: declaration
                        .split_once('[')
                        .map(|(head, _)| {
                            head.split_ascii_whitespace()
                                .last()
                                .expect("PTX aggregate parameter name")
                                .to_owned()
                        })
                        .expect("PTX aggregate parameter width"),
                    scalar_type: "b8",
                    bytes,
                    alignment,
                }
            } else {
                panic!("unsupported PTX parameter declaration: {declaration}")
            }
        })
        .collect()
}

fn expected_ptx_parameters(
    symbol: &str,
    bundle_bytes: usize,
    tensor_map_alignment: usize,
) -> Vec<PtxParameter> {
    let parameter = |index, scalar_type, bytes, alignment| PtxParameter {
        name: format!("{symbol}_param_{index}"),
        scalar_type,
        bytes,
        alignment,
    };
    if bundle_bytes == 32
        && ["_splitk2_", "_splitk4_", "_splitk8_"]
            .iter()
            .any(|family| symbol.contains(family))
    {
        vec![
            parameter(0, "u64", 8, None),
            parameter(1, "u64", 8, None),
            parameter(2, "u64", 8, None),
            parameter(3, "u64", 8, None),
            parameter(4, "u64", 8, None),
            parameter(5, "u64", 8, None),
            parameter(6, "b8", 32, Some(4)),
        ]
    } else if bundle_bytes == 32 {
        vec![
            parameter(0, "u64", 8, None),
            parameter(1, "u64", 8, None),
            parameter(2, "u64", 8, None),
            parameter(3, "u64", 8, None),
            parameter(4, "b8", 32, Some(4)),
        ]
    } else if symbol.contains("_streamk") || symbol.contains("_tma_fma_") {
        // The exact-F32 routes share the stream-K parameter order with a
        // 32-byte bundle: output rows, columns, reduction, output stride,
        // split count, k tiles per split behind alpha and beta.
        let bundle = if symbol.contains("_tma_fma_") { 32 } else { 40 };
        vec![
            parameter(0, "u64", 8, None),
            parameter(1, "u64", 8, None),
            parameter(2, "u64", 8, None),
            parameter(3, "b8", 128, Some(tensor_map_alignment)),
            parameter(4, "b8", 128, Some(tensor_map_alignment)),
            parameter(5, "u64", 8, None),
            parameter(6, "b8", bundle, Some(4)),
        ]
    } else {
        vec![
            parameter(0, "u64", 8, None),
            parameter(1, "b8", 128, Some(tensor_map_alignment)),
            parameter(2, "b8", 128, Some(tensor_map_alignment)),
            parameter(3, "u64", 8, None),
            parameter(4, "b8", 40, Some(4)),
        ]
    }
}

fn normalized_reduction_bundle_offset(symbol: &str, bundle_bytes: usize) -> usize {
    match (
        bundle_bytes,
        symbol.starts_with("nn_"),
        symbol.starts_with("tn_"),
    ) {
        (32, true, _) => 12,
        (32, false, true) => 8,
        (32, false, false) => 16,
        (40, true, _) => 28,
        (40, false, true) => 24,
        (40, false, false) => 32,
        _ => panic!("unexpected TF32 parameter bundle width {bundle_bytes}"),
    }
}

fn ptx_instruction(line: &str) -> Option<(&str, &str)> {
    let mut instruction = line.trim();
    if instruction.is_empty()
        || instruction.ends_with(':')
        || instruction.starts_with(['.', '{', '}'])
    {
        return None;
    }
    if instruction.starts_with('@') {
        instruction = instruction.split_once(char::is_whitespace)?.1.trim_start();
    }
    let (opcode, operands) = instruction
        .split_once(char::is_whitespace)
        .unwrap_or((instruction.trim_end_matches(';'), ""));
    Some((opcode, operands.trim_end_matches(';').trim()))
}

fn ptx_operands(operands: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0_u32;
    for (offset, byte) in operands.bytes().enumerate() {
        match byte {
            b'{' | b'[' | b'(' => depth += 1,
            b'}' | b']' | b')' => {
                assert!(depth > 0, "unbalanced PTX operand list: {operands}");
                depth -= 1;
            }
            b',' if depth == 0 => {
                result.push(operands[start..offset].trim());
                start = offset + 1;
            }
            _ => {}
        }
    }
    assert_eq!(depth, 0, "unbalanced PTX operand list: {operands}");
    if start < operands.len() || !operands.is_empty() {
        result.push(operands[start..].trim());
    }
    result
}

fn ptx_destination_registers<'a>(opcode: &str, operands: &'a [&'a str]) -> Vec<&'a str> {
    if [
        "st.",
        "st::",
        "red.",
        "red::",
        "bra",
        "brx",
        "ret",
        "exit",
        "trap",
        "brkpt",
        "jmp",
        "bar.sync",
        "bar.arrive",
        "bar.warp.sync",
        "barrier.sync",
        "barrier.arrive",
        "membar.",
        "fence.",
        "prefetch.",
        "prefetchu.",
    ]
    .iter()
    .any(|prefix| opcode.starts_with(prefix))
    {
        return Vec::new();
    }
    let Some(first) = operands.first() else {
        return Vec::new();
    };
    if let Some(vector) = first
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
    {
        let destinations: Vec<_> = vector
            .split(',')
            .map(str::trim)
            .filter(|register| !register.is_empty())
            .collect();
        assert!(
            !destinations.is_empty()
                && destinations
                    .iter()
                    .all(|register| register.starts_with('%')),
            "unknown PTX multi-destination {first} for {opcode}"
        );
        return destinations;
    }
    if first.contains('|') {
        let destinations: Vec<_> = first.split('|').map(str::trim).collect();
        assert!(
            destinations
                .iter()
                .all(|register| register.starts_with('%')),
            "unknown PTX multi-destination {first} for {opcode}"
        );
        return destinations;
    }
    first
        .starts_with('%')
        .then_some(*first)
        .into_iter()
        .collect()
}

fn ptx_line_successors(lines: &[&str]) -> Vec<Vec<usize>> {
    let mut labels = BTreeMap::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some(label) = line.trim().strip_suffix(':') {
            assert!(
                labels.insert(label.to_owned(), index).is_none(),
                "duplicate PTX label {label}"
            );
        }
    }
    let mut successors = vec![Vec::new(); lines.len()];
    for (index, line) in lines.iter().enumerate() {
        let next = (index + 1 < lines.len()).then_some(index + 1);
        let Some((opcode, operands)) = ptx_instruction(line) else {
            successors[index].extend(next);
            continue;
        };
        if matches!(opcode, "ret" | "exit" | "trap" | "brkpt") {
            continue;
        }
        if opcode.starts_with("bra") {
            let target = operands
                .split_ascii_whitespace()
                .last()
                .expect("direct PTX branch target");
            successors[index].push(
                *labels
                    .get(target)
                    .unwrap_or_else(|| panic!("unknown PTX branch target {target}")),
            );
            if line.trim_start().starts_with('@') {
                successors[index].extend(next);
            }
        } else {
            successors[index].extend(next);
        }
    }
    successors
}

fn propagated_reduction_registers(
    lines: &[&str],
    load_index: usize,
    initial: String,
) -> Vec<BTreeSet<String>> {
    let successors = ptx_line_successors(lines);
    let mut predecessors = vec![Vec::new(); lines.len()];
    for (line, targets) in successors.iter().enumerate() {
        for target in targets {
            predecessors[*target].push(line);
        }
    }
    let transfer = |index: usize, before: &BTreeSet<String>| {
        let Some((opcode, operands)) = ptx_instruction(lines[index]) else {
            return before.clone();
        };
        let operands = ptx_operands(operands);
        let destinations = ptx_destination_registers(opcode, &operands);
        let preserves_value = matches!(opcode, "mov.u32" | "mov.s32" | "mov.b32")
            || (opcode.starts_with("cvt.")
                && opcode
                    .split('.')
                    .skip(1)
                    .all(|part| matches!(part, "u32" | "s32" | "b32")));
        let source_is_tracked =
            preserves_value && operands.len() == 2 && before.contains(operands[1]);
        let mut executed = before.clone();
        for destination in &destinations {
            executed.remove(*destination);
        }
        if index == load_index {
            assert!(
                !lines[index].trim_start().starts_with('@')
                    && destinations.as_slice() == [initial.as_str()],
                "normalized reduction load must be one unconditional destination"
            );
            executed.insert(initial.clone());
        } else if source_is_tracked {
            executed.insert(operands[0].to_owned());
        }
        if lines[index].trim_start().starts_with('@') {
            executed.intersection(before).cloned().collect()
        } else {
            executed
        }
    };

    let mut before = vec![None::<BTreeSet<String>>; lines.len()];
    let mut after = vec![None::<BTreeSet<String>>; lines.len()];
    for _ in 0..=lines.len() * 4 {
        let mut changed = false;
        for index in 0..lines.len() {
            let next_before = if index == 0 {
                Some(BTreeSet::new())
            } else {
                let incoming: Vec<_> = predecessors[index]
                    .iter()
                    .filter_map(|predecessor| after[*predecessor].as_ref())
                    .collect();
                incoming.first().map(|first| {
                    incoming
                        .iter()
                        .skip(1)
                        .fold((*first).clone(), |set, value| {
                            set.intersection(value).cloned().collect()
                        })
                })
            };
            let next_after = next_before.as_ref().map(|state| transfer(index, state));
            if before[index] != next_before || after[index] != next_after {
                before[index] = next_before;
                after[index] = next_after;
                changed = true;
            }
        }
        if !changed {
            return before.into_iter().map(Option::unwrap_or_default).collect();
        }
    }
    panic!("PTX provenance analysis did not converge")
}

fn ptx_line_predicate(line: &str) -> Option<(&str, bool)> {
    let predicate = line
        .trim_start()
        .strip_prefix('@')?
        .split_ascii_whitespace()
        .next()?;
    let (predicate, negated) = predicate
        .strip_prefix('!')
        .map_or((predicate, false), |predicate| (predicate, true));
    Some((predicate, negated))
}

fn ptx_predicate_register(predicate: &str) -> bool {
    predicate
        .strip_prefix("%p")
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
}

fn proven_predicate_before_lines(
    lines: &[&str],
    definition_index: usize,
    predicate: &str,
) -> Vec<bool> {
    assert!(
        ptx_line_predicate(lines[definition_index]).is_none(),
        "guard predicate definition must be unconditional"
    );
    let successors = ptx_line_successors(lines);
    let mut predecessors = vec![Vec::new(); lines.len()];
    for (line, targets) in successors.iter().enumerate() {
        for target in targets {
            predecessors[*target].push(line);
        }
    }
    let transfer = |index: usize, before: bool| {
        let Some((opcode, operands)) = ptx_instruction(lines[index]) else {
            return before;
        };
        let operands = ptx_operands(operands);
        let destinations = ptx_destination_registers(opcode, &operands);
        let mut executed = before && !destinations.contains(&predicate);
        if index == definition_index {
            assert_eq!(
                destinations.as_slice(),
                [predicate],
                "guard setp must define exactly one predicate"
            );
            executed = true;
        }
        if ptx_line_predicate(lines[index]).is_some() {
            executed && before
        } else {
            executed
        }
    };
    let mut before = vec![None::<bool>; lines.len()];
    let mut after = vec![None::<bool>; lines.len()];
    for _ in 0..=lines.len() * 4 {
        let mut changed = false;
        for index in 0..lines.len() {
            let next_before = if index == 0 {
                Some(false)
            } else {
                let incoming: Vec<_> = predecessors[index]
                    .iter()
                    .filter_map(|predecessor| after[*predecessor])
                    .collect();
                incoming.first().map(|first| {
                    incoming
                        .iter()
                        .skip(1)
                        .fold(*first, |state, value| state && *value)
                })
            };
            let next_after = next_before.map(|state| transfer(index, state));
            if before[index] != next_before || after[index] != next_after {
                before[index] = next_before;
                after[index] = next_after;
                changed = true;
            }
        }
        if !changed {
            return before.into_iter().map(Option::unwrap_or_default).collect();
        }
    }
    panic!("PTX predicate provenance analysis did not converge")
}

fn assert_k0_cfg_dominates_entry(entry: &str, symbol: &str, bundle_bytes: usize) {
    let offset = normalized_reduction_bundle_offset(symbol, bundle_bytes);
    let lines: Vec<_> = entry.lines().collect();
    let (load_index, reduction_register) = lines
        .iter()
        .enumerate()
        .find_map(|(index, line)| {
            let compact: String = line
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect();
            if !compact.contains("ld.param")
                || !(compact.contains(".u32")
                    || compact.contains(".s32")
                    || compact.contains(".b32"))
                || !compact.contains(&format!("+{offset}]"))
            {
                return None;
            }
            let destination = line
                .split_once("ld.param")?
                .1
                .split_once(',')?
                .0
                .split_ascii_whitespace()
                .last()?;
            Some((index, destination.to_owned()))
        })
        .unwrap_or_else(|| {
            let loads = entry
                .lines()
                .filter(|line| line.contains("ld.param"))
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "{symbol} must load normalized reduction at bundle +{offset}; parameter loads:\n{loads}"
            )
        });
    let propagated = propagated_reduction_registers(&lines, load_index, reduction_register.clone());
    let mut guards = Vec::new();
    for (setp_index, line) in lines.iter().enumerate().skip(load_index + 1) {
        let Some((opcode, operands)) = ptx_instruction(line) else {
            continue;
        };
        if ptx_line_predicate(line).is_some() {
            continue;
        }
        let zero_when_predicate = if opcode.starts_with("setp.eq.") {
            true
        } else if opcode.starts_with("setp.ne.") {
            false
        } else {
            continue;
        };
        let operands = ptx_operands(operands);
        if operands.len() < 3
            || !((propagated[setp_index].contains(operands[1]) && operands[2] == "0")
                || (propagated[setp_index].contains(operands[2]) && operands[1] == "0"))
        {
            continue;
        }
        let predicate = operands[0];
        if !ptx_predicate_register(predicate) {
            continue;
        }
        let proven_predicate = proven_predicate_before_lines(&lines, setp_index, predicate);
        for (branch_index, branch) in lines.iter().enumerate().skip(setp_index + 1) {
            let Some((branch_predicate, negated)) = ptx_line_predicate(branch) else {
                continue;
            };
            let Some((branch_opcode, branch_operands)) = ptx_instruction(branch) else {
                continue;
            };
            if branch_predicate != predicate
                || !branch_opcode.starts_with("bra")
                || !proven_predicate[branch_index]
            {
                continue;
            }
            guards.push((
                setp_index,
                predicate.to_owned(),
                zero_when_predicate,
                branch_index,
                branch_operands
                    .split_ascii_whitespace()
                    .last()
                    .expect("direct PTX branch target")
                    .to_owned(),
                !negated,
            ));
        }
    }
    assert_eq!(
        guards.len(),
        1,
        "{symbol} requires exactly one outer zero-reduction guard propagated from parameter 4"
    );
    let (_, _, zero_when_predicate, branch_index, target, branch_when_predicate) =
        guards.pop().expect("one zero-reduction guard");

    let zero_takes_branch = branch_when_predicate == zero_when_predicate;

    let is_terminator = |line: &str| {
        ptx_instruction(line).is_some_and(|(opcode, _)| {
            opcode.starts_with("bra") || matches!(opcode, "ret" | "exit" | "trap")
        })
    };
    assert!(
        !lines.iter().any(|line| line.contains("call")),
        "{symbol} CFG checker fails closed on device calls"
    );
    let mut leaders = BTreeSet::from([0_usize]);
    for (index, line) in lines.iter().enumerate() {
        if line.trim_end().ends_with(':') {
            leaders.insert(index);
        }
        if is_terminator(line) && index + 1 < lines.len() {
            leaders.insert(index + 1);
        }
    }
    let leaders: Vec<_> = leaders.into_iter().collect();
    let ranges: Vec<_> = leaders
        .iter()
        .enumerate()
        .map(|(index, start)| {
            (
                *start,
                leaders.get(index + 1).copied().unwrap_or(lines.len()),
            )
        })
        .collect();
    let block_for_line = |line: usize| {
        ranges
            .iter()
            .position(|(start, end)| *start <= line && line < *end)
            .unwrap_or_else(|| panic!("{symbol} CFG line {line}"))
    };
    let mut label_blocks = BTreeMap::new();
    for (block, (start, end)) in ranges.iter().copied().enumerate() {
        for line in &lines[start..end] {
            let trimmed = line.trim();
            if let Some(label) = trimmed.strip_suffix(':') {
                label_blocks.insert(label.to_owned(), block);
            }
        }
    }
    let mut successors = vec![Vec::<usize>::new(); ranges.len()];
    for (block, (start, end)) in ranges.iter().copied().enumerate() {
        let last = lines[start..end]
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty() && !line.trim_end().ends_with(':'))
            .copied()
            .unwrap_or("");
        if ["ret;", "exit;", "trap;"]
            .iter()
            .any(|term| last.contains(term))
        {
            continue;
        }
        if let Some((opcode, operands)) = ptx_instruction(last)
            && opcode.starts_with("bra")
        {
            let target = operands
                .split_ascii_whitespace()
                .last()
                .unwrap_or_else(|| panic!("{symbol} indirect branch: {last}"));
            let target_block = *label_blocks
                .get(target)
                .unwrap_or_else(|| panic!("{symbol} unknown branch target {target}"));
            successors[block].push(target_block);
            if last.contains('@') && block + 1 < ranges.len() {
                successors[block].push(block + 1);
            }
        } else if block + 1 < ranges.len() {
            successors[block].push(block + 1);
        }
        successors[block].sort_unstable();
        successors[block].dedup();
    }
    let guard_block = block_for_line(branch_index);
    let target_block = *label_blocks.get(&target).expect("guard target block");
    let fallthrough_block = block_for_line(branch_index + 1);
    let (zero_successor, nonzero_successor) = if zero_takes_branch {
        (target_block, fallthrough_block)
    } else {
        (fallthrough_block, target_block)
    };

    let mut predecessors = vec![Vec::<usize>::new(); ranges.len()];
    for (block, edges) in successors.iter().enumerate() {
        for successor in edges {
            predecessors[*successor].push(block);
        }
    }
    let reachable = |start: usize| {
        let mut seen = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(block) = pending.pop() {
            if seen.insert(block) {
                pending.extend(successors[block].iter().copied());
            }
        }
        seen
    };
    let entry_reachable = reachable(0);
    let all_blocks: BTreeSet<_> = (0..ranges.len()).collect();
    let mut dominators = vec![all_blocks.clone(); ranges.len()];
    dominators[0] = BTreeSet::from([0]);
    loop {
        let mut changed = false;
        for block in 1..ranges.len() {
            if !entry_reachable.contains(&block) {
                continue;
            }
            let mut reachable_predecessors = predecessors[block]
                .iter()
                .copied()
                .filter(|predecessor| entry_reachable.contains(predecessor));
            let first = reachable_predecessors
                .next()
                .expect("reachable non-entry block has a reachable predecessor");
            let mut next = dominators[first].clone();
            for predecessor in reachable_predecessors {
                next = next
                    .intersection(&dominators[predecessor])
                    .copied()
                    .collect();
            }
            next.insert(block);
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let protected = [
        "cp.async.bulk.tensor",
        "prefetch.tensormap",
        "mbarrier.",
        "bar.",
        "wgmma.",
        "setmaxnreg.",
        "tcgen05.",
        "mma.sync.",
        "ldmatrix.",
        "cvt_to_shared_descriptor",
    ];
    for (block, (start, end)) in ranges.iter().copied().enumerate() {
        let body = lines[start..end].join("\n");
        if entry_reachable.contains(&block) && protected.iter().any(|opcode| body.contains(opcode))
        {
            assert!(
                dominators[block].contains(&guard_block),
                "{symbol} zero guard block {guard_block} does not dominate protected block {block}; \
                 dominators={:?}; body:\n{body}",
                dominators[block]
            );
        }
    }
    let zero_reachable = reachable(zero_successor);
    let nonzero_reachable = reachable(nonzero_successor);
    let zero_terminals: Vec<_> = zero_reachable
        .iter()
        .copied()
        .filter(|block| successors[*block].is_empty())
        .collect();
    assert!(
        !zero_terminals.is_empty(),
        "{symbol} K=0 CFG must terminate"
    );
    for block in zero_terminals {
        let (start, end) = ranges[block];
        let terminal = lines[start..end]
            .iter()
            .rev()
            .find_map(|line| ptx_instruction(line).map(|(opcode, _)| opcode))
            .unwrap_or_else(|| panic!("{symbol} K=0 terminal block {block} is empty"));
        assert!(
            matches!(terminal, "ret" | "exit"),
            "{symbol} K=0 terminal block {block} ends in {terminal}, not ret/exit"
        );
    }
    for block in &zero_reachable {
        let (start, end) = ranges[*block];
        let body = lines[start..end].join("\n");
        assert!(
            !protected.iter().any(|opcode| body.contains(opcode)),
            "{symbol} K=0 reaches protected block {block}"
        );
    }
    let zero_body = zero_reachable
        .iter()
        .map(|block| {
            let (start, end) = ranges[*block];
            lines[start..end].join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        zero_body.contains("st.global"),
        "{symbol} K=0 CFG must execute the output epilogue"
    );
    let allowed_fp = if symbol.starts_with("nn_") {
        BTreeSet::from(["mul.rn.f32", "fma.rn.f32"])
    } else if symbol.starts_with("tn_") {
        BTreeSet::from(["fma.rn.f32"])
    } else {
        BTreeSet::from(["mul.rn.f32"])
    };
    let mut observed_fp = BTreeSet::new();
    for line in zero_body.lines() {
        let Some((opcode, _)) = ptx_instruction(line) else {
            continue;
        };
        if !opcode.contains(".f32") {
            continue;
        }
        if allowed_fp.contains(opcode) {
            observed_fp.insert(opcode);
            continue;
        }
        let nonarithmetic = ["ld.", "st.", "mov.", "setp.", "selp."]
            .iter()
            .any(|prefix| opcode.starts_with(prefix));
        assert!(
            nonarithmetic,
            "{symbol} K=0 contains forbidden f32 opcode {opcode}: {line}"
        );
    }
    assert_eq!(
        observed_fp, allowed_fp,
        "{symbol} K=0 epilogue scalar FP contract"
    );
    let nonzero_body = nonzero_reachable
        .iter()
        .map(|block| {
            let (start, end) = ranges[*block];
            lines[start..end].join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (transfer, matrix) = if symbol.contains("_sm90a_") {
        ("cp.async.bulk.tensor", "wgmma.")
    } else if symbol.contains("_sm100_") {
        ("cp.async.bulk.tensor", "tcgen05.")
    } else if symbol.contains("_sm120_") {
        ("cp.async.bulk.tensor", "mma.sync.")
    } else {
        ("cp.async", "mma.sync.")
    };
    assert!(
        nonzero_body.contains(transfer) && nonzero_body.contains(matrix),
        "{symbol} nonzero CFG must contain {transfer} and {matrix}"
    );
}

fn cuda_tool(name: &str) -> PathBuf {
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(root) = std::env::var_os(variable) {
            let candidate = PathBuf::from(root).join("bin").join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(name)
}

fn metric_before(line: &str, marker: &str) -> Option<u64> {
    line.split_once(marker)?
        .0
        .split_ascii_whitespace()
        .last()?
        .parse()
        .ok()
}

fn metric_after(line: &str, marker: &str, suffix: &str) -> Option<u64> {
    line.split_once(marker)?
        .1
        .split_once(suffix)?
        .0
        .trim()
        .parse()
        .ok()
}

fn tf32_resource_caps(symbol: &str) -> (u64, u64) {
    if symbol.ends_with("_s3_pair_streamk") {
        // One resident CTA per multiprocessor owns the whole register file.
        return (255, 73_856);
    }
    if symbol.contains("_tma_fma_") {
        // Two dense 16-deep stages plus one mbarrier per stage; the NT
        // k-vector arms keep a float4 B fragment per column and run two
        // blocks per multiprocessor.
        let shared = if symbol.contains("_m64n64_") {
            16_400
        } else {
            24_592
        };
        let registers = if symbol.ends_with("_kvec") { 255 } else { 168 };
        return (registers, shared);
    }
    if symbol.contains("_m80n32_bk64_s2") {
        return (128, 57_472);
    }
    if symbol.contains("_sm80_") {
        let stage = if symbol.contains("_s2") {
            2
        } else if symbol.contains("_s3") {
            3
        } else if symbol.contains("_s4") {
            4
        } else {
            panic!("unknown SM80 TF32 stage for {symbol}")
        };
        if symbol.contains("_m128n64_") {
            assert!(matches!(stage, 2 | 3), "invalid SM80 M128N64 stage");
            let stage_bytes = if symbol.starts_with("tn_") {
                26_624
            } else {
                27_648
            };
            (192, stage_bytes * stage)
        } else if symbol.contains("_m64n64_") {
            assert!(matches!(stage, 2 | 3), "invalid SM80 M64N64 stage");
            (128, 18_432 * stage)
        } else if symbol.contains("_m16n32_") {
            assert!(matches!(stage, 3 | 4), "invalid SM80 M16N32 stage");
            let shared = if symbol.starts_with("nn_") {
                assert_eq!(stage, 4, "invalid SM80 NN M16N32 stage");
                29_696
            } else if symbol.starts_with("tn_") {
                assert_eq!(stage, 4, "invalid SM80 TN M16N32 stage");
                32_768
            } else if symbol.starts_with("nt_") {
                6_912 * stage
            } else {
                panic!("unknown SM80 TF32 operation for {symbol}")
            };
            (96, shared)
        } else if symbol.contains("_m32n32_") {
            assert!(
                symbol.starts_with("nt_") && matches!(stage, 3 | 4),
                "invalid SM80 NT M32N32 stage"
            );
            (96, 9_216 * stage)
        } else if symbol.contains("_m16n16_") {
            assert_eq!(stage, 4, "invalid SM80 M16N16 stage");
            let shared = if symbol.starts_with("nn_") {
                21_504
            } else if symbol.starts_with("tn_") {
                24_576
            } else if symbol.starts_with("nt_") {
                18_432
            } else {
                panic!("unknown SM80 TF32 operation for {symbol}")
            };
            (96, shared)
        } else {
            panic!("unknown SM80 TF32 tile for {symbol}")
        }
    } else if symbol.contains("_sm90a_") {
        let registers = if symbol.ends_with("_wg1") {
            168
        } else if symbol.ends_with("_wg2") {
            128
        } else {
            panic!("unknown SM90a TF32 schedule for {symbol}")
        };
        (registers, 73_984)
    } else if symbol.contains("_sm100_") {
        let stage = if symbol.contains("_s4_") {
            4
        } else if symbol.contains("_s3_") {
            3
        } else if symbol.contains("_s2_") {
            2
        } else {
            panic!("unknown SM100 TF32 stage for {symbol}")
        };
        let shared = if symbol.contains("_m128n64_") {
            256 + 24_576 * stage
        } else if symbol.contains("_m128n128_") {
            256 + 32_768 * stage
        } else {
            panic!("unknown SM100 TF32 tile for {symbol}")
        };
        (128, shared)
    } else if symbol.contains("_sm120_") {
        assert!(
            symbol.contains("_m128n64_")
                || symbol.contains("_m64n128_")
                || symbol.contains("_m64n64_"),
            "unknown SM120 TF32 tile for {symbol}"
        );
        let stage = if symbol.contains("_s2") {
            2
        } else if symbol.contains("_s3") {
            3
        } else if symbol.contains("_s4") {
            4
        } else {
            panic!("unknown SM120 TF32 stage for {symbol}")
        };
        let shared = if symbol.contains("_m64n64_") {
            assert_eq!(stage, 2, "invalid SM120 M64N64 stage");
            128 + 16_384 * stage
        } else {
            128 + 24_576 * stage
        };
        (128, shared)
    } else {
        panic!("unknown TF32 resource family for {symbol}")
    }
}

fn assemble_and_disassemble_tf32(
    ptx: &str,
    target: &str,
    label: &str,
) -> (String, String, String, String) {
    let directory = tempfile::tempdir().expect("TF32 ptxas tempdir");
    let input = directory.path().join("tf32.ptx");
    let cubin = directory.path().join("tf32.cubin");
    std::fs::write(&input, ptx).expect("write TF32 PTX");
    let assembly = Command::new(cuda_tool("ptxas"))
        .arg(format!("-arch={target}"))
        .arg("-v")
        .arg("-lineinfo")
        .arg(&input)
        .arg("-o")
        .arg(&cubin)
        .output()
        .unwrap_or_else(|error| panic!("run ptxas for {label}: {error}"));
    assert!(
        assembly.status.success(),
        "ptxas failed for {label}/{target}: {}",
        String::from_utf8_lossy(&assembly.stderr)
    );
    let report = String::from_utf8_lossy(&assembly.stderr).into_owned();
    // Below CUDA 12.9 the assembler spills kernels the contract toolkits keep
    // in registers; the loader excludes such a symbol on that toolkit, so the
    // contract reports the spill there instead of failing on it.
    let toolkit_variance = loaded_nvrtc_version() < (12, 9);
    for line in report.lines() {
        for marker in [
            " bytes stack frame",
            " bytes spill stores",
            " bytes spill loads",
        ] {
            if let Some(value) = metric_before(line, marker) {
                if value != 0 && toolkit_variance {
                    eprintln!("{label}/{target} uses local resources on this toolkit: {line}");
                    continue;
                }
                assert_eq!(value, 0, "{label}/{target} local resource: {line}");
            }
        }
    }

    let disassembly = Command::new(cuda_tool("nvdisasm"))
        .args(["--print-line-info-inline", "--separate-functions"])
        .arg(&cubin)
        .output()
        .unwrap_or_else(|error| panic!("run nvdisasm for {label}: {error}"));
    assert!(
        disassembly.status.success(),
        "nvdisasm failed for {label}/{target}: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    let cfg = Command::new(cuda_tool("nvdisasm"))
        .args([
            "--output-control-flow-graph-with-basic-blocks",
            "--print-instr-offsets-cfg",
        ])
        .arg(&cubin)
        .output()
        .unwrap_or_else(|error| panic!("run nvdisasm CFG for {label}: {error}"));
    assert!(
        cfg.status.success(),
        "nvdisasm CFG failed for {label}/{target}: {}",
        String::from_utf8_lossy(&cfg.stderr)
    );
    let resources = Command::new(cuda_tool("cuobjdump"))
        .arg("--dump-resource-usage")
        .arg(&cubin)
        .output()
        .unwrap_or_else(|error| panic!("run cuobjdump for {label}: {error}"));
    assert!(
        resources.status.success(),
        "cuobjdump failed for {label}/{target}: {}",
        String::from_utf8_lossy(&resources.stderr)
    );
    (
        report,
        String::from_utf8(resources.stdout).expect("TF32 resources must be UTF-8"),
        String::from_utf8(disassembly.stdout).expect("TF32 SASS must be UTF-8"),
        String::from_utf8(cfg.stdout).expect("TF32 SASS CFG must be UTF-8"),
    )
}

fn assert_cuobjdump_zero_resources(report: &str, symbols: &BTreeSet<String>, label: &str) {
    for symbol in symbols {
        let markers = [
            format!("Function {symbol}:"),
            format!("Function : {symbol}"),
        ];
        let (start, marker) = markers
            .iter()
            .find_map(|marker| report.find(marker).map(|start| (start, marker)))
            .unwrap_or_else(|| panic!("{label}/{symbol} missing cuobjdump resource record"));
        assert_eq!(
            markers
                .iter()
                .map(|candidate| report.matches(candidate).count())
                .sum::<usize>(),
            1,
            "{label}/{symbol} requires exactly one cuobjdump record"
        );
        let tail = &report[start + marker.len()..];
        let end = tail.find("Function ").unwrap_or(tail.len());
        let record = &tail[..end];
        if loaded_nvrtc_version() < (12, 9)
            && !(record.contains("STACK:0") && record.contains("LOCAL:0"))
        {
            // Below CUDA 12.9 the assembler spills this symbol; the loader
            // excludes it on that toolkit, so the record is reported here.
            eprintln!("{label}/{symbol} uses local resources on this toolkit: {record}");
        } else {
            assert!(
                record.contains("STACK:0") && record.contains("LOCAL:0"),
                "{label}/{symbol} resource record must explicitly report STACK:0 LOCAL:0: {record}"
            );
        }
        let registers = metric_after(record, "REG:", " ")
            .or_else(|| metric_after(record, "REG:", "\n"))
            .unwrap_or_else(|| panic!("{label}/{symbol} missing cuobjdump REG value: {record}"));
        let shared = metric_after(record, "SHARED:", " ")
            .or_else(|| metric_after(record, "SHARED:", "\n"))
            .unwrap_or_else(|| panic!("{label}/{symbol} missing cuobjdump SHARED value: {record}"));
        let (register_cap, shared_cap) = tf32_resource_caps(symbol);
        assert!(
            registers <= register_cap && shared <= shared_cap,
            "{label}/{symbol} exceeds cuobjdump caps REG {registers}/{register_cap}, SHARED {shared}/{shared_cap}"
        );
    }
}

fn assemble_tf32_checker(ptx: &str, target: &str, label: &str) {
    let directory = tempfile::tempdir().expect("TF32 checker tempdir");
    let input = directory.path().join("checker.ptx");
    let cubin = directory.path().join("checker.cubin");
    std::fs::write(&input, ptx).expect("write TF32 checker PTX");
    let output = Command::new(cuda_tool("ptxas"))
        .arg(format!("-arch={target}"))
        .arg("-g-tmem-access-check")
        .arg("-v")
        .arg(&input)
        .arg("-o")
        .arg(&cubin)
        .output()
        .unwrap_or_else(|error| panic!("run checker ptxas for {label}: {error}"));
    assert!(
        output.status.success(),
        "checker ptxas failed for {label}/{target}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_per_entry_zero_resources(report: &str, symbols: &BTreeSet<String>, label: &str) {
    let mut property_headers = BTreeMap::<String, usize>::new();
    let mut properties = BTreeMap::<String, Vec<(u64, u64, u64)>>::new();
    let mut usage = BTreeMap::<String, Vec<(u64, u64)>>::new();
    let mut compiled = BTreeMap::<String, usize>::new();
    let mut current = None;
    for line in report.lines() {
        if let Some((_, tail)) = line.split_once("Compiling entry function '")
            && let Some((symbol, _)) = tail.split_once('\'')
        {
            *compiled.entry(symbol.to_owned()).or_default() += 1;
        }
        if let Some((_, symbol)) = line.split_once("Function properties for ") {
            let symbol = symbol.trim().to_owned();
            *property_headers.entry(symbol.clone()).or_default() += 1;
            current = Some(symbol);
            continue;
        }
        let Some(symbol) = current.as_ref() else {
            continue;
        };
        let stack = metric_before(line, " bytes stack frame");
        let stores = metric_before(line, " bytes spill stores");
        let loads = metric_before(line, " bytes spill loads");
        if let (Some(stack), Some(stores), Some(loads)) = (stack, stores, loads) {
            properties
                .entry(symbol.clone())
                .or_default()
                .push((stack, stores, loads));
        }
        if let Some(registers) = metric_after(line, "Used ", " registers") {
            let shared = metric_before(line, " bytes smem").unwrap_or(0);
            usage
                .entry(symbol.clone())
                .or_default()
                .push((registers, shared));
            current = None;
        }
    }
    for symbol in symbols {
        assert_eq!(
            compiled.get(symbol),
            Some(&1),
            "{label}/{symbol} requires exactly one ptxas compile record"
        );
        assert_eq!(
            property_headers.get(symbol),
            Some(&1),
            "{label}/{symbol} requires exactly one ptxas property header"
        );
        let records = properties.get(symbol).map(Vec::as_slice);
        if loaded_nvrtc_version() < (12, 9)
            && matches!(records, Some([(stack, stores, loads)]) if (*stack, *stores, *loads) != (0, 0, 0))
        {
            // Below CUDA 12.9 the assembler spills this symbol; the loader
            // excludes it on that toolkit, so the record is reported here.
            eprintln!("{label}/{symbol} uses local resources on this toolkit: {records:?}");
        } else {
            assert_eq!(
                records,
                Some(&[(0, 0, 0)][..]),
                "{label}/{symbol} requires an explicit zero stack/spill ptxas record"
            );
        }
        let records = usage
            .get(symbol)
            .unwrap_or_else(|| panic!("{label}/{symbol} missing ptxas register/shared usage"));
        assert_eq!(records.len(), 1, "{label}/{symbol} duplicate ptxas usage");
        let (registers, shared) = records[0];
        let (register_cap, shared_cap) = tf32_resource_caps(symbol);
        assert!(
            registers <= register_cap && shared <= shared_cap,
            "{label}/{symbol} exceeds ptxas caps REG {registers}/{register_cap}, shared {shared}/{shared_cap}"
        );
    }
}

fn sass_entry<'a>(sass: &'a str, symbol: &str) -> &'a str {
    let function_marker = format!("Function : {symbol}");
    if let Some(start) = sass.find(&function_marker) {
        let tail = &sass[start..];
        let end = tail[function_marker.len()..]
            .find("Function : ")
            .map(|offset| function_marker.len() + offset)
            .unwrap_or(tail.len());
        return &tail[..end];
    }
    let label_marker = format!("\n{symbol}:\n");
    let start = sass
        .find(&label_marker)
        .map(|offset| offset + 1)
        .unwrap_or_else(|| panic!("SASS is missing function {symbol}"));
    let tail = &sass[start..];
    let end = tail
        .find("\n//--------------------- .text.")
        .or_else(|| tail.find("\n\t.section\t.text."))
        .unwrap_or(tail.len());
    &tail[..end]
}

fn dot_function_graph<'a>(dot: &'a str, symbol: &str) -> &'a str {
    let markers = [
        format!("subgraph \"cluster_{symbol}\""),
        format!("digraph \"{symbol}\""),
    ];
    assert_eq!(
        markers
            .iter()
            .map(|marker| dot.matches(marker).count())
            .sum::<usize>(),
        1,
        "{symbol} requires exactly one nvdisasm DOT graph"
    );
    let (start, _) = markers
        .iter()
        .find_map(|marker| dot.find(marker).map(|start| (start, marker)))
        .expect("checked DOT function marker");
    let open = dot[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("DOT function opening brace");
    let mut depth = 0_u32;
    let mut quoted = false;
    let mut escaped = false;
    for (relative, byte) in dot.as_bytes()[open..].iter().copied().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &dot[start..=open + relative];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated nvdisasm DOT graph for {symbol}")
}

fn dot_identifier(source: &str) -> String {
    let source = source.trim();
    if let Some(quoted) = source
        .rsplit_once('\n')
        .map_or(source, |(_, tail)| tail)
        .strip_prefix('"')
        && let Some(end) = quoted.find('"')
    {
        return quoted[..end].to_owned();
    }
    source
        .split_ascii_whitespace()
        .last()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .trim_matches('"')
        .to_owned()
}

fn parse_dot_cfg(
    graph: &str,
    symbol: &str,
) -> (BTreeMap<String, String>, BTreeMap<String, Vec<String>>) {
    let mut nodes = BTreeMap::new();
    let mut successors = BTreeMap::<String, Vec<String>>::new();
    let mut previous_line = None;
    for line in graph.lines() {
        if line.trim_start().starts_with("[label=")
            && let Some(identifier_line) = previous_line
        {
            let identifier = dot_identifier(identifier_line);
            if !identifier.is_empty() {
                assert!(
                    nodes.insert(identifier.clone(), line.to_owned()).is_none(),
                    "{symbol} duplicate DOT node {identifier}"
                );
            }
        }
        previous_line = Some(line);
    }
    let mut statements = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, byte) in graph.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b';' {
            statements.push(&graph[start..offset]);
            start = offset + 1;
        }
    }
    statements.push(&graph[start..]);
    for statement in statements {
        if let Some((from, tail)) = statement.split_once("->") {
            let from = dot_identifier(from);
            let target = dot_identifier(tail.split_once('[').map_or(tail, |(head, _)| head));
            if !from.is_empty() && !target.is_empty() {
                successors.entry(from).or_default().push(target);
            }
        } else if let Some((identifier, attributes)) = statement.split_once('[')
            && attributes.contains("label=")
        {
            let identifier = dot_identifier(identifier);
            if !identifier.is_empty() && !nodes.contains_key(&identifier) {
                nodes.insert(identifier, attributes.to_owned());
            }
        }
    }
    assert!(
        !nodes.is_empty(),
        "{symbol} nvdisasm DOT has no basic blocks"
    );
    for (from, targets) in &mut successors {
        assert!(
            nodes.contains_key(from),
            "{symbol} DOT edge from unknown {from}; known={:?}",
            nodes.keys().collect::<Vec<_>>()
        );
        targets.sort();
        targets.dedup();
        for target in targets.iter() {
            assert!(
                nodes.contains_key(target),
                "{symbol} DOT edge to unknown {target}; known={:?}",
                nodes.keys().collect::<Vec<_>>()
            );
        }
    }
    for node in nodes.keys() {
        successors.entry(node.clone()).or_default();
    }
    (nodes, successors)
}

fn sass_normal_successors(
    nodes: &BTreeMap<String, String>,
    successors: &BTreeMap<String, Vec<String>>,
    symbol: &str,
) -> BTreeMap<String, Vec<String>> {
    let mut normal = successors.clone();
    for (node, body) in nodes {
        if body.contains("CALL.REL.NOINC") && body.contains("__cuda_sm10x_tcgen05_guardrail_trap_")
        {
            assert_eq!(
                body.matches("CALL.REL.NOINC").count(),
                1,
                "{symbol} TCGEN trap block {node} must contain one call"
            );
            assert_eq!(
                successors[node].len(),
                1,
                "{symbol} TCGEN trap block {node} must expose one synthetic fallthrough"
            );
            normal.get_mut(node).expect("validated DOT node").clear();
        }
    }
    normal
}

const K0_GUARD_ANCHOR: (&str, u64) = ("mamba_tf32_k0_guard", 1001);
const K0_BRANCH_ANCHOR: (&str, u64) = ("mamba_tf32_k0_branch", 1002);
const K0_ZERO_STORE_ANCHOR: (&str, u64) = ("mamba_tf32_k0_zero_store", 2001);

#[derive(Clone, Debug)]
struct SassInstruction<'a> {
    offset: u64,
    predicate: Option<(&'a str, bool)>,
    mnemonic: &'a str,
    operands: &'a str,
    file: String,
    line: u64,
}

type ParsedSassInstruction<'a> = (u64, Option<(&'a str, bool)>, &'a str, &'a str);

fn sass_line_directive(line: &str) -> Option<(String, u64)> {
    let (_, tail) = line.split_once("//## File \"")?;
    let (file, tail) = tail.split_once("\", line ")?;
    Some((
        file.to_owned(),
        tail.split_ascii_whitespace()
            .next()?
            .parse()
            .expect("nvdisasm source line number"),
    ))
}

fn sass_instruction(line: &str) -> Option<ParsedSassInstruction<'_>> {
    let (_, tail) = line.split_once("/*")?;
    let (offset, instruction) = tail.split_once("*/")?;
    let offset = u64::from_str_radix(offset.trim(), 16).ok()?;
    let mut instruction = instruction.trim();
    let predicate = instruction.starts_with('@').then(|| {
        let (guard, remainder) = instruction
            .split_once(char::is_whitespace)
            .expect("predicated SASS instruction body");
        instruction = remainder.trim_start();
        let guard = guard.strip_prefix('@').expect("checked predicate prefix");
        let (guard, negated) = guard
            .strip_prefix('!')
            .map_or((guard, false), |guard| (guard, true));
        (guard, negated)
    });
    let (mnemonic, operands) = instruction
        .split_once(char::is_whitespace)
        .unwrap_or((instruction.trim_end_matches(';'), ""));
    Some((
        offset,
        predicate,
        mnemonic,
        operands.trim().trim_end_matches(';').trim(),
    ))
}

fn sass_line_instructions<'a>(entry: &'a str, symbol: &str) -> Vec<SassInstruction<'a>> {
    let mut location = None;
    let mut instructions = Vec::new();
    let mut offsets = BTreeSet::new();
    for line in entry.lines() {
        if let Some(next) = sass_line_directive(line) {
            location = Some(next);
            continue;
        }
        let Some((offset, predicate, mnemonic, operands)) = sass_instruction(line) else {
            continue;
        };
        assert!(
            offsets.insert(offset),
            "{symbol} duplicate SASS instruction offset {offset:x}"
        );
        let Some((file, source_line)) = location.as_ref() else {
            continue;
        };
        instructions.push(SassInstruction {
            offset,
            predicate,
            mnemonic,
            operands,
            file: file.clone(),
            line: *source_line,
        });
    }
    assert!(!instructions.is_empty(), "{symbol} has no line-mapped SASS");
    instructions
}

fn dot_instruction_offsets(body: &str) -> BTreeSet<u64> {
    let bytes = body.as_bytes();
    let mut offsets = BTreeSet::new();
    for colon in bytes
        .iter()
        .enumerate()
        .filter_map(|(index, byte)| (*byte == b':').then_some(index))
    {
        let mut start = colon;
        while start > 0 && bytes[start - 1].is_ascii_hexdigit() {
            start -= 1;
        }
        if colon - start >= 4
            && start > 0
            && (matches!(bytes[start - 1], b'>' | b'l' | b'"' | b'|')
                || bytes[start - 1].is_ascii_whitespace())
            && let Ok(offset) = u64::from_str_radix(&body[start..colon], 16)
        {
            offsets.insert(offset);
        }
    }
    offsets
}

fn sass_offset_node(
    nodes: &BTreeMap<String, String>,
    offset: u64,
    symbol: &str,
    role: &str,
) -> String {
    let matches: Vec<_> = nodes
        .iter()
        .filter_map(|(node, body)| {
            dot_instruction_offsets(body)
                .contains(&offset)
                .then_some(node.clone())
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "{symbol} {role} offset {offset:x} must bind one DOT node"
    );
    matches[0].clone()
}

fn unique_sass_offset_node(nodes: &BTreeMap<String, String>, offset: u64) -> Option<String> {
    let mut matches = nodes.iter().filter_map(|(node, body)| {
        dot_instruction_offsets(body)
            .contains(&offset)
            .then_some(node)
    });
    let node = matches.next()?.clone();
    matches.next().is_none().then_some(node)
}

fn source_anchor(source: &str, anchor: (&str, u64), label: &str) {
    let directive = format!("#line {} \"{}\"", anchor.1, anchor.0);
    assert_eq!(
        source
            .lines()
            .filter(|line| line.trim() == directive)
            .count(),
        1,
        "{label} requires exactly one {directive}"
    );
}

fn sass_writes_predicate(instruction: &SassInstruction<'_>, predicate: &str) -> bool {
    let mnemonic = instruction
        .mnemonic
        .split_once('.')
        .map_or(instruction.mnemonic, |(mnemonic, _)| mnemonic);
    let operands = ptx_operands(instruction.operands);
    operands.first() == Some(&predicate)
        || (mnemonic.ends_with("SETP") && operands.get(1) == Some(&predicate))
        || mnemonic == "R2P"
}

fn sass_branch_predicate<'a>(instruction: &'a SassInstruction<'a>) -> Option<(&'a str, bool)> {
    if !instruction.mnemonic.starts_with("BRA") {
        return None;
    }
    if let Some(predicate) = instruction.predicate {
        return Some(predicate);
    }
    if instruction.mnemonic != "BRA.U" {
        return None;
    }
    let operands = ptx_operands(instruction.operands);
    if operands.len() < 2 {
        return None;
    }
    let (predicate, negated) = operands[0]
        .strip_prefix('!')
        .map_or((operands[0], false), |predicate| (predicate, true));
    (sass_register(predicate, "UP", "UPT") && predicate != "UPT").then_some((predicate, negated))
}

fn sass_guard_candidates(instructions: &[SassInstruction<'_>]) -> Vec<(u64, u64)> {
    let mut candidates = Vec::new();
    for (branch_index, branch) in instructions.iter().enumerate() {
        let Some((predicate, _)) = sass_branch_predicate(branch) else {
            continue;
        };
        let Some(compare) = instructions[..branch_index]
            .iter()
            .rev()
            .find(|instruction| sass_writes_predicate(instruction, predicate))
        else {
            continue;
        };
        if compare.predicate.is_none()
            && (compare.mnemonic.starts_with("ISETP") || compare.mnemonic.starts_with("UISETP"))
        {
            candidates.push((compare.offset, branch.offset));
        }
    }
    candidates
}

const SASS_K0_PROTECTED: [&str; 14] = [
    "UTMALDG",
    "HGMMA",
    "WGMMA",
    "UTCHMMA",
    "TCGEN",
    "TMEM",
    "LDTM",
    "STTM",
    "HMMA",
    "LDGSTS",
    "BAR.",
    "UTCATOMSWS",
    "UVIRTCOUNT",
    "ATOMS.",
];

struct SassK0Selection {
    compare_offset: u64,
    branch_offset: u64,
    guard: String,
    regions: Vec<BTreeSet<String>>,
    region_bodies: Vec<String>,
    zero_index: usize,
}

fn select_sass_k0_guard(
    nodes: &BTreeMap<String, String>,
    successors: &BTreeMap<String, Vec<String>>,
    instructions: &[SassInstruction<'_>],
    symbol: &str,
    label: &str,
) -> SassK0Selection {
    let (transfer, matrix): (&[&str], &[&str]) = match label {
        "SM80" => (&["LDGSTS"], &["HMMA"]),
        "SM90a" => (&["UTMALDG"], &["HGMMA", "WGMMA"]),
        "SM100" => (&["UTMALDG"], &["UTCHMMA", "TCGEN"]),
        "SM120" => (&["UTMALDG"], &["HMMA"]),
        _ => panic!("unknown SASS CFG family {label}"),
    };
    let reachable = |start: &str| {
        let mut seen = BTreeSet::new();
        let mut pending = vec![start.to_owned()];
        while let Some(node) = pending.pop() {
            if seen.insert(node.clone()) {
                pending.extend(successors[&node].iter().cloned());
            }
        }
        seen
    };
    let mut matches = Vec::new();
    for (compare_offset, branch_offset) in sass_guard_candidates(instructions) {
        let Some(guard) = unique_sass_offset_node(nodes, branch_offset) else {
            continue;
        };
        if unique_sass_offset_node(nodes, compare_offset).as_ref() != Some(&guard)
            || successors[&guard].len() != 2
        {
            continue;
        }
        let regions: Vec<_> = successors[&guard]
            .iter()
            .map(|successor| reachable(successor))
            .collect();
        let region_bodies: Vec<_> = regions
            .iter()
            .map(|region| {
                region
                    .iter()
                    .map(|block| nodes[block].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect();
        let zero_indices: Vec<_> = (0..2)
            .filter(|index| {
                let body = &region_bodies[*index];
                body.contains("STG")
                    && body.contains("EXIT")
                    && !SASS_K0_PROTECTED.iter().any(|opcode| body.contains(opcode))
            })
            .collect();
        if zero_indices.len() != 1 {
            continue;
        }
        let zero_index = zero_indices[0];
        let nonzero = &region_bodies[1 - zero_index];
        if !transfer.iter().any(|opcode| nonzero.contains(opcode))
            || !matrix.iter().any(|opcode| nonzero.contains(opcode))
        {
            continue;
        }
        matches.push(SassK0Selection {
            compare_offset,
            branch_offset,
            guard,
            regions,
            region_bodies,
            zero_index,
        });
    }
    assert!(
        !matches.is_empty(),
        "{label}/{symbol} requires a compare/branch with epilogue-only and matrix successors"
    );
    matches.sort_by_key(|selection| selection.branch_offset);
    matches.remove(0)
}

fn assert_sass_cfg_corroboration(
    dot: &str,
    line_sass: &str,
    source: &str,
    symbol: &str,
    label: &str,
) {
    let graph = dot_function_graph(dot, symbol);
    let (nodes, raw_successors) = parse_dot_cfg(graph, symbol);
    let successors = sass_normal_successors(&nodes, &raw_successors, symbol);
    let line_entry = sass_entry(line_sass, symbol);
    source_anchor(source, K0_GUARD_ANCHOR, label);
    source_anchor(source, K0_BRANCH_ANCHOR, label);
    source_anchor(source, K0_ZERO_STORE_ANCHOR, label);
    let instructions = sass_line_instructions(line_entry, symbol);
    let selection = select_sass_k0_guard(&nodes, &successors, &instructions, symbol, label);
    let compare_offset = selection.compare_offset;
    let branch_offset = selection.branch_offset;
    let guard = selection.guard.clone();
    let compare = instructions
        .iter()
        .find(|instruction| instruction.offset == compare_offset)
        .expect("anchored compare instruction");
    let branch = instructions
        .iter()
        .find(|instruction| instruction.offset == branch_offset)
        .expect("anchored branch instruction");
    let compare_operands = ptx_operands(compare.operands);
    assert!(
        compare_operands.first().is_some_and(|predicate| {
            (sass_register(predicate, "P", "PT") && *predicate != "PT")
                || (sass_register(predicate, "UP", "UPT") && *predicate != "UPT")
        }),
        "{label}/{symbol} K=0 compare has no concrete predicate destination"
    );
    let compare_predicate = compare_operands[0];
    let (branch_predicate, _) = sass_branch_predicate(branch).expect("anchored conditional branch");
    assert_eq!(
        branch_predicate, compare_predicate,
        "{label}/{symbol} K=0 branch does not consume the compare predicate"
    );
    assert!(
        compare_offset < branch_offset
            && !instructions.iter().any(|instruction| {
                compare_offset < instruction.offset
                    && instruction.offset < branch_offset
                    && sass_writes_predicate(instruction, compare_predicate)
            }),
        "{label}/{symbol} K=0 predicate is overwritten before the branch"
    );
    let raw_targets: BTreeSet<_> = raw_successors.values().flatten().cloned().collect();
    let entries: Vec<_> = nodes
        .keys()
        .filter(|node| !raw_targets.contains(*node))
        .cloned()
        .collect();
    assert_eq!(entries.len(), 1, "{label}/{symbol} DOT entry block");
    let entry = &entries[0];
    let mut reachable = BTreeSet::new();
    let mut pending = vec![entry.clone()];
    while let Some(node) = pending.pop() {
        if reachable.insert(node.clone()) {
            pending.extend(successors[&node].iter().cloned());
        }
    }
    let mut predecessors: BTreeMap<_, Vec<_>> = reachable
        .iter()
        .map(|node| (node.clone(), Vec::new()))
        .collect();
    for from in &reachable {
        for target in &successors[from] {
            if reachable.contains(target) {
                predecessors
                    .get_mut(target)
                    .expect("reachable DOT target")
                    .push(from.clone());
            }
        }
    }
    let mut dominators: BTreeMap<String, BTreeSet<String>> = reachable
        .iter()
        .map(|node| {
            let initial = if node == entry {
                BTreeSet::from([node.clone()])
            } else {
                reachable.clone()
            };
            (node.clone(), initial)
        })
        .collect();
    loop {
        let mut changed = false;
        for node in reachable.iter().filter(|node| *node != entry) {
            let incoming = &predecessors[node];
            let first = incoming.first().unwrap_or_else(|| {
                panic!("{label}/{symbol} reachable node {node} has no predecessor")
            });
            let mut next = dominators[first].clone();
            for predecessor in incoming.iter().skip(1) {
                next = next
                    .intersection(&dominators[predecessor])
                    .cloned()
                    .collect();
            }
            next.insert(node.clone());
            if next != dominators[node] {
                dominators.insert(node.clone(), next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    assert!(
        nodes[&guard].contains("BRA") && successors[&guard].len() == 2,
        "{label}/{symbol} selected guard must be a conditional branch node"
    );
    let zero_index = selection.zero_index;
    let zero_body = &selection.region_bodies[zero_index];
    assert!(
        !SASS_K0_PROTECTED
            .iter()
            .any(|opcode| zero_body.contains(opcode))
            && zero_body.contains("EXIT"),
        "{label}/{symbol} anchored zero SASS region is unsafe or unterminated"
    );
    if label == "SM100"
        && symbol.contains("_sm100_tcgen_tf32_")
        && loaded_nvrtc_version() >= (12, 9)
    {
        assert_tcgen_management_cfg(
            &nodes,
            &successors,
            &dominators,
            &instructions,
            &selection.regions[zero_index],
            &selection.regions[1 - zero_index],
            symbol,
        );
    }
    for (node, body) in &nodes {
        if reachable.contains(node) && SASS_K0_PROTECTED.iter().any(|opcode| body.contains(opcode))
        {
            assert!(
                dominators[node].contains(&guard),
                "{label}/{symbol} SASS guard does not dominate protected node {node}"
            );
        }
    }
}

fn sass_register(token: &str, prefix: &str, zero: &str) -> bool {
    token == zero
        || token.strip_prefix(prefix).is_some_and(|index| {
            !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn shared_atomic_address(operand: &str) -> Option<(&str, u64)> {
    let address = operand.strip_prefix('[')?.strip_suffix(']')?;
    let (base, displacement) = address.split_once('+')?;
    if !sass_register(base, "R", "RZ") || base == "RZ" {
        return None;
    }
    let displacement = displacement.strip_prefix("0x")?;
    Some((base, u64::from_str_radix(displacement, 16).ok()?))
}

fn tmem_guardrail_address(operand: &str) -> Option<(&str, u64)> {
    let address = operand.strip_prefix('[')?.strip_suffix(']')?;
    let (base, displacement) = address.split_once('+')?;
    if !sass_register(base, "UR", "URZ") || base == "URZ" {
        return None;
    }
    let displacement = displacement.strip_prefix("0x")?;
    Some((base, u64::from_str_radix(displacement, 16).ok()?))
}

type TcgenManagement<'a> = (
    Vec<&'a SassInstruction<'a>>,
    Vec<&'a SassInstruction<'a>>,
    &'a SassInstruction<'a>,
);

fn tcgen_management_syntax<'a>(
    instructions: &'a [SassInstruction<'a>],
    symbol: &str,
) -> TcgenManagement<'a> {
    let finds: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "UTCATOMSWS.FIND_AND_SET.ALIGN")
        .collect();
    assert!(!finds.is_empty(), "{symbol} has no TCGEN allocation FIND");
    let mut uniform_register = None;
    for find in &finds {
        let operands = ptx_operands(find.operands);
        assert!(
            operands.len() == 3
                && sass_register(operands[0], "UP", "UPT")
                && operands[0] != "UPT"
                && sass_register(operands[1], "UR", "URZ")
                && operands[1] != "URZ"
                && operands[1] == operands[2],
            "{symbol} malformed {} {}",
            find.mnemonic,
            find.operands
        );
        if let Some(expected) = uniform_register {
            assert_eq!(operands[1], expected, "{symbol} allocation register drift");
        } else {
            uniform_register = Some(operands[1]);
        }
    }

    let ors: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "ATOMS.OR")
        .collect();
    assert_eq!(ors.len(), 2, "{symbol} requires the paired TCGEN ATOMS.OR");
    let mut bases = BTreeSet::new();
    let mut displacements = BTreeSet::new();
    for atomic in &ors {
        let operands = ptx_operands(atomic.operands);
        assert!(
            operands.len() == 3
                && operands[0] == "RZ"
                && sass_register(operands[2], "R", "RZ")
                && operands[2] != "RZ",
            "{symbol} malformed ATOMS.OR {}",
            atomic.operands
        );
        let (base, displacement) = shared_atomic_address(operands[1])
            .unwrap_or_else(|| panic!("{symbol} malformed ATOMS.OR address {}", operands[1]));
        bases.insert(base);
        displacements.insert(displacement);
    }
    assert_eq!(bases.len(), 1, "{symbol} ATOMS.OR base registers differ");
    assert_eq!(
        displacements,
        BTreeSet::from([0x14, 0x18]),
        "{symbol} ATOMS.OR management offsets"
    );

    let deallocations: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "UTCATOMSWS.AND")
        .collect();
    assert_eq!(
        deallocations.len(),
        1,
        "{symbol} requires one TCGEN deallocation"
    );
    let deallocation = deallocations[0];
    let operands = ptx_operands(deallocation.operands);
    assert!(
        operands.len() == 2 && operands[0] == "URZ" && Some(operands[1]) == uniform_register,
        "{symbol} malformed TCGEN deallocation {}",
        deallocation.operands
    );

    let guardrail_ands: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "ATOMS.AND")
        .collect();
    let mut guardrail_offsets = BTreeSet::new();
    if !guardrail_ands.is_empty() {
        assert_eq!(
            guardrail_ands.len(),
            2,
            "{symbol} requires a complete TCGEN guardrail pair"
        );
        let mut bases = BTreeSet::new();
        let mut displacements = BTreeSet::new();
        for guardrail in guardrail_ands {
            let operands = ptx_operands(guardrail.operands);
            assert!(
                guardrail.predicate.is_some()
                    && operands.len() == 3
                    && operands[0] == "RZ"
                    && sass_register(operands[2], "R", "RZ")
                    && operands[2] != "RZ",
                "{symbol} malformed TCGEN guardrail {}",
                guardrail.operands
            );
            let (base, displacement) = tmem_guardrail_address(operands[1]).unwrap_or_else(|| {
                panic!("{symbol} malformed TCGEN guardrail address {}", operands[1])
            });
            bases.insert(base);
            displacements.insert(displacement);
            guardrail_offsets.insert(guardrail.offset);
        }
        assert_eq!(bases.len(), 1, "{symbol} guardrail base registers differ");
        assert_eq!(
            displacements,
            BTreeSet::from([0x14, 0x18]),
            "{symbol} guardrail management offsets"
        );
    }

    for (index, instruction) in instructions.iter().enumerate() {
        let mnemonic = instruction.mnemonic;
        let operands = ptx_operands(instruction.operands);
        let management_window =
            &instructions[index.saturating_sub(4)..(index + 5).min(instructions.len())];
        let allowed_management_redux = mnemonic == "REDUX"
            && operands.len() == 2
            && sass_register(operands[0], "UR", "URZ")
            && operands[0] != "URZ"
            && sass_register(operands[1], "R", "RZ")
            && operands[1] != "RZ"
            && management_window
                .iter()
                .any(|nearby| nearby.mnemonic == "UTCATOMSWS.AND");
        let allowed_management = guardrail_offsets.contains(&instruction.offset)
            || allowed_management_redux
            || matches!(
                mnemonic,
                "UTCATOMSWS.FIND_AND_SET.ALIGN" | "UTCATOMSWS.AND" | "ATOMS.OR"
            );
        let numeric_atomic_or_reduction = mnemonic.contains("ATOM")
            || mnemonic.starts_with("RED")
            || mnemonic.starts_with("URED")
            || mnemonic.starts_with("SURED");
        assert!(
            allowed_management || !mnemonic.contains("ATOMSWS"),
            "{symbol} unknown TCGEN management mnemonic {mnemonic}"
        );
        assert!(
            allowed_management || !numeric_atomic_or_reduction,
            "{symbol} numeric atomic/reduction instruction {instruction:?}; neighborhood={:?}",
            &instructions[index.saturating_sub(6)..(index + 7).min(instructions.len())]
        );
    }
    (finds, ors, deallocation)
}

fn assert_tcgen_management_cfg(
    nodes: &BTreeMap<String, String>,
    successors: &BTreeMap<String, Vec<String>>,
    dominators: &BTreeMap<String, BTreeSet<String>>,
    instructions: &[SassInstruction<'_>],
    zero_region: &BTreeSet<String>,
    nonzero_region: &BTreeSet<String>,
    symbol: &str,
) {
    let (finds, ors, deallocation) = tcgen_management_syntax(instructions, symbol);
    let mut allocation_nodes = BTreeSet::new();
    for atomic in &ors {
        allocation_nodes.insert(sass_offset_node(
            nodes,
            atomic.offset,
            symbol,
            "TCGEN allocation OR",
        ));
    }
    assert_eq!(
        allocation_nodes.len(),
        1,
        "{symbol} paired ATOMS.OR must share one allocation block"
    );
    let allocation_node = allocation_nodes
        .pop_first()
        .expect("one TCGEN allocation node");
    let first_or = ors
        .iter()
        .map(|instruction| instruction.offset)
        .min()
        .expect("paired ATOMS.OR");
    let last_or = ors
        .iter()
        .map(|instruction| instruction.offset)
        .max()
        .expect("paired ATOMS.OR");
    assert!(
        finds.iter().any(|find| {
            let find_node = sass_offset_node(nodes, find.offset, symbol, "TCGEN allocation FIND");
            dominators[&allocation_node].contains(&find_node)
                && (find_node != allocation_node || find.offset < first_or)
        }),
        "{symbol} ATOMS.OR allocation block is not dominated by FIND"
    );

    let deallocation_node =
        sass_offset_node(nodes, deallocation.offset, symbol, "TCGEN deallocation");
    assert_ne!(
        allocation_node, deallocation_node,
        "{symbol} TCGEN allocation/deallocation block alias"
    );
    assert!(
        nonzero_region.contains(&allocation_node)
            && nonzero_region.contains(&deallocation_node)
            && !zero_region.contains(&allocation_node)
            && !zero_region.contains(&deallocation_node),
        "{symbol} TCGEN management must be confined to the nonzero region"
    );
    let matrix: Vec<_> = instructions
        .iter()
        .filter(|instruction| {
            instruction.mnemonic.starts_with("UTCHMMA")
                || instruction.mnemonic.starts_with("TCGEN")
                || instruction.mnemonic.starts_with("TMEM")
                || instruction.mnemonic.starts_with("LDTM")
                || instruction.mnemonic.starts_with("STTM")
        })
        .map(|instruction| {
            (
                instruction,
                sass_offset_node(nodes, instruction.offset, symbol, "TCGEN matrix"),
            )
        })
        .collect();
    assert!(!matrix.is_empty(), "{symbol} has no TCGEN matrix work");
    let mut allocation_reachable = BTreeSet::new();
    let mut pending = vec![allocation_node.clone()];
    while let Some(node) = pending.pop() {
        if allocation_reachable.insert(node.clone()) {
            pending.extend(successors[&node].iter().cloned());
        }
    }
    let allocation_barrier = instructions
        .iter()
        .filter(|instruction| {
            instruction.mnemonic.starts_with("BAR.SYNC") && last_or < instruction.offset
        })
        .filter_map(|instruction| {
            let node = sass_offset_node(
                nodes,
                instruction.offset,
                symbol,
                "TCGEN allocation barrier",
            );
            (allocation_reachable.contains(&node)
                && matrix
                    .iter()
                    .all(|(_, matrix_node)| dominators[matrix_node].contains(&node)))
            .then_some((instruction, node))
        })
        .min_by_key(|(instruction, _)| instruction.offset)
        .unwrap_or_else(|| panic!("{symbol} has no allocation-to-matrix CTA barrier"));
    for (instruction, matrix_node) in &matrix {
        assert!(
            nonzero_region.contains(matrix_node)
                && dominators[matrix_node].contains(&allocation_barrier.1)
                && (matrix_node != &allocation_barrier.1
                    || allocation_barrier.0.offset < instruction.offset),
            "{symbol} TCGEN allocation barrier does not dominate matrix offset {:x}",
            instruction.offset
        );
    }

    let normal_exits: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "EXIT")
        .filter_map(|instruction| {
            let node = sass_offset_node(nodes, instruction.offset, symbol, "normal exit");
            nonzero_region
                .contains(&node)
                .then_some((instruction, node))
        })
        .collect();
    assert!(
        !normal_exits.is_empty(),
        "{symbol} has no normal nonzero exit"
    );
    let normal_exit_nodes: BTreeSet<_> =
        normal_exits.iter().map(|(_, node)| node.clone()).collect();
    let last_matrix = matrix
        .iter()
        .map(|(instruction, _)| instruction.offset)
        .max()
        .expect("TCGEN matrix work");
    let release_barrier = instructions
        .iter()
        .filter(|instruction| {
            instruction.mnemonic.starts_with("BAR.SYNC") && last_matrix < instruction.offset
        })
        .filter_map(|instruction| {
            let node = sass_offset_node(nodes, instruction.offset, symbol, "TCGEN release barrier");
            (nonzero_region.contains(&node)
                && normal_exits.iter().all(|(exit, exit_node)| {
                    dominators[exit_node].contains(&node)
                        && (exit_node != &node || instruction.offset < exit.offset)
                }))
            .then_some((instruction, node))
        })
        .min_by_key(|(instruction, _)| instruction.offset)
        .unwrap_or_else(|| panic!("{symbol} has no matrix-to-exit CTA barrier"));

    let physical_deallocations: Vec<_> = instructions
        .iter()
        .filter(|instruction| instruction.mnemonic.starts_with("UVIRTCOUNT.DEALLOC"))
        .collect();
    assert_eq!(
        physical_deallocations.len(),
        1,
        "{symbol} requires one physical TCGEN deallocation"
    );
    let physical_deallocation = physical_deallocations[0];
    let physical_deallocation_node = sass_offset_node(
        nodes,
        physical_deallocation.offset,
        symbol,
        "physical TCGEN deallocation",
    );
    assert!(
        dominators[&physical_deallocation_node].contains(&release_barrier.1)
            && (physical_deallocation_node != release_barrier.1
                || release_barrier.0.offset < physical_deallocation.offset),
        "{symbol} physical TCGEN deallocation precedes the release barrier"
    );
    assert!(
        dominators[&deallocation_node].contains(&physical_deallocation_node)
            && (deallocation_node != physical_deallocation_node
                || physical_deallocation.offset < deallocation.offset),
        "{symbol} guardrail cleanup precedes physical TCGEN deallocation"
    );

    let terminal_exits: Vec<_> = normal_exits
        .iter()
        .filter(|(_, node)| successors[node].is_empty())
        .collect();
    assert!(
        terminal_exits.iter().any(|(instruction, node)| {
            dominators[node].contains(&deallocation_node)
                && (node != &deallocation_node || deallocation.offset < instruction.offset)
        }),
        "{symbol} has no terminal exit after TCGEN deallocation"
    );
    for (instruction, exit_node) in &normal_exits {
        assert!(
            dominators[exit_node].contains(&release_barrier.1)
                && (exit_node != &release_barrier.1
                    || release_barrier.0.offset < instruction.offset),
            "{symbol} normal exit precedes the TCGEN release barrier"
        );
    }
    for node in nonzero_region
        .iter()
        .filter(|node| successors[*node].is_empty())
    {
        assert!(
            normal_exit_nodes.contains(node)
                || (nodes[node].contains("CALL.REL.NOINC")
                    && nodes[node].contains("__cuda_sm10x_tcgen05_guardrail_trap_")),
            "{symbol} nonzero CFG has unknown sink {node}"
        );
    }
}

fn assert_sass_entry_contract(sass: &str, symbol: &str, label: &str) {
    let entry = sass_entry(sass, symbol);
    if loaded_nvrtc_version() < (12, 9) && (entry.contains("LDL") || entry.contains("STL")) {
        // Below CUDA 12.9 the assembler spills this symbol; the loader
        // excludes it on that toolkit, so the local traffic is reported here.
        eprintln!("{label}/{symbol} uses local memory on this toolkit");
    } else {
        assert!(
            !entry.contains("LDL") && !entry.contains("STL"),
            "{label}/{symbol} must not use local memory"
        );
    }
    let instructions = sass_line_instructions(entry, symbol);
    if label == "SM100" && symbol.contains("_sm100_tcgen_tf32_") {
        // CUDA 12.8 assembles the tcgen allocation with a different pairing;
        // the family is not offered on that toolkit, so its SASS shape is
        // not part of the contract there.
        if loaded_nvrtc_version() < (12, 9) {
            eprintln!("{label}/{symbol}: tcgen management pin skipped below CUDA 12.9");
            return;
        }
        tcgen_management_syntax(&instructions, symbol);
    } else {
        for instruction in &instructions {
            let mnemonic = instruction.mnemonic;
            assert!(
                !mnemonic.contains("ATOM")
                    && !mnemonic.starts_with("RED")
                    && !mnemonic.starts_with("URED")
                    && !mnemonic.starts_with("SURED")
                    && !mnemonic.contains("ATOMSWS"),
                "{label}/{symbol} contains atomic/reduction mnemonic {mnemonic}"
            );
        }
    }
    if symbol.starts_with("nn_") {
        assert!(entry.contains("FFMA") && entry.contains("FMUL"));
    } else if symbol.starts_with("tn_") {
        assert!(
            entry.contains("FFMA"),
            "{label}/{symbol} TN is missing FFMA"
        );
    } else {
        assert!(
            entry.contains("FMUL"),
            "{label}/{symbol} NT is missing FMUL"
        );
    }
}

fn compact_cuda_executable_scope(source: &str, marker: &str) -> String {
    source_mask(braced_scope_after(source, marker))
        .lines()
        .filter(|line| !line.trim_start().starts_with("#line"))
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn assert_cuda_bundle_layout(source: &str, name: &str, fields: &[(&str, &str)], bytes: usize) {
    let structure = compact_code(braced_scope_after(source, &format!("struct {name}")));
    let mut cursor = 0;
    for (field_type, field) in fields {
        let declaration = format!("{field_type}{field};");
        let offset = structure[cursor..]
            .find(&declaration)
            .map(|relative| cursor + relative)
            .unwrap_or_else(|| panic!("{name} is missing ordered field {declaration}"));
        cursor = offset + declaration.len();
    }
    assert_eq!(
        structure.matches(';').count(),
        fields.len(),
        "{name} must contain only the frozen fields"
    );

    let code = compact_code(source);
    assert!(
        code.contains(&format!("static_assert(sizeof({name})=={bytes}")),
        "{name} must freeze size {bytes}"
    );
    assert!(
        code.contains(&format!("static_assert(alignof({name})==4")),
        "{name} must freeze alignment 4"
    );
    assert!(
        code.contains(&format!("static_assert(__is_standard_layout({name})")),
        "{name} must prove standard layout"
    );
    for (_, field) in fields {
        assert!(
            code.contains(&format!("static_assert(sizeof((({name}*)0)->{field})==4")),
            "{name} must freeze {field} at four bytes"
        );
    }
}

fn m16n8_owner(lane: usize, row_offset: usize, column_offset: usize) -> Vec<(usize, usize)> {
    let group = lane >> 2;
    let thread = lane & 3;
    vec![
        (row_offset + group, column_offset + 2 * thread),
        (row_offset + group, column_offset + 2 * thread + 1),
        (row_offset + group + 8, column_offset + 2 * thread),
        (row_offset + group + 8, column_offset + 2 * thread + 1),
    ]
}

fn assert_single_owner<F>(tile: (usize, usize), threads: usize, owner: F, label: &str)
where
    F: Fn(usize) -> Vec<(usize, usize)>,
{
    let (tile_rows, tile_columns) = tile;
    let mut counts = vec![0_u8; tile_rows * tile_columns];
    for thread in 0..threads {
        for (row, column) in owner(thread) {
            assert!(
                row < tile_rows && column < tile_columns,
                "{label} owner {thread} produced ({row},{column}) out of range"
            );
            counts[row * tile_columns + column] += 1;
        }
    }
    assert!(
        counts.iter().all(|count| *count == 1),
        "{label} must assign every output exactly once"
    );

    for (rows, columns) in [
        (1, 1),
        (tile_rows - 1, tile_columns - 1),
        (tile_rows, tile_columns),
        (tile_rows + 1, tile_columns + 1),
        (2 * tile_rows + 1, 2 * tile_columns + 1),
    ] {
        let row_tiles = rows.div_ceil(tile_rows);
        let column_tiles = columns.div_ceil(tile_columns);
        let mut grid_counts = vec![0_u8; row_tiles * column_tiles];
        for block in 0..row_tiles * column_tiles {
            let row_tile = block / column_tiles;
            let column_tile = block % column_tiles;
            grid_counts[row_tile * column_tiles + column_tile] += 1;
        }
        assert!(
            grid_counts.iter().all(|count| *count == 1),
            "{label} tail grid must assign every logical tile exactly once"
        );
    }
}

#[test]
fn hardware_artifact_expected_exports_cover_wide_without_changing_frozen_base() {
    const WIDE: &str = "nn_sm80_mma_tf32_m128n128_bk32_s3";
    assert_eq!(expected_sm80_symbols().len(), 18);
    assert!(!expected_sm80_symbols().contains(WIDE));
    for (cc, count, wide) in [
        ((8, 0), 19, true),
        ((8, 6), 19, true),
        ((8, 7), 19, true),
        ((8, 9), 19, true),
        ((9, 0), 25, true),
        ((10, 0), 55, true),
        ((10, 3), 55, true),
        ((10, 7), 55, true),
        ((11, 0), 55, true),
        ((12, 0), 36, false),
        ((12, 1), 36, false),
    ] {
        let symbols = expected_hardware_symbols(cc);
        assert_eq!(symbols.len(), count, "hardware artifact CC={cc:?}");
        assert_eq!(symbols.contains(WIDE), wide, "hardware artifact CC={cc:?}");
    }
}

#[derive(Clone, Copy)]
enum SpecializedTf32Family {
    Sm90a,
    Sm100,
    Sm120,
}

const RELEASE_TARGET_MATRIX: &[(&str, &str, SpecializedTf32Family)] = &[
    ("compute_90a", "sm_90a", SpecializedTf32Family::Sm90a),
    ("compute_100f", "sm_100f", SpecializedTf32Family::Sm100),
    ("compute_100a", "sm_100a", SpecializedTf32Family::Sm100),
    ("compute_103f", "sm_103f", SpecializedTf32Family::Sm100),
    ("compute_103a", "sm_103a", SpecializedTf32Family::Sm100),
    ("compute_107f", "sm_107f", SpecializedTf32Family::Sm100),
    ("compute_107a", "sm_107a", SpecializedTf32Family::Sm100),
    ("compute_110f", "sm_110f", SpecializedTf32Family::Sm100),
    ("compute_110a", "sm_110a", SpecializedTf32Family::Sm100),
    ("compute_120", "sm_120", SpecializedTf32Family::Sm120),
    ("compute_121", "sm_121", SpecializedTf32Family::Sm120),
];

fn specialized_family_contract(
    family: SpecializedTf32Family,
) -> (&'static str, &'static str, bool, BTreeSet<String>) {
    match family {
        SpecializedTf32Family::Sm90a => ("SM90a", SM90A_SOURCE, false, expected_sm90a_symbols()),
        SpecializedTf32Family::Sm100 => ("SM100", SM100_SOURCE, false, expected_sm100_symbols()),
        SpecializedTf32Family::Sm120 => ("SM120", SM120_SOURCE, true, expected_sm120_symbols()),
    }
}

/// Whether the loaded toolkit can name a release target: the CC 10.7 pair
/// arrived with CUDA 13.4, every other entry with 13.2 or earlier.
fn release_target_compiles(nvrtc_target: &str, nvrtc: (i32, i32)) -> bool {
    !nvrtc_target.contains("107") || nvrtc >= (13, 4)
}

fn release_entry_target_count(nvrtc: (i32, i32)) -> usize {
    RELEASE_TARGET_MATRIX
        .iter()
        .filter(|(nvrtc_target, _, _)| release_target_compiles(nvrtc_target, nvrtc))
        .map(|(_, _, family)| specialized_family_contract(*family).3.len())
        .sum()
}

fn contains_opcode_prefix(source: &str, prefix: &str) -> bool {
    source.match_indices(prefix).any(|(offset, _)| {
        source[..offset].chars().next_back().is_none_or(|previous| {
            !previous.is_ascii_alphanumeric() && !matches!(previous, '_' | '.')
        })
    })
}

fn opcode_context(source: &str, prefix: &str) -> Vec<String> {
    let lines = source.lines().collect::<Vec<_>>();
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| contains_opcode_prefix(line, prefix))
        .flat_map(|(index, _)| {
            let begin = index.saturating_sub(8);
            let end = (index + 9).min(lines.len());
            lines[begin..end].iter().map(|line| (*line).to_owned())
        })
        .collect()
}

fn contains_float_mad_opcode(source: &str) -> bool {
    source.lines().any(|line| {
        ptx_instruction(line).is_some_and(|(opcode, _)| {
            opcode.starts_with("mad.") && opcode.split('.').any(|part| part == "f32")
        })
    })
}

fn tf32_rna_finite_normal(bits: u32) -> u32 {
    let exponent = bits & 0x7f80_0000;
    assert!(exponent != 0 && exponent != 0x7f80_0000);
    bits.wrapping_add(0x1000) & !0x1fff
}

#[test]
fn exact_f32_policy_is_the_default_and_tf32_storage_is_its_only_door() {
    let declaration = braced_scope_after(CONTEXT_SOURCE, "pub enum F32TriadPolicy");
    assert_contains_all(
        declaration,
        &[
            "#[default]",
            "ExactScalarFma = 0",
            "AllowDeterministicTf32 = 1",
        ],
        "F32 triad policy",
    );
    assert!(
        source_mask(CONTEXT_SOURCE).contains("Clone, Copy, Debug, Default, PartialEq, Eq, Hash"),
        "F32TriadPolicy must remain a copyable, hashable defaulted identity field"
    );
    assert!(
        CONTEXT_SOURCE[..CONTEXT_SOURCE
            .find("pub enum F32TriadPolicy")
            .expect("F32TriadPolicy declaration")]
            .ends_with("#[repr(u8)]\n"),
        "F32TriadPolicy must use a stable u8 representation"
    );

    // The storage precision is the one public door to deterministic TF32:
    // the numeric contract is derived from the dtype, never parsed from
    // the environment.
    let numeric = source_mask(braced_scope_after(
        CONTEXT_SOURCE,
        "pub(crate) fn f32_numeric(self) -> F32TriadPolicy",
    ));
    assert_contains_all(
        &numeric,
        &[
            "Self::Tf32 => F32TriadPolicy::AllowDeterministicTf32",
            "Self::F32 | Self::Bf16 | Self::F16 => F32TriadPolicy::ExactScalarFma",
        ],
        "storage precision to f32 numeric contract",
    );
    // A withdrawn control is read through a string literal, so this scan
    // runs on the raw source, not on the mask that blanks literals.
    for withdrawn in [
        "MAMBA_RS_BI_F32_POLICY",
        "MAMBA_RS_BI_HALF_POLICY",
        "MAMBA_RS_BI_GEMM_FAMILY",
        "MAMBA_RS_BI_TENSOR_CORES",
        "MAMBA_RS_BATCH_INVARIANT",
        "MAMBA_RS_FAST_GEMM",
        "MAMBA_RS_ARCH_RUNG",
        "parse_env_value(value: &str) -> Result<Self, String>",
    ] {
        assert!(
            !CONTEXT_SOURCE.contains(withdrawn),
            "the context must not read the withdrawn control {withdrawn}"
        );
    }
    assert!(
        CONTEXT_SOURCE.contains("mode: std::env::var(\"MAMBA_RS_GEMM_MODE\")"),
        "one environment variable, resolved in one place is missing the mode read"
    );
    assert_contains_all(
        CONTEXT_SOURCE,
        &["fn resolve_gemm_env(values: GemmEnvValues, role: GemmRole)"],
        "one environment variable, resolved in one place",
    );
}

#[test]
fn deterministic_tf32_policy_is_separate_from_cublas_tf32_state() {
    assert_contains_all(
        CONTEXT_SOURCE,
        &[
            "cublas_tf32:",
            "f32_triad_policy:",
            "pub(crate) fn set_f32_triad_policy(&self, policy: F32TriadPolicy)",
            "pub(crate) fn f32_triad_policy(&self) -> F32TriadPolicy",
            "pub(crate) fn tf32(&self) -> bool",
        ],
        "cuBLAS/triad TF32 state separation",
    );
    assert!(
        !source_mask(CONTEXT_SOURCE).contains("\n    tf32: std::cell::Cell<bool>"),
        "the internal legacy field must be named cublas_tf32"
    );

    let set_mode = source_mask(braced_scope_after(
        CONTEXT_SOURCE,
        "pub fn set_gemm_mode(&self, mode: GemmMode)",
    ));
    assert_contains_all(
        &set_mode,
        &["change_math_mode", "mode.cublas_math()"],
        "GEMM mode owns the cuBLAS math mode",
    );
    assert!(
        !set_mode.contains("f32_triad_policy.set"),
        "the mode must not mutate the deterministic triad policy"
    );
    assert_contains_all(
        &source_mask(CONTEXT_SOURCE),
        &["cublasSetMathMode"],
        "cuBLAS math-mode backend",
    );

    let set_policy = source_mask(braced_scope_after(
        CONTEXT_SOURCE,
        "pub(crate) fn set_f32_triad_policy",
    ));
    assert_contains_all(
        &set_policy,
        &["graphs_captured", "f32_triad_policy.set(policy)"],
        "deterministic TF32 policy setter",
    );
    assert!(
        !set_policy.contains("cublasSetMathMode") && !set_policy.contains("cublas_tf32.set"),
        "deterministic triad policy must not mutate the cuBLAS handle"
    );
}

#[test]
fn context_and_resolved_route_identity_keep_tf32_domains_distinct() {
    let policy = braced_scope_after(IDENTITY_SOURCE, "pub struct GemmPolicy");
    assert_contains_all(
        policy,
        &["cublas_tf32: bool", "f32_triad_policy: F32TriadPolicy"],
        "context GEMM identity",
    );
    assert_contains_all(
        IDENTITY_SOURCE,
        &[
            "MmaTf32Rna",
            "Sm90aWgmmaTf32Tma",
            "Sm100Tcgen05Tf32Tma",
            "Sm120TmaMmaTf32Rna",
            "NUMERIC_CONTRACT_DOMAIN",
            "ARTIFACT_DIGEST_DOMAIN",
            "COMPILER_TARGET_DOMAIN",
            "DRIVER_BUILD_DIGEST_DOMAIN",
            "TUNING_TABLE_REVISION",
            "SCHEDULE_REVISION",
            "ResolvedInstructionFamily",
            "ResolvedInstructionShape",
            "ResolvedOperandConversion",
            "RegisterCvtRnaTf32F32",
            "TensorMapTfloat32",
            "TensorMapUint32ThenCvtRnaTf32F32",
        ],
        "resolved TF32 route identity",
    );
    for variant in [
        "MmaTf32Rna",
        "Sm90aWgmmaTf32Tma",
        "Sm100Tcgen05Tf32Tma",
        "Sm120TmaMmaTf32Rna",
    ] {
        assert!(
            IDENTITY_SOURCE.matches(variant).count() >= 2,
            "{variant} must have distinct physical-backend and numeric-contract identities"
        );
    }
    let route = braced_scope_after(IDENTITY_SOURCE, "pub struct ResolvedGemmRoute");
    assert_contains_all(
        route,
        &[
            "backend:",
            "numeric_contract:",
            "symbol:",
            "target:",
            "artifact:",
            "compiler:",
            "device:",
            "tile:",
            "bk:",
            "stages:",
            "tensor_map_revision:",
            "instruction_family:",
            "instruction_shape:",
            "operand_conversion:",
        ],
        "resolved GEMM route",
    );
}

#[test]
fn exact_policy_never_selects_tf32_and_allow_policy_falls_back_to_the_exact_family() {
    assert_contains_all(
        CONTRACT_SOURCE,
        &[
            "pub struct F32TriadRequest",
            "op: ResolvedGemmOp",
            "pub struct F32TriadShape",
            "m: usize",
            "k: usize",
            "n: usize",
            "lda: usize",
            "ldb: usize",
            "ldc: usize",
            "shape: F32TriadShape",
            "pub struct F32TriadOperands",
            "output: CUptr",
            "a: CUptr",
            "b: CUptr",
            "bias: Option<CUptr>",
            "alpha: f32",
            "beta: f32",
            "pub enum Tf32PhysicalRoute",
            "MmaTf32Rna(Tf32PortableRoute)",
            "Sm90aWgmmaTf32Tma(Tf32Sm90aRoute)",
            "Sm100Tcgen05Tf32Tma(Tf32Sm100Route)",
            "Sm120TmaMmaTf32Rna(Tf32Sm120Route)",
            "pub enum F32TriadSelection",
            "ScalarFma",
            "Tf32(Tf32PhysicalRoute)",
            "pub struct Tf32QualifiedModule",
            "module_kind: ModuleKind",
            "target: CudaTarget",
            "artifact: ArtifactIdentity",
            "compiler: CompilerIdentity",
            "device: DeviceIdentity",
            "device_caps: DeviceCaps",
            "sm120_fma_exclusions: Sm120FmaExclusions",
            "pub struct F32TriadAvailability",
            "portable: Option<Tf32QualifiedModule>",
            "specialized: Option<Tf32QualifiedModule>",
        ],
        "public TF32 selection contract",
    );

    let public_resolver = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_f32_triad_auto",
    ));
    assert_contains_all(
        &public_resolver,
        &[
            "policy: F32TriadPolicy",
            "request: F32TriadRequest",
            "availability: F32TriadAvailability",
            "Result<F32TriadSelection, String>",
            "resolve_f32_triad_auto_impl(policy, request, None, availability)",
        ],
        "public automatic TF32 resolver",
    );
    let operand_resolver = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub(super) fn resolve_f32_triad_auto_with_operands",
    ));
    assert_contains_all(
        &operand_resolver,
        &[
            "policy: F32TriadPolicy",
            "request: F32TriadRequest",
            "operands: F32TriadOperands",
            "availability: F32TriadAvailability",
            "Result<F32TriadSelection, String>",
            "resolve_f32_triad_auto_impl(policy, request, Some(operands), availability)",
        ],
        "operand-aware automatic TF32 resolver",
    );
    let resolver = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "fn resolve_f32_triad_auto_impl",
    ));
    assert_contains_all(
        &resolver,
        &[
            "policy: F32TriadPolicy",
            "request: F32TriadRequest",
            "availability: F32TriadAvailability",
            "Result<F32TriadSelection, String>",
            "F32TriadPolicy::ExactScalarFma",
            "F32TriadPolicy::AllowDeterministicTf32",
            "F32TriadSelection::ScalarFma",
            "request",
            "availability",
            "validate",
        ],
        "automatic TF32 resolver",
    );
    let exact = resolver
        .find("F32TriadPolicy::ExactScalarFma")
        .expect("exact policy branch");
    let allow = resolver
        .find("F32TriadPolicy::AllowDeterministicTf32")
        .expect("allow policy branch");
    let exact_branch = if exact < allow {
        &resolver[exact..allow]
    } else {
        &resolver[exact..]
    };
    assert!(
        exact_branch.contains("exact_or_scalar_selection")
            && !exact_branch.contains("F32TriadSelection::Tf32"),
        "exact policy must resolve through the exact-or-scalar selection"
    );
    // Both policies share one floor: the exact family when the operands admit
    // it, the scalar chain otherwise, and never a TF32 route.
    let floor = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "fn exact_or_scalar_selection",
    ));
    assert_contains_all(
        &floor,
        &[
            "sm120_fma_exact_route",
            "F32TriadSelection::ScalarFma",
            "F32TriadSelection::ExactSm120Fma",
        ],
        "exact-or-scalar selection floor",
    );
    assert!(
        !floor.contains("F32TriadSelection::Tf32"),
        "the exact-or-scalar floor must never select a TF32 route"
    );
    let sm120_exact = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "fn sm120_fma_exact_route",
    ));
    assert_eq!(
        sm120_exact.matches("sm120_fma_exclusions").count(),
        2,
        "SM120 exact selection must filter measured and generic routes independently"
    );
    let forced_scope = braced_scope_after(DISPATCH_SOURCE, "pub fn resolve_tf32_forced");
    let forced = source_mask(forced_scope);
    assert_contains_all(
        &forced,
        &["Sm120TmaFmaExact", "sm120_fma_exclusions"],
        "forced exact-F32 per-symbol rejection",
    );
    assert!(
        forced_scope.contains("forced exact-F32 route {symbol} is excluded on this toolkit"),
        "forced exact-F32 rejection must identify the unavailable symbol and toolkit"
    );
    let allow_branch = &resolver[allow..];
    assert!(
        allow_branch.contains("exact_or_scalar_selection"),
        "allow policy must fall back to the exact family, not to the plain scalar chain"
    );
    assert!(
        !resolver.contains("Instant::")
            && !resolver.contains("elapsed(")
            && !resolver.contains("CudaEvent"),
        "automatic TF32 dispatch must use frozen cells rather than runtime timing"
    );
    assert!(
        resolver.matches("return Ok(").count() < 2 || resolver.contains("match policy"),
        "automatic resolver must branch on policy instead of returning a constant selection"
    );
    assert!(
        resolver.contains("TF32_TUNING_TABLE_REVISION")
            || resolver.contains("F32_TF32_TUNING_REVISION"),
        "automatic resolver must consult a revisioned frozen table"
    );

    for (marker, fields) in [
        (
            "pub struct Tf32PortableRoute",
            &["tile: Tf32PortableTile", "stages: Tf32PortableStages"][..],
        ),
        (
            "pub struct Tf32Sm90aRoute",
            &["schedule: Sm90aWarpgroupSchedule"][..],
        ),
        (
            "pub struct Tf32Sm100Route",
            &[
                "tile: Sm100Tile",
                "stages: Sm100Stages",
                "schedule: Sm100Schedule",
            ][..],
        ),
        (
            "pub struct Tf32Sm120Route",
            &["tile: Tf32Sm120Tile", "stages: Tf32Sm120Stages"][..],
        ),
    ] {
        assert_contains_all(braced_scope_after(CONTRACT_SOURCE, marker), fields, marker);
    }
}

#[test]
fn staged_behavioral_resolver_scaffold_is_explicit_and_nontrivial() {
    assert_contains_all(
        CONTRACT_SOURCE,
        &[
            "pub struct F32TriadShape",
            "pub struct F32TriadRequest",
            "pub struct F32TriadAvailability",
            "pub enum Tf32PhysicalRoute",
            "pub enum F32TriadSelection",
        ],
        "behavioral resolver scaffold",
    );
    let public_automatic = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_f32_triad_auto",
    ));
    assert!(
        public_automatic
            .contains("resolve_f32_triad_auto_impl(policy, request, None, availability)",),
        "public automatic resolver must delegate to the shared validated implementation"
    );
    let operand_automatic = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub(super) fn resolve_f32_triad_auto_with_operands",
    ));
    assert!(
        operand_automatic.contains(
            "resolve_f32_triad_auto_impl(policy, request, Some(operands), availability)",
        ),
        "operand-aware automatic resolver must delegate with concrete operands"
    );
    let automatic = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "fn resolve_f32_triad_auto_impl",
    ));
    let forced = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_tf32_forced",
    ));
    for (name, scope) in [("automatic", &automatic), ("forced", &forced)] {
        assert!(
            scope.contains("request.") && scope.contains("availability."),
            "{name} resolver must consume request and qualified availability behaviorally"
        );
        assert!(
            scope.contains("validate") || scope.contains("ensure_"),
            "{name} resolver must validate rather than return unconditional Ok/Err"
        );
    }
    assert_contains_all(
        &forced,
        &[
            "Tf32PhysicalRoute::MmaTf32Rna",
            "Tf32PhysicalRoute::Sm90aWgmmaTf32Tma",
            "Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma",
            "Tf32PhysicalRoute::Sm120TmaMmaTf32Rna",
        ],
        "forced resolver physical-family admission",
    );
}

#[test]
fn graph_identity_rejects_policy_and_physical_route_drift_before_launch() {
    let route_snapshot = braced_scope_after(CONTEXT_SOURCE, "pub fn gemm_route(&self)");
    assert_contains_all(
        route_snapshot,
        &["self.gemm_policy()", "GemmRouteIdentity"],
        "context graph route snapshot",
    );
    let policy_snapshot = braced_scope_after(CONTEXT_SOURCE, "pub(crate) fn gemm_policy(&self)");
    assert_contains_all(
        policy_snapshot,
        &["cublas_tf32", "f32_triad_policy", "bi_gemm_family"],
        "context graph policy snapshot",
    );
    assert_contains_all(
        TRAINING_GRAPH_SOURCE,
        &["ensure_current(ctx.gemm_route()"],
        "training graph replay guard",
    );
    assert_contains_all(
        IDENTITY_SOURCE,
        &[
            "build_resolved_gemm_launch_set",
            "ROUTE_INDEX_DOMAIN",
            "LAUNCH_COUNT_DOMAIN",
        ],
        "ordered resolved-route graph guard",
    );
    assert!(
        source_mask(LAUNCH_SOURCE).contains("resolved_launch_set")
            && source_mask(LAUNCH_SOURCE).contains("ensure_current(live_launch_set"),
        "prepared specialized routes must compare their resolved launch identity on replay"
    );
}

#[test]
fn graph_plan_records_every_resolved_launch_and_replays_without_allocation() {
    assert_contains_all(
        IDENTITY_SOURCE,
        &[
            "pub(crate) struct CapturedGemmGraphPlan",
            "context: GemmRouteIdentity",
            "launches: ResolvedGemmLaunchSet",
            "routes: Box<[ResolvedGemmRoute]>",
            "struct ResolvedGemmLaunchSetBuilder",
            "fn push(",
            "fn finish(",
        ],
        "ordered graph launch plan",
    );
    assert_contains_all(
        CONTEXT_SOURCE,
        &[
            "begin_gemm_route_recording",
            "record_resolved_gemm_route",
            "GemmRouteRecordingGuard",
            "finish(self) -> Result<CapturedGemmGraphPlan, String>",
        ],
        "context-scoped graph route recorder",
    );
    assert_contains_all(
        TRAINING_GRAPH_SOURCE,
        &["CapturedGemmGraphPlan", "with_validated_gemm_graph_launch"],
        "f32 graph physical-route replay guard",
    );
}

#[test]
fn graph_capture_and_launch_use_the_single_allocation_free_plan_boundary() {
    let capture = source_mask(braced_scope_after(
        GRAPH_CAPTURE_SOURCE,
        "pub(crate) unsafe fn capture_into_graph_with_gemm_plan",
    ));
    assert_contains_all(
        &capture,
        &[
            "ctx: &GpuCtx",
            "route_capacity: usize",
            "manifest: &PreparedGemmCaptureManifest",
            "Result<(CudaGraph, Option<CapturedGemmGraphPlan>), String>",
            "manifest.validate_capture_request(ctx.gemm_route(), route_capacity)",
            "begin_gemm_route_recording(route_capacity)",
            "capture_into_graph(&ctx.stream",
            "recording.finish_against_manifest(manifest)",
        ],
        "common graph capture/route-plan wrapper",
    );
    for forbidden in ["recording.finish()", "recording.finish_eager_manifest()"] {
        assert!(
            !capture.contains(forbidden),
            "common graph capture may not use the old/raw {forbidden} boundary"
        );
    }
    let capture_body = &capture[capture.find('{').expect("capture body")..];
    let validation = capture_body
        .find("manifest.validate_capture_request")
        .expect("manifest validation");
    let recorder = capture_body
        .find("begin_gemm_route_recording")
        .expect("recorder");
    let capture_start = capture_body.find("capture_into_graph").expect("capture");
    assert!(
        validation < recorder && recorder < capture_start,
        "manifest validation and route-recorder allocation must precede cuStreamBeginCapture"
    );

    let launch = source_mask(method_scope_for_type(
        IDENTITY_SOURCE,
        "CapturedGemmGraphPlan",
        "with_validated_launch",
    ));
    assert_contains_all(
        &launch,
        &[
            "ctx: &GpuCtx",
            "label: &str",
            "FnOnce()",
            "ensure_current",
            "launch()",
        ],
        "validated graph launch closure boundary",
    );
    assert!(
        launch.find("ensure_current").expect("plan validation")
            < launch.rfind("launch()").expect("launch closure"),
        "physical plan validation must finish before invoking the launch closure"
    );
    for forbidden in [
        "Vec::", "vec![", ".collect", "Box::", "alloc(", "encode", "compile",
    ] {
        assert!(
            !launch.contains(forbidden),
            "validated graph launch closure may not perform {forbidden}"
        );
    }

    let identity_tests = active_test_module_scope(IDENTITY_SOURCE, "cache_and_header_tests")
        .unwrap_or_else(|error| panic!("{error}"));
    let mutation = direct_test_function_scope(
        identity_tests,
        "validated_launch_rejects_policy_mutation_before_closure",
        "#[test]#[ignore=]",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_contains_all(
        mutation,
        &[
            "Cell::new(0_u64)",
            "set_f32_triad_policy",
            "with_validated_launch",
            "attempts.set(attempts.get() + 1)",
            "assert!(result.is_err())",
            "assert_eq!(attempts.get(), 0)",
        ],
        "host launch-attempt counter proof",
    );
}

#[test]
fn physical_graph_capture_has_one_opaque_package_and_guarded_graph_boundary() {
    let capture = source_mask(braced_scope_after(
        GRAPH_CAPTURE_SOURCE,
        "unsafe fn capture_into_graph_with_physical_plan",
    ));
    assert_code_contains_all(
        &capture,
        &[
            "package: PreparedPhysicalGraphPackage",
            "Result<CapturedPhysicalGraph, String>",
            "package.bind_launches()",
            "finish_recording_physical_capture",
            "CapturedPhysicalGraph::new",
        ],
        "sealed physical graph capture transaction",
    );
    for forbidden in [
        "FnOnce",
        "body:",
        "Result<(CudaGraph",
        "Result<(cudarc::driver::CudaGraph",
    ] {
        assert!(
            !capture.contains(forbidden),
            "physical capture must not expose arbitrary callback or split graph/plan capability: {forbidden}"
        );
    }

    let holder = source_mask(braced_scope_after(
        GRAPH_CAPTURE_SOURCE,
        "pub(super) struct CapturedPhysicalGraph",
    ));
    assert_code_contains_all(
        &holder,
        &["graph: CudaGraph", "plan: CapturedPhysicalGraphPlan"],
        "owned physical graph holder",
    );
    let guarded_launch = compact_code(&source_mask(braced_scope_after(
        GRAPH_CAPTURE_SOURCE,
        "pub(super) fn launch(&self",
    )));
    assert_code_contains_all(
        &guarded_launch,
        &["self.plan.validate_replay", "self.graph.launch()"],
        "guarded physical graph replay",
    );
    assert!(
        !guarded_launch.contains("FnOnce"),
        "guarded physical graph replay may not accept an arbitrary launch closure"
    );

    let package = source_mask(braced_scope_after(
        BLAS_SOURCE,
        "pub(super) struct PreparedPhysicalGraphPackage",
    ));
    assert_code_contains_all(
        &package,
        &[
            "observer: Option<RecordingPhysicalObserver>",
            "prefix:",
            "triad:",
            "suffix:",
        ],
        "opaque preprepared physical enqueue package",
    );

    let enqueue = source_mask(braced_scope_after(
        BLAS_SOURCE,
        "pub(super) unsafe fn enqueue(",
    ));
    assert_code_contains_all(
        &enqueue,
        &["enqueue_prepared_physical_launch"],
        "fixed prepared physical enqueue sequence",
    );
    for forbidden in [
        "format!",
        "String",
        "Vec::",
        "try_reserve",
        "launch_builder",
        ".arg(",
        ".resolve(",
        "physical_digest",
        "allocation",
        "selector",
        "fallback",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "physical capture enqueue may not perform {forbidden}"
        );
    }

    let triad_enqueue = source_mask(method_scope_for_type(
        LAUNCH_SOURCE,
        "BoundTriadPhysicalGraphSequence",
        "enqueue",
    ));
    assert_code_contains_all(
        &triad_enqueue,
        &["enqueue_prepared_physical_launch"],
        "fixed prepared variable-triad enqueue sequence",
    );
    for forbidden in [
        "format!",
        "String",
        "Vec::",
        "try_reserve",
        "launch_builder",
        ".arg(",
        ".resolve(",
        "physical_digest",
        "allocation",
        "selector",
        "fallback",
    ] {
        assert!(
            !triad_enqueue.contains(forbidden),
            "variable-triad capture enqueue may not perform {forbidden}"
        );
    }

    let capture_transaction = source_mask(braced_scope_after(
        GRAPH_CAPTURE_SOURCE,
        "unsafe fn capture_prepared_physical_launches",
    ));
    let begin_tail = &capture_transaction[capture_transaction
        .find(".begin_capture(")
        .expect("physical begin_capture")..];
    let capture_active = &begin_tail[begin_tail
        .find("?;")
        .expect("successful begin_capture terminator")
        + 2
        ..begin_tail
            .find("let end_result = stream.end_capture")
            .expect("physical end_capture")];
    assert_code_contains_all(
        capture_active,
        &["launches.enqueue(observer)"],
        "capture-active prepared enqueue interval",
    );
    for forbidden in [
        "format!",
        "String",
        "Vec::",
        "try_reserve",
        "launch_builder",
        ".arg(",
        ".resolve(",
        "physical_digest",
        "allocation",
        "selector",
        "fallback",
        "compile",
        "descriptor",
        "hash",
    ] {
        assert!(
            !capture_active.contains(forbidden),
            "capture-active interval may not perform {forbidden}"
        );
    }

    let package_binding = source_mask(braced_scope_after(
        BLAS_SOURCE,
        "pub(super) fn bind_launches",
    ));
    assert_code_contains_all(
        &package_binding,
        &["bind_direct_launches", "triad.bind"],
        "sealed pre-capture segment binding",
    );
    let direct_binding = source_mask(braced_scope_after(BLAS_SOURCE, "fn bind_direct_launches"));
    assert_code_contains_all(
        &direct_binding,
        &["launch_builder", "builder.arg", "try_reserve_exact"],
        "pre-capture direct function and argument binding",
    );
    let scalar_binding = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) fn bind<'a>",
    ));
    assert_code_contains_all(
        &scalar_binding,
        &["launch_builder", "builder.arg", "try_reserve_exact"],
        "pre-capture variable scalar sequence binding",
    );
    let bind = capture
        .find("package.bind_launches()")
        .expect("package binding");
    let begin = capture
        .find("capture_prepared_physical_launches")
        .expect("physical capture transaction");
    assert!(
        bind < begin,
        "all LaunchArgs growth must finish before capture"
    );
}

#[test]
fn physical_graph_package_covers_all_prepared_triad_route_shapes() {
    assert!(
        LAUNCH_SOURCE.contains("const PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES: usize = 128;"),
        "pre-bound physical arguments must hold the CUDA 13 CUtensorMap ABI"
    );
    let request = source_mask(braced_scope_after(
        BLAS_SOURCE,
        "pub(in crate::mamba_ssm::gpu) struct F32PhysicalGraphPackageRequest",
    ));
    assert_code_contains_all(
        &request,
        &["prepared:", "output:", "a:", "b:", "capacity:"],
        "sealed F32 physical package request",
    );
    for forbidden in ["ResolvedPhysicalKernelLaunch", "LaunchArgs", "CudaFunction"] {
        assert!(
            !request.contains(forbidden),
            "F32 package request may not accept caller-built {forbidden}"
        );
    }

    let parts = source_mask(braced_scope_after(
        BLAS_SOURCE,
        "fn prepare_f32_physical_graph_parts",
    ));
    assert_code_contains_all(
        &parts,
        &[
            "physical_graph_is_direct",
            "prepare_prepared_f32_direct_graph_sequence",
            "ResolvedGemmOp::Nn",
            "prepare_prepared_f32_forward_graph_sequence",
            "ResolvedGemmOp::Tn",
            "prepare_prepared_f32_backward_dw_graph_sequence",
            "ResolvedGemmOp::Nt",
            "prepare_prepared_f32_backward_dx_graph_sequence",
        ],
        "operation-complete sealed F32 package producer",
    );
    let direct = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) fn prepare_prepared_f32_direct_graph_sequence",
    ));
    assert_code_contains_all(
        &direct,
        &[
            "PreparedF32Kind::ScalarZero",
            "PreparedF32Kind::Tf32",
            "PreparedF32Kind::Scalar(_)",
            "PhysicalLaunchObservation::gemm",
            "resolve_physical_launch_observation",
            "PhysicalScalarKernelArguments::new",
        ],
        "prepared ScalarZero and TF32 ABI producer",
    );

    for dispatcher in [
        "fn gemm_bi_forward_sub_with_control",
        "fn gemm_bi_backward_dw_with_control",
        "fn gemm_bi_backward_dx_with_control",
    ] {
        let scope = source_mask(braced_scope_after(LAUNCH_SOURCE, dispatcher));
        assert!(
            scope.contains("scalar_launch_builder"),
            "{dispatcher} must route every exact scalar ABI through the prepared builder"
        );
        assert!(
            !scope.contains("stream.launch_builder"),
            "{dispatcher} has a scalar launch that bypasses prepared ABI capture"
        );
    }

    let scalar_plan_fields =
        source_mask(braced_scope_after(LAUNCH_SOURCE, "fn scalar_plan_fields"));
    for variant in [
        "NnUltraThin",
        "NnNarrowSmall",
        "NnNarrow",
        "NnGemv",
        "NnSplitKThinTail",
        "NnSplitKThin",
        "NnSplitKSlim",
        "NnFinal",
        "TnGemv",
        "TnNarrow",
        "TnNarrowSplitM",
        "TnSplitM",
        "TnFinal",
        "NtNarrow",
        "NtSmallBatchWide",
        "NtGemv",
        "NtSplitKTail",
        "NtSplitKMain",
        "NtSplitKSlim",
        "NtMidBatchWide",
        "NtFinal",
    ] {
        assert!(
            scalar_plan_fields.contains(variant),
            "prepared scalar route census omitted {variant}"
        );
    }
}

#[test]
fn exact_graph_inventory_wires_decode_and_existing_prefill_training_holders() {
    let remaining_holders = [
        (
            "M1 f32 training",
            TRAINING_GRAPH_SOURCE,
            "GpuMambaF32TrainingStepGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M1 pooled prefill",
            PREFILL_SOURCE,
            "PrefillPooledGraph",
            "capture",
            &["launch"][..],
        ),
        (
            "M3 f32 training",
            MAMBA3_TRAINING_GRAPH_SOURCE,
            "GpuMamba3F32TrainingStepGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M3 full prefill",
            MAMBA3_PREFILL_SOURCE,
            "Mamba3PrefillGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M3 pooled prefill",
            MAMBA3_PREFILL_SOURCE,
            "Mamba3PrefillPooledGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M1 bf16 training",
            TRAINING_GRAPH_SOURCE,
            "GpuMambaTrainingStepGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M1 raw f16 trainer",
            TRAINER_SOURCE,
            "MambaTrainerMixed",
            "capture_graph_f16",
            &["step_f16"][..],
        ),
        (
            "M3 bf16 training",
            MAMBA3_TRAINING_GRAPH_SOURCE,
            "GpuMamba3TrainingStepGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M3 raw f16 trainer",
            MAMBA3_TRAINER_SOURCE,
            "Mamba3TrainerMixed",
            "capture_graph_f16",
            &["step_f16"][..],
        ),
    ];
    assert_eq!(remaining_holders.len(), 9, "remaining owner census");
    for (label, source, holder, capture, replays) in remaining_holders {
        let structure = source_mask(struct_scope_for_type(source, holder));
        assert_code_contains_all(
            &structure,
            &[
                "Option<CapturedGemmGraphPlan>",
                "GemmRoute",
                "has_gemm_work",
            ],
            &format!("{label} retained graph contract"),
        );
        let capture = source_mask(method_scope_for_type(source, holder, capture));
        assert_code_contains_all(
            &capture,
            &[
                "capture_into_graph_with_gemm_plan",
                "require_deterministic_gemm_graph_plan",
                "has_gemm_work",
            ],
            &format!("{label} scoped capture"),
        );
        assert!(
            !capture.contains("require_f32_triad_graph_plan"),
            "{label} must not retain the F32/Triad-only admission guard"
        );
        assert!(
            capture.find("capture_into_graph_with_gemm_plan").unwrap()
                < capture
                    .find("require_deterministic_gemm_graph_plan")
                    .unwrap(),
            "{label} must validate the completed capture's deterministic GEMM coverage"
        );
        for replay in replays {
            let replay = source_mask(method_scope_for_type(source, holder, replay));
            assert_code_contains_all(
                &replay,
                &["with_validated_gemm_graph_launch", "has_gemm_work"],
                &format!("{label}::{replay} shared replay guard"),
            );
            assert!(
                graph_launches_are_guarded(&replay),
                "{label}::{replay} may not launch outside the validated closure"
            );
        }
    }

    let recorder_owners = [
        (
            "M1 pooled prefill",
            PREFILL_SOURCE,
            "PrefillPooledGraph",
            "capture",
        ),
        (
            "M3 prefill",
            MAMBA3_PREFILL_SOURCE,
            "Mamba3Prefill",
            "run_full",
        ),
        (
            "M1 mixed f16 trainer",
            TRAINER_SOURCE,
            "MambaTrainerMixed",
            "eager_f16_forward_backward",
        ),
        (
            "M1 mixed bf16 trainer",
            TRAINER_SOURCE,
            "MambaTrainerMixed",
            "step_eager",
        ),
        (
            "M1 f32 trainer",
            TRAINER_SOURCE,
            "MambaTrainerF32",
            "step_eager",
        ),
        (
            "M3 mixed f16 trainer",
            MAMBA3_TRAINER_SOURCE,
            "Mamba3TrainerMixed",
            "eager_f16_forward_backward",
        ),
        (
            "M3 mixed bf16 trainer",
            MAMBA3_TRAINER_SOURCE,
            "Mamba3TrainerMixed",
            "step_eager",
        ),
        (
            "M3 f32 trainer",
            MAMBA3_TRAINER_SOURCE,
            "Mamba3TrainerF32",
            "step_eager",
        ),
    ];
    for (label, source, owner, method) in recorder_owners {
        let recorder = source_mask(method_scope_for_type(source, owner, method));
        assert_code_contains_all(
            &recorder,
            &[
                "has_gemm_work",
                "if has_gemm_work",
                "prepare_inference_arch_rung",
                "record_eager_gemm_manifest",
            ],
            &format!("{label} eager recorder preparation"),
        );
        assert!(
            recorder.find("if has_gemm_work").unwrap()
                < recorder.find("prepare_inference_arch_rung").unwrap()
                && recorder.find("prepare_inference_arch_rung").unwrap()
                    < recorder.find("record_eager_gemm_manifest").unwrap(),
            "{label} must prepare the selected Inference rung conditionally before recording"
        );
    }

    let decode_holders = [
        (
            INFERENCE_SOURCE,
            "GpuMambaInference",
            "capture_graph",
            &["step", "step_gpu_only"][..],
        ),
        (
            INFERENCE_SOURCE,
            "GpuMambaInferenceMixed",
            "capture_graph",
            &["step", "step_gpu_only"][..],
        ),
        (
            INFERENCE_SOURCE,
            "GpuMambaInferenceMixed",
            "capture_graph_mixed_native",
            &["step_mixed_native", "step_gpu_only_mixed_native"][..],
        ),
        (
            MAMBA3_INFERENCE_SOURCE,
            "Mamba3GpuInferenceEngine",
            "capture_graph",
            &["step", "step_gpu_only"][..],
        ),
        (
            MAMBA3_INFERENCE_SOURCE,
            "Mamba3GpuInferenceMixed",
            "capture_graph_mixed_native",
            &["step_mixed_native", "step_gpu_only_mixed_native"][..],
        ),
    ];
    for (source, holder, capture, replays) in decode_holders {
        let structure = source_mask(struct_scope_for_type(source, holder));
        assert_code_contains_all(
            &structure,
            &[
                "Option<CapturedGemmGraphPlan>",
                "Cell<Option<PreparedGemmCaptureManifest>>",
                "GemmRoute",
            ],
            holder,
        );
        let capture = source_mask(method_scope_for_type(source, holder, capture));
        assert_code_contains_all(
            &capture,
            &[
                ".take()",
                "capture_into_graph_with_gemm_plan",
                "require_deterministic_gemm_graph_plan",
                "has_gemm_work",
            ],
            holder,
        );
        assert!(!capture.contains("require_f32_triad_graph_plan"));
        assert!(
            capture
                .find("require_deterministic_gemm_graph_plan")
                .unwrap()
                < capture.find("self.graph = Some(graph)").unwrap(),
            "{holder} validates before installation"
        );
        for replay in replays {
            let replay = source_mask(method_scope_for_type(source, holder, replay));
            assert_code_contains_all(
                &replay,
                &[
                    ".set(None)",
                    "prepare_inference_arch_rung",
                    "record_eager_gemm_manifest",
                    "launch_captured_graph",
                ],
                holder,
            );
            assert!(
                replay.find(".set(None)").unwrap() < replay.find(".upload(").unwrap(),
                "{holder} clears permit before upload"
            );
            assert!(graph_launches_are_guarded(&replay));
        }
        let scope = method_scope_for_type(source, holder, "launch_captured_graph");
        assert_code_contains_all(
            scope,
            &[
                "with_validated_gemm_graph_launch",
                "has_gemm_work",
                "captured_gemm_plan",
            ],
            &format!("{holder} shared replay helper"),
        );
        assert!(
            graph_launches_are_guarded(scope),
            "{holder} replay helper may not launch outside the validated closure"
        );
    }
    for (capture, path) in [
        ("capture_graph", "Legacy"),
        ("capture_graph_mixed_native", "Native"),
    ] {
        let capture = method_scope_for_type(INFERENCE_SOURCE, "GpuMambaInferenceMixed", capture);
        assert_code_contains_all(
            capture,
            &[&format!(
                "self.captured_path = Some(MixedGraphPath::{path})"
            )],
            "M1 mixed graph association",
        );
    }
    let launch = method_scope_for_type(
        INFERENCE_SOURCE,
        "GpuMambaInferenceMixed",
        "launch_captured_graph",
    );
    assert_code_contains_all(
        launch,
        &["ensure_graph_path(path)"],
        "M1 mixed replay association",
    );
    let seam =
        active_production_function_scope(GRAPH_CAPTURE_SOURCE, "with_validated_gemm_graph_launch")
            .unwrap();
    assert_code_contains_all(
        seam,
        &[
            "ensure_gemm_usable",
            "require_deterministic_gemm_graph_plan",
            "plan.with_validated_launch",
            "None => launch()",
        ],
        "shared decode guard",
    );
}

#[test]
fn typed_fallback_records_scalar_routes_without_reading_f32_policy() {
    for (entry, shared, fallback) in [
        (
            "fn gemm_bi_forward_typed",
            "fn gemm_bi_forward_typed_in",
            "record_physical_exact_scalar_f32_forward",
        ),
        (
            "fn gemm_bi_backward_dw_typed",
            "fn gemm_bi_backward_dw_typed_in",
            "record_physical_exact_scalar_f32_backward_dw",
        ),
        (
            "fn gemm_bi_backward_dx_typed",
            "fn gemm_bi_backward_dx_typed_in",
            "record_physical_exact_scalar_f32_backward_dx",
        ),
    ] {
        // The dx recorder takes its arguments through one more hop; the
        // observer split lives in that hop's body.
        let split = if fallback == "record_physical_exact_scalar_f32_backward_dx" {
            "record_physical_exact_scalar_f32_backward_dx_with_arguments"
        } else {
            fallback
        };
        let entry_scope = source_mask(braced_scope_after(BLAS_SOURCE, entry));
        assert!(
            entry_scope.contains(shared.trim_start_matches("fn ")),
            "{entry} must delegate to its observer-generic branch body"
        );
        let shared_scope = source_mask(braced_scope_after(BLAS_SOURCE, shared));
        assert!(
            shared_scope.contains(fallback),
            "{shared} must route its scalar f32 fallback through physical observation"
        );
        assert!(
            !shared_scope.contains("f32_triad_policy")
                && !shared_scope.contains("AllowDeterministicTf32"),
            "{shared} typed numeric contract must ignore f32 TF32 policy"
        );
        let fallback_scope = source_mask(braced_scope_after(LAUNCH_SOURCE, split));
        assert_code_contains_all(
            &fallback_scope,
            &[
                "if O::ENABLED",
                "launch_cached_f32_triad_observed",
                "launch_cached_f32_triad",
            ],
            &format!("{fallback} compile-time observer split"),
        );
    }
    let physical_controller = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "impl<O: PhysicalLaunchObserver> ScalarLaunchController for PhysicalScalarLaunchControl",
    ));
    assert!(
        physical_controller.contains("enqueue_with_physical_observation")
            && physical_controller.contains("PhysicalLaunchObservation::gemm")
            && !physical_controller.contains("observe_then_enqueue")
            && !physical_controller.contains("builder.launch("),
        "the observed scalar controller must pass opaque semantic identity to the concrete real-enqueue primitive"
    );
}

fn assert_unique_live_delegation(source: &str, entry: &str, target: &str) -> Result<(), String> {
    let scope = unique_named_item_scope_at_depth(source, "fn", entry, 0)?;
    let mask = source_mask(scope);
    let call = format!("{target}(");
    if mask.matches(&call).count() != 1 {
        return Err(format!(
            "{entry} must contain exactly one live call to {target}"
        ));
    }
    if mask.matches(target).count() != 1 {
        return Err(format!(
            "{entry} contains a dead or duplicate {target} reference"
        ));
    }
    let call_offset = mask.find(&call).unwrap();
    let prefix = &mask[..call_offset];
    let mut semicolons = prefix.match_indices(';').map(|item| item.0).rev();
    let _last_statement = semicolons.next();
    let control_start = semicolons.next().map_or(0, |offset| offset + 1);
    let control_segment = &prefix[control_start..];
    if control_segment.contains("if false")
        || control_segment.contains("return;")
        || control_segment.contains("||")
    {
        return Err(format!(
            "{entry} delegates through dead or deferred control flow"
        ));
    }
    Ok(())
}

fn active_inlined_production_function_scope<'a>(
    source: &'a str,
    name: &str,
) -> Result<&'a str, String> {
    let function = unique_named_item_scope_at_depth(source, "fn", name, 0)?;
    let item_start = function.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if attributes != "#[inline(always)]" {
        return Err(format!(
            "production function {name} must have exactly one #[inline(always)] attribute"
        ));
    }
    Ok(function)
}

fn direct_function_body(source: &str, name: &str) -> Result<String, String> {
    let mask = source_mask(source);
    let open = function_body_open(&mask, 0, name)?;
    let close = matching_delimiter(&mask, open, b'{', b'}')
        .ok_or_else(|| format!("function {name} has an unterminated body"))?;
    Ok(compact_code(&mask[open + 1..close]))
}

fn validate_concrete_physical_enqueue_primitive(source: &str) -> Result<(), String> {
    let function =
        active_inlined_production_function_scope(source, "enqueue_with_physical_observation")?;
    let compact = compact_code(&source_mask(function));
    let signature = compact
        .split_once('{')
        .map(|(signature, _)| signature)
        .ok_or_else(|| "physical enqueue primitive has no body".to_string())?;
    let expected_signature = compact_code(
        "fn enqueue_with_physical_observation<O: PhysicalLaunchObserver>(observer: &mut O, builder: &mut cudarc::driver::LaunchArgs<'_>, config: cudarc::driver::LaunchConfig, observation: Option<PhysicalLaunchObservation>,) -> Result<(), PhysicalCudaLaunchError>",
    );
    if signature != expected_signature {
        return Err(format!(
            "physical enqueue primitive must accept only an observer, a concrete pre-bound cudarc builder, one config, and opaque semantic observation: actual={signature:?}, expected={expected_signature:?}"
        ));
    }
    if !compact_code(&source_mask(source))
        .contains(&format!("pub(super)unsafe{expected_signature}{{"))
    {
        return Err("physical enqueue primitive must stay GPU-private and unsafe".into());
    }
    if function.contains("FnOnce")
        || function.contains("FnMut")
        || function.contains("Fn(")
        || signature.contains("ResolvedPhysicalKernelLaunch")
    {
        return Err("physical enqueue primitive regained a callback or caller-built node".into());
    }
    let body = direct_function_body(function, "enqueue_with_physical_observation")?;
    let expected_body = compact_code(&source_mask(
        r#"
        let mut authority = physical_observer_private::Authority::new();
        if O::ENABLED {
            let record = observation
                .ok_or_else(|| "recording CUDA launch has no physical observation".to_string())
                .and_then(|observation| observation.resolve(observer, config))
                .and_then(|launch| {
                    physical_observer_private::Sealed::record_before_enqueue(
                        observer,
                        &mut authority,
                        launch,
                    )
                });
            if let Err(error) = record {
                physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
                return Err(PhysicalCudaLaunchError::Identity(error));
            }
        }
        match unsafe { builder.launch(config) } {
            Ok(_) => Ok(()),
            Err(error) => {
                if O::ENABLED {
                    physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
                }
                Err(PhysicalCudaLaunchError::Driver(error))
            }
        }
        "#,
    ));
    if body != expected_body {
        return Err(
            "physical enqueue primitive must resolve and privately record once before its sole same-config real cudarc launch, with sticky failure invalidation"
                .into(),
        );
    }
    Ok(())
}

fn validate_prepared_physical_enqueue_primitive(source: &str) -> Result<(), String> {
    let function =
        active_inlined_production_function_scope(source, "enqueue_prepared_physical_launch")?;
    let compact = compact_code(&source_mask(function));
    let signature = compact
        .split_once('{')
        .map(|(signature, _)| signature)
        .ok_or_else(|| "prepared physical enqueue primitive has no body".to_string())?;
    let expected_signature = compact_code(
        "fn enqueue_prepared_physical_launch(observer: &mut RecordingPhysicalObserver, builder: &mut cudarc::driver::LaunchArgs<'_>, config: cudarc::driver::LaunchConfig, launch: ResolvedPhysicalKernelLaunch,) -> Result<(), PhysicalCudaLaunchError>",
    );
    if signature != expected_signature {
        return Err(format!(
            "prepared physical enqueue primitive must accept only the recorder, concrete pre-bound cudarc builder, one config, and one resolved node: actual={signature:?}, expected={expected_signature:?}"
        ));
    }
    if !compact_code(&source_mask(source))
        .contains(&format!("pub(super)unsafe{expected_signature}{{"))
    {
        return Err("prepared physical enqueue primitive must stay GPU-private and unsafe".into());
    }
    if function.contains("FnOnce") || function.contains("FnMut") || function.contains("Fn(") {
        return Err("prepared physical enqueue primitive regained a callback".into());
    }
    let body = direct_function_body(function, "enqueue_prepared_physical_launch")?;
    let expected_body = compact_code(&source_mask(
        r#"
        let mut authority = physical_observer_private::Authority::new();
        if let Err(error) =
            physical_observer_private::Sealed::record_before_enqueue(observer, &mut authority, launch)
        {
            physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
            return Err(PhysicalCudaLaunchError::Identity(error));
        }
        match unsafe { builder.launch(config) } {
            Ok(_) => Ok(()),
            Err(error) => {
                physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
                Err(PhysicalCudaLaunchError::Driver(error))
            }
        }
        "#,
    ));
    if body != expected_body {
        return Err(
            "prepared physical enqueue primitive must privately record exactly once before its sole same-config concrete cudarc launch and make Driver failure sticky"
                .into(),
        );
    }
    if matching_call_ranges(function, "record_before_enqueue")?.len() != 1
        || matching_call_ranges(function, "builder.launch")?.len() != 1
    {
        return Err(
            "prepared physical enqueue topology must contain one recorder operation and one concrete Driver launch"
                .into(),
        );
    }
    Ok(())
}

fn matching_call_ranges(source: &str, callee: &str) -> Result<Vec<(usize, usize)>, String> {
    let mask = source_mask(source);
    let mut ranges = Vec::new();
    for offset in token_offsets(&mask, callee) {
        let open = skip_ascii_whitespace(&mask, offset + callee.len());
        if mask.as_bytes().get(open) != Some(&b'(') {
            continue;
        }
        let close = matching_delimiter(&mask, open, b'(', b')')
            .ok_or_else(|| format!("unterminated {callee} call"))?;
        ranges.push((offset, close + 1));
    }
    Ok(ranges)
}

fn top_level_arguments(call: &str) -> Result<Vec<&str>, String> {
    let mask = source_mask(call);
    let open = mask
        .find('(')
        .ok_or_else(|| "call is missing its opening parenthesis".to_string())?;
    let close = matching_delimiter(&mask, open, b'(', b')')
        .ok_or_else(|| "call is missing its closing parenthesis".to_string())?;
    let mut parens = 0_u32;
    let mut brackets = 0_u32;
    let mut braces = 0_u32;
    let mut argument_start = open + 1;
    let mut arguments = Vec::new();
    for (relative, byte) in mask.as_bytes()[open + 1..close].iter().copied().enumerate() {
        let offset = open + 1 + relative;
        match byte {
            b'(' => parens += 1,
            b')' => parens = parens.checked_sub(1).ok_or("unbalanced call parentheses")?,
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1).ok_or("unbalanced call brackets")?,
            b'{' => braces += 1,
            b'}' => braces = braces.checked_sub(1).ok_or("unbalanced call braces")?,
            b',' if parens == 0 && brackets == 0 && braces == 0 => {
                let argument = call[argument_start..offset].trim();
                if !argument.is_empty() {
                    arguments.push(argument);
                }
                argument_start = offset + 1;
            }
            _ => {}
        }
    }
    let trailing = call[argument_start..close].trim();
    if !trailing.is_empty() {
        arguments.push(trailing);
    }
    Ok(arguments)
}

fn assert_no_raw_cuda_launch(scope: &str, owner: &str) -> Result<(), String> {
    if !matching_call_ranges(scope, "launch")?.is_empty() {
        return Err(format!("{owner} contains a raw cudarc launch bypass"));
    }
    Ok(())
}

fn assert_calls_with_config(
    scope: &str,
    owner: &str,
    callee: &str,
    expected_count: usize,
    config_index: usize,
    expected_config: &str,
) -> Result<(), String> {
    let calls = matching_call_ranges(scope, callee)?;
    if calls.len() != expected_count {
        return Err(format!(
            "{owner} must call {callee} exactly {expected_count} times, found {}",
            calls.len()
        ));
    }
    for (start, end) in calls {
        let arguments = top_level_arguments(&scope[start..end])?;
        let config = arguments
            .get(config_index)
            .ok_or_else(|| format!("{owner} {callee} call omits its config argument"))?;
        if compact_code(&source_mask(config)) != compact_code(expected_config) {
            return Err(format!(
                "{owner} passes the wrong config to {callee}: {}",
                compact_code(&source_mask(config))
            ));
        }
    }
    Ok(())
}

fn collect_rust_sources(directory: &Path, output: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(directory).expect("read Rust source directory") {
        let path = entry.expect("read Rust source entry").path();
        if path.is_dir() {
            collect_rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = std::fs::read_to_string(&path).expect("read Rust source file");
            output.push((path, source));
        }
    }
}

fn direct_function_names_at_depth(source: &str, expected_depth: u32) -> Vec<String> {
    let mask = source_mask(source);
    let mut names = Vec::new();
    let mut braces = 0_u32;
    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut cursor = 0;
    while cursor < mask.len() {
        if braces == expected_depth
            && parentheses == 0
            && brackets == 0
            && token_at(&mask, cursor, "fn")
        {
            let start = skip_ascii_whitespace(&mask, cursor + "fn".len());
            let end = identifier_end(&mask, start);
            names.push(mask[start..end].to_string());
        }
        match mask.as_bytes()[cursor] {
            b'{' => braces += 1,
            b'}' => {
                braces = braces
                    .checked_sub(1)
                    .expect("balanced function-name braces")
            }
            b'(' => parentheses += 1,
            b')' => {
                parentheses = parentheses
                    .checked_sub(1)
                    .expect("balanced function-name parentheses")
            }
            b'[' => brackets += 1,
            b']' => {
                brackets = brackets
                    .checked_sub(1)
                    .expect("balanced function-name brackets")
            }
            _ => {}
        }
        cursor += 1;
    }
    names
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PhysicalSensitiveCall {
    GemmObservation,
    InferenceObservation,
    InputTransformObservation,
    ConversionObservation,
    ConversionArguments,
    ObserverConstructor,
    ObserverConstructionAuthority,
    PreparedObserverFactory,
    ObserverFinalizer,
    ObserverCaptureFinalizer,
    ObserverFinalizationAuthority,
    ObserverCaptureFinalizationAuthority,
    ObservationResolver,
    RecorderMutation,
    RecorderInvalidation,
    RecorderFinalizer,
    SealedRecorderMutation,
    SealedRecorderInvalidation,
    Submission,
    PreparedSubmission,
    ObservationResolution,
}

#[derive(Clone, Debug)]
struct RustCodeToken {
    text: String,
    start: usize,
}

fn rust_code_tokens(source: &str) -> Vec<RustCodeToken> {
    let mask = source_mask(source);
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < mask.len() {
        let byte = mask.as_bytes()[cursor];
        if byte.is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        if is_identifier_byte(byte) {
            let end = identifier_end(&mask, cursor);
            tokens.push(RustCodeToken {
                text: mask[cursor..end].to_owned(),
                start: cursor,
            });
            cursor = end;
        } else {
            tokens.push(RustCodeToken {
                text: char::from(byte).to_string(),
                start: cursor,
            });
            cursor += 1;
        }
    }
    tokens
}

fn physical_name_aliases(tokens: &[RustCodeToken]) -> BTreeMap<String, String> {
    fn resolved(aliases: &BTreeMap<String, String>, name: &str) -> String {
        let mut current = name.to_owned();
        for _ in 0..=aliases.len() {
            let Some(next) = aliases.get(&current) else {
                break;
            };
            if next == &current {
                break;
            }
            current = next.clone();
        }
        current
    }

    let mut aliases = BTreeMap::new();
    for _ in 0..=tokens.len() {
        let before = aliases.len();
        for (index, token) in tokens.iter().enumerate() {
            if token.text == "type" {
                let Some(alias) = tokens.get(index + 1).map(|token| token.text.clone()) else {
                    continue;
                };
                let Some(equal) = tokens[index + 2..]
                    .iter()
                    .position(|token| token.text == "=")
                    .map(|offset| index + 2 + offset)
                else {
                    continue;
                };
                let end = tokens[equal + 1..]
                    .iter()
                    .position(|token| token.text == ";")
                    .map(|offset| equal + 1 + offset)
                    .unwrap_or(tokens.len());
                if let Some(target) = tokens[equal + 1..end].iter().rev().find(|candidate| {
                    candidate
                        .text
                        .as_bytes()
                        .first()
                        .is_some_and(u8::is_ascii_alphabetic)
                }) {
                    aliases.insert(alias, resolved(&aliases, &target.text));
                }
            }
            if token.text != "use" {
                continue;
            }
            let end = tokens[index + 1..]
                .iter()
                .position(|candidate| candidate.text == ";")
                .map(|offset| index + 1 + offset)
                .unwrap_or(tokens.len());
            let Some(as_index) = tokens[index + 1..end]
                .iter()
                .rposition(|candidate| candidate.text == "as")
                .map(|offset| index + 1 + offset)
            else {
                continue;
            };
            let Some(alias) = tokens.get(as_index + 1).map(|token| token.text.clone()) else {
                continue;
            };
            let identifiers = tokens[index + 1..as_index]
                .iter()
                .filter(|candidate| {
                    candidate
                        .text
                        .as_bytes()
                        .first()
                        .is_some_and(u8::is_ascii_alphabetic)
                })
                .map(|candidate| candidate.text.as_str())
                .collect::<Vec<_>>();
            let Some(target) = identifiers.last() else {
                continue;
            };
            let target = resolved(&aliases, target);
            let canonical = identifiers
                .iter()
                .rev()
                .nth(1)
                .map(|owner| resolved(&aliases, owner))
                .filter(|owner| {
                    matches!(
                        owner.as_str(),
                        "PhysicalLaunchObservation"
                            | "PhysicalConversionArguments"
                            | "RecordingPhysicalObserver"
                            | "PhysicalTraceRecorder"
                            | "Sealed"
                    )
                })
                .map_or(target.clone(), |owner| format!("{owner}::{target}"));
            aliases.insert(alias, canonical);
        }
        if aliases.len() == before {
            break;
        }
    }
    aliases
}

fn resolved_physical_name(aliases: &BTreeMap<String, String>, name: &str) -> String {
    let mut current = name.to_owned();
    for _ in 0..=aliases.len() {
        let Some(next) = aliases.get(&current) else {
            break;
        };
        if next == &current {
            break;
        }
        current = next.clone();
    }
    current
}

fn token_path_owner(
    tokens: &[RustCodeToken],
    index: usize,
    aliases: &BTreeMap<String, String>,
) -> Option<String> {
    if index < 3 || tokens[index - 2].text != ":" || tokens[index - 1].text != ":" {
        return None;
    }
    tokens[..index - 2]
        .iter()
        .rev()
        .find(|candidate| {
            candidate
                .text
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic)
        })
        .map(|owner| resolved_physical_name(aliases, &owner.text))
}

fn token_is_path_member(
    tokens: &[RustCodeToken],
    index: usize,
    aliases: &BTreeMap<String, String>,
    owner: &str,
) -> bool {
    token_path_owner(tokens, index, aliases).as_deref() == Some(owner)
}

fn token_starts_call(tokens: &[RustCodeToken], index: usize) -> bool {
    if tokens.get(index + 1).is_some_and(|token| token.text == "(") {
        return true;
    }
    if tokens.get(index + 1).is_some_and(|token| token.text == ":")
        && tokens.get(index + 2).is_some_and(|token| token.text == ":")
        && tokens.get(index + 3).is_some_and(|token| token.text == "<")
    {
        let mut depth = 0_u32;
        for token in &tokens[index + 3..] {
            match token.text.as_str() {
                "<" => depth += 1,
                ">" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        let close = token.start;
                        return tokens
                            .iter()
                            .any(|candidate| candidate.start > close && candidate.text == "(");
                    }
                }
                _ => {}
            }
        }
    }
    false
}

fn physical_typed_receivers(
    tokens: &[RustCodeToken],
    aliases: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let sensitive_types = [
        "PhysicalLaunchObservation",
        "PhysicalTraceRecorder",
        "RecordingPhysicalObserver",
    ];
    let mut receivers = BTreeMap::new();
    let mut parentheses = 0_u32;
    for (index, token) in tokens.iter().enumerate() {
        match token.text.as_str() {
            "(" => parentheses += 1,
            ")" => parentheses = parentheses.saturating_sub(1),
            _ => {}
        }
        if parentheses == 0
            || !token
                .text
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic)
            || !tokens.get(index + 1).is_some_and(|token| token.text == ":")
            || tokens.get(index + 2).is_some_and(|token| token.text == ":")
        {
            continue;
        }
        let end = tokens[index + 2..]
            .iter()
            .position(|candidate| matches!(candidate.text.as_str(), "," | ")"))
            .map(|offset| index + 2 + offset)
            .unwrap_or(tokens.len());
        let receiver_type = tokens[index + 2..end]
            .iter()
            .map(|candidate| resolved_physical_name(aliases, &candidate.text))
            .find(|candidate| sensitive_types.contains(&candidate.as_str()));
        if let Some(receiver_type) = receiver_type {
            receivers.insert(token.text.clone(), receiver_type);
        }
    }
    for _ in 0..=tokens.len() {
        let before = receivers.len();
        for (index, token) in tokens.iter().enumerate() {
            if token.text != "let" {
                continue;
            }
            let mut binding = index + 1;
            if tokens
                .get(binding)
                .is_some_and(|candidate| candidate.text == "mut")
            {
                binding += 1;
            }
            let Some(binding_name) = tokens.get(binding).map(|token| token.text.clone()) else {
                continue;
            };
            let Some(equal) = tokens[binding + 1..]
                .iter()
                .position(|candidate| candidate.text == "=")
                .map(|offset| binding + 1 + offset)
            else {
                continue;
            };
            let end = tokens[equal + 1..]
                .iter()
                .position(|candidate| candidate.text == ";")
                .map(|offset| equal + 1 + offset)
                .unwrap_or(tokens.len());
            let rhs = &tokens[equal + 1..end];
            let direct_type = rhs
                .iter()
                .map(|candidate| resolved_physical_name(aliases, &candidate.text))
                .find(|candidate| sensitive_types.contains(&candidate.as_str()));
            let direct_receiver = rhs
                .iter()
                .find(|candidate| {
                    candidate
                        .text
                        .as_bytes()
                        .first()
                        .is_some_and(u8::is_ascii_alphabetic)
                })
                .and_then(|candidate| receivers.get(&candidate.text))
                .filter(|_| {
                    rhs.iter()
                        .filter(|candidate| {
                            candidate
                                .text
                                .as_bytes()
                                .first()
                                .is_some_and(u8::is_ascii_alphabetic)
                        })
                        .count()
                        == 1
                })
                .cloned();
            if let Some(receiver_type) = direct_type.or(direct_receiver) {
                receivers.insert(binding_name, receiver_type);
            }
        }
        if receivers.len() == before {
            break;
        }
    }
    receivers
}

fn token_dot_receiver_type(
    tokens: &[RustCodeToken],
    index: usize,
    receivers: &BTreeMap<String, String>,
) -> Option<String> {
    if index < 2 || tokens[index - 1].text != "." {
        return None;
    }
    receivers.get(&tokens[index - 2].text).cloned()
}

fn physical_sensitive_calls(
    source: &str,
    identity_owner: bool,
) -> Vec<(usize, PhysicalSensitiveCall)> {
    let tokens = rust_code_tokens(source);
    let aliases = physical_name_aliases(&tokens);
    let receivers = physical_typed_receivers(&tokens, &aliases);
    let mut calls = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 && tokens[index - 1].text == "fn" {
            continue;
        }
        let resolved = resolved_physical_name(&aliases, &token.text);
        let kind = match resolved.as_str() {
            "PhysicalLaunchObservation::gemm" => Some(PhysicalSensitiveCall::GemmObservation),
            "gemm"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalLaunchObservation") =>
            {
                Some(PhysicalSensitiveCall::GemmObservation)
            }
            "PhysicalLaunchObservation::conversion" => {
                Some(PhysicalSensitiveCall::ConversionObservation)
            }
            "PhysicalLaunchObservation::inference" => {
                Some(PhysicalSensitiveCall::InferenceObservation)
            }
            "inference"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalLaunchObservation") =>
            {
                Some(PhysicalSensitiveCall::InferenceObservation)
            }
            "PhysicalLaunchObservation::input_transform" => {
                Some(PhysicalSensitiveCall::InputTransformObservation)
            }
            "input_transform"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalLaunchObservation") =>
            {
                Some(PhysicalSensitiveCall::InputTransformObservation)
            }
            "conversion"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalLaunchObservation") =>
            {
                Some(PhysicalSensitiveCall::ConversionObservation)
            }
            "PhysicalConversionArguments::new" => Some(PhysicalSensitiveCall::ConversionArguments),
            "new"
                if token_is_path_member(
                    &tokens,
                    index,
                    &aliases,
                    "PhysicalConversionArguments",
                ) =>
            {
                Some(PhysicalSensitiveCall::ConversionArguments)
            }
            "RecordingPhysicalObserver::with_argument_identity" => {
                Some(PhysicalSensitiveCall::ObserverConstructor)
            }
            "with_argument_identity"
                if token_is_path_member(&tokens, index, &aliases, "RecordingPhysicalObserver") =>
            {
                Some(PhysicalSensitiveCall::ObserverConstructor)
            }
            "prepare_recording_physical_observer" => {
                Some(PhysicalSensitiveCall::ObserverConstructionAuthority)
            }
            "prepare_physical_observer" => Some(PhysicalSensitiveCall::PreparedObserverFactory),
            "RecordingPhysicalObserver::finish" => Some(PhysicalSensitiveCall::ObserverFinalizer),
            "finish"
                if token_is_path_member(&tokens, index, &aliases, "RecordingPhysicalObserver")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("RecordingPhysicalObserver")
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::ObserverFinalizer)
            }
            "RecordingPhysicalObserver::finish_capture" => {
                Some(PhysicalSensitiveCall::ObserverCaptureFinalizer)
            }
            "finish_capture"
                if token_is_path_member(&tokens, index, &aliases, "RecordingPhysicalObserver")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("RecordingPhysicalObserver")
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::ObserverCaptureFinalizer)
            }
            "finish_recording_physical_observer" => {
                Some(PhysicalSensitiveCall::ObserverFinalizationAuthority)
            }
            "finish_recording_physical_capture" => {
                Some(PhysicalSensitiveCall::ObserverCaptureFinalizationAuthority)
            }
            "PhysicalLaunchObservation::resolve" => {
                Some(PhysicalSensitiveCall::ObservationResolver)
            }
            "resolve"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalLaunchObservation")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("PhysicalLaunchObservation")
                        && token_starts_call(&tokens, index))
                    || (identity_owner
                        && index >= 1
                        && tokens[index - 1].text == "."
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::ObservationResolver)
            }
            "PhysicalTraceRecorder::record" => Some(PhysicalSensitiveCall::RecorderMutation),
            "record"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalTraceRecorder")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("PhysicalTraceRecorder")
                        && token_starts_call(&tokens, index))
                    || (identity_owner
                        && index >= 1
                        && tokens[index - 1].text == "."
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::RecorderMutation)
            }
            "PhysicalTraceRecorder::invalidate_enqueue" => {
                Some(PhysicalSensitiveCall::RecorderInvalidation)
            }
            "invalidate_enqueue"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalTraceRecorder")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("PhysicalTraceRecorder")
                        && token_starts_call(&tokens, index))
                    || (identity_owner
                        && index >= 1
                        && tokens[index - 1].text == "."
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::RecorderInvalidation)
            }
            "PhysicalTraceRecorder::finish" => Some(PhysicalSensitiveCall::RecorderFinalizer),
            "finish"
                if token_is_path_member(&tokens, index, &aliases, "PhysicalTraceRecorder")
                    || (token_dot_receiver_type(&tokens, index, &receivers).as_deref()
                        == Some("PhysicalTraceRecorder")
                        && token_starts_call(&tokens, index)) =>
            {
                Some(PhysicalSensitiveCall::RecorderFinalizer)
            }
            "Sealed::record_before_enqueue" => Some(PhysicalSensitiveCall::SealedRecorderMutation),
            "record_before_enqueue" if token_is_path_member(&tokens, index, &aliases, "Sealed") => {
                Some(PhysicalSensitiveCall::SealedRecorderMutation)
            }
            "Sealed::invalidate_enqueue" => Some(PhysicalSensitiveCall::SealedRecorderInvalidation),
            "invalidate_enqueue" if token_is_path_member(&tokens, index, &aliases, "Sealed") => {
                Some(PhysicalSensitiveCall::SealedRecorderInvalidation)
            }
            "enqueue_with_physical_observation" => Some(PhysicalSensitiveCall::Submission),
            "enqueue_prepared_physical_launch" => Some(PhysicalSensitiveCall::PreparedSubmission),
            "resolve_physical_launch_observation" => {
                Some(PhysicalSensitiveCall::ObservationResolution)
            }
            _ => None,
        };
        if let Some(kind) = kind {
            calls.push((token.start, kind));
        }
    }
    calls
}

fn validate_owner_sensitive_function_topology(identity_source: &str) -> Result<(), String> {
    let tokens = rust_code_tokens(identity_source);
    let aliases = physical_name_aliases(&tokens);
    let mut sensitive_types = [
        "CapturedPhysicalGraphPlan",
        "PhysicalConversionArguments",
        "PhysicalLaunchObservation",
        "PhysicalTraceRecorder",
        "RecordedPhysicalTrace",
        "RecordingPhysicalObserver",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    loop {
        let before = sensitive_types.len();
        for (index, token) in tokens.iter().enumerate() {
            if token.text != "type" {
                continue;
            }
            let Some(alias) = tokens.get(index + 1).map(|token| token.text.clone()) else {
                continue;
            };
            let end = tokens[index + 2..]
                .iter()
                .position(|candidate| candidate.text == ";")
                .map(|offset| index + 2 + offset)
                .unwrap_or(tokens.len());
            let Some(equal) = tokens[index + 2..end]
                .iter()
                .position(|candidate| candidate.text == "=")
                .map(|offset| index + 2 + offset)
            else {
                continue;
            };
            if tokens[equal + 1..end].iter().any(|candidate| {
                sensitive_types.contains(&candidate.text)
                    || sensitive_types.contains(&resolved_physical_name(&aliases, &candidate.text))
            }) {
                sensitive_types.insert(alias);
            }
        }
        if sensitive_types.len() == before {
            break;
        }
    }

    let allowed = [
        "enqueue_prepared_physical_launch",
        "enqueue_with_physical_observation",
        "finish_recording_physical_capture",
        "finish_recording_physical_observer",
        "prepare_recording_physical_observer",
        "recorded_physical_trace_for_test",
        "resolve_physical_launch_observation",
    ];
    let mut braces = 0_u32;
    for (index, token) in tokens.iter().enumerate() {
        if matches!(token.text.as_str(), "const" | "static") {
            let start = tokens[..index]
                .iter()
                .rposition(|candidate| matches!(candidate.text.as_str(), ";" | "{" | "}"))
                .map_or(0, |offset| offset + 1);
            let prefix = &tokens[start..index];
            let item_declaration = !tokens
                .get(index.wrapping_sub(1))
                .is_some_and(|candidate| candidate.text == "'")
                && !prefix.iter().any(|candidate| {
                    matches!(
                        candidate.text.as_str(),
                        "enum" | "fn" | "impl" | "mod" | "struct" | "trait" | "type"
                    )
                });
            let visible =
                item_declaration && prefix.iter().any(|candidate| candidate.text == "pub");
            let equal = tokens[index + 1..]
                .iter()
                .position(|candidate| candidate.text == "=")
                .map(|offset| index + 1 + offset)
                .unwrap_or(tokens.len());
            let sensitive_signature = tokens[index + 1..equal].iter().any(|candidate| {
                sensitive_types.contains(&candidate.text)
                    || sensitive_types.contains(&resolved_physical_name(&aliases, &candidate.text))
            });
            if visible && sensitive_signature {
                return Err(format!(
                    "owner visible {} exposes a private physical authority type",
                    token.text
                ));
            }
        }
        if braces == 0 && token.text == "fn" {
            let Some(name) = tokens
                .get(index + 1)
                .map(|candidate| candidate.text.as_str())
            else {
                continue;
            };
            if !name.as_bytes().first().is_some_and(u8::is_ascii_alphabetic) {
                continue;
            }
            let end = tokens[index + 2..]
                .iter()
                .position(|candidate| matches!(candidate.text.as_str(), "{" | ";"))
                .map(|offset| index + 2 + offset)
                .unwrap_or(tokens.len());
            let sensitive_signature = tokens[index + 2..end].iter().any(|candidate| {
                sensitive_types.contains(&candidate.text)
                    || sensitive_types.contains(&resolved_physical_name(&aliases, &candidate.text))
            });
            if sensitive_signature && !allowed.contains(&name) {
                return Err(format!(
                    "owner function {name} consumes or returns a private physical authority type"
                ));
            }
        }
        match token.text.as_str() {
            "{" => braces += 1,
            "}" => braces = braces.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

fn audited_function_scope<'a>(
    source: &'a str,
    name: &str,
    expected_attributes: &str,
) -> Result<&'a str, String> {
    let function = unique_named_item_scope_at_depth(source, "fn", name, 0)?;
    let item_start = function.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if attributes != expected_attributes {
        return Err(format!(
            "audited function {name} has attributes {attributes:?}, expected {expected_attributes:?}"
        ));
    }
    Ok(function)
}

fn validate_sensitive_scope(
    source: &str,
    scope: &str,
    expected: &[(PhysicalSensitiveCall, usize)],
    all_calls: &[(usize, PhysicalSensitiveCall)],
    covered: &mut Vec<usize>,
) -> Result<(), String> {
    let start = scope.as_ptr() as usize - source.as_ptr() as usize;
    let end = start + scope.len();
    let mut actual = BTreeMap::new();
    for &(offset, kind) in all_calls {
        if (start..end).contains(&offset) {
            *actual.entry(kind).or_insert(0_usize) += 1;
            covered.push(offset);
        }
    }
    let expected = expected.iter().copied().collect::<BTreeMap<_, _>>();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "audited physical call-site census changed: actual={actual:?}, expected={expected:?} in {:?}",
            scope.lines().next().unwrap_or("").trim()
        ))
    }
}

fn validate_physical_sensitive_call_site_ownership(
    identity_source: &str,
    sources: &[(PathBuf, String)],
) -> Result<(), String> {
    let mut found_identity_owner = false;
    for (path, stored_source) in sources {
        let source = if path.ends_with("mamba_ssm/gpu/kernel_identity.rs") {
            found_identity_owner = true;
            identity_source
        } else {
            stored_source.as_str()
        };
        let calls =
            physical_sensitive_calls(source, path.ends_with("mamba_ssm/gpu/kernel_identity.rs"));
        let mut covered = Vec::new();
        if path.ends_with("mamba_ssm/gpu/kernel_identity.rs") {
            let identity_tests = active_test_module_scope(source, "physical_launch_tests")?;
            let inference_support = active_test_module_scope(source, "inference_test_support")?;
            let observer = direct_test_function_scope(inference_support, "observer", "")?;
            for (scope, visibility) in [
                (inference_support, "pub(incrate::mamba_ssm::gpu)"),
                (observer, "pub(incrate::mamba_ssm::gpu)"),
                (
                    active_production_method_scope(source, "PhysicalLaunchObservation", "resolve")?,
                    "",
                ),
            ] {
                let start = scope.as_ptr() as usize - source.as_ptr() as usize;
                let mask = source_mask(source);
                if compact_code(&mask[item_prefix_start(&mask, start)..start]) != visibility {
                    return Err("Inference test observer authority visibility changed".into());
                }
            }
            for (scope, expected) in [
                (
                    observer,
                    &[(PhysicalSensitiveCall::ObserverConstructor, 1)][..],
                ),
                (
                    active_production_method_scope(source, "PhysicalLaunchObservation", "resolve")?,
                    &[(PhysicalSensitiveCall::ObservationResolver, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_recording_physical_observer", "")?,
                    &[(PhysicalSensitiveCall::ObserverConstructor, 1)][..],
                ),
                (
                    audited_function_scope(source, "resolve_physical_launch_observation", "")?,
                    &[(PhysicalSensitiveCall::ObservationResolver, 1)][..],
                ),
                (
                    active_inlined_production_function_scope(
                        source,
                        "enqueue_with_physical_observation",
                    )?,
                    &[
                        (PhysicalSensitiveCall::ObservationResolver, 1),
                        (PhysicalSensitiveCall::SealedRecorderMutation, 1),
                        (PhysicalSensitiveCall::SealedRecorderInvalidation, 2),
                    ][..],
                ),
                (
                    active_inlined_production_function_scope(
                        source,
                        "enqueue_prepared_physical_launch",
                    )?,
                    &[
                        (PhysicalSensitiveCall::SealedRecorderMutation, 1),
                        (PhysicalSensitiveCall::SealedRecorderInvalidation, 2),
                    ][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "&mut T",
                        "record_before_enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::SealedRecorderMutation, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(source, "&mut T", "invalidate_enqueue")?,
                    &[(PhysicalSensitiveCall::SealedRecorderInvalidation, 1)][..],
                ),
                (
                    active_production_method_scope(source, "RecordingPhysicalObserver", "finish")?,
                    &[(PhysicalSensitiveCall::RecorderFinalizer, 1)][..],
                ),
                (
                    active_production_method_scope(
                        source,
                        "RecordingPhysicalObserver",
                        "finish_capture",
                    )?,
                    &[(PhysicalSensitiveCall::RecorderFinalizer, 1)][..],
                ),
                (
                    active_production_method_scope(
                        source,
                        "RecordingPhysicalObserver",
                        "record_before_enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::RecorderMutation, 1)][..],
                ),
                (
                    active_production_method_scope(
                        source,
                        "RecordingPhysicalObserver",
                        "invalidate_enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::RecorderInvalidation, 1)][..],
                ),
                (
                    audited_function_scope(source, "finish_recording_physical_observer", "")?,
                    &[(PhysicalSensitiveCall::ObserverFinalizer, 1)][..],
                ),
                (
                    audited_function_scope(source, "finish_recording_physical_capture", "")?,
                    &[(PhysicalSensitiveCall::ObserverCaptureFinalizer, 1)][..],
                ),
                (
                    audited_function_scope(
                        source,
                        "recorded_physical_trace_for_test",
                        "#[cfg(test)]",
                    )?,
                    &[
                        (PhysicalSensitiveCall::RecorderMutation, 1),
                        (PhysicalSensitiveCall::RecorderFinalizer, 1),
                    ][..],
                ),
                (
                    direct_test_function_scope(
                        identity_tests,
                        "private_recorder_preserves_order_and_sticky_invalidates_failures",
                        "#[test]",
                    )?,
                    &[
                        (PhysicalSensitiveCall::RecorderMutation, 4),
                        (PhysicalSensitiveCall::RecorderInvalidation, 1),
                        (PhysicalSensitiveCall::RecorderFinalizer, 1),
                    ][..],
                ),
            ] {
                validate_sensitive_scope(source, scope, expected, &calls, &mut covered)?;
            }
        } else if path.ends_with("mamba_ssm/gpu/blas.rs") {
            let graph_tests = active_test_module_scope(source, "physical_graph_tests")?;
            let inventory_tests = active_test_module_scope(source, "matvec_inventory_cuda_tests")?;
            for (scope, expected) in [
                (
                    audited_function_scope(source, "launch_bi_gemm", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "launch_bi_matvec", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    direct_test_function_scope(graph_tests, "record_tied_half_f32_trace", "")?,
                    &[
                        (PhysicalSensitiveCall::PreparedObserverFactory, 1),
                        (PhysicalSensitiveCall::ObserverFinalizationAuthority, 1),
                    ][..],
                ),
                (
                    direct_test_function_scope(
                        inventory_tests,
                        "typed_matvec_physical_observation_preserves_public_output_and_storage",
                        "#[test]#[ignore=]",
                    )?,
                    &[
                        (PhysicalSensitiveCall::PreparedObserverFactory, 1),
                        (PhysicalSensitiveCall::ObserverFinalizationAuthority, 1),
                    ][..],
                ),
                (
                    direct_test_function_scope(
                        inventory_tests,
                        "triad_native_half_context_inventory_records_all_projection_terminals",
                        "#[test]#[ignore=]",
                    )?,
                    &[
                        (PhysicalSensitiveCall::PreparedObserverFactory, 2),
                        (PhysicalSensitiveCall::ObserverFinalizationAuthority, 2),
                    ][..],
                ),
                (
                    audited_function_scope(source, "conversion_observation", "")?,
                    &[
                        (PhysicalSensitiveCall::ConversionObservation, 1),
                        (PhysicalSensitiveCall::ConversionArguments, 1),
                    ][..],
                ),
                (
                    audited_function_scope(source, "bi_upcast_to_f32", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "bi_downcast_from_f32", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_half_physical_observer", "")?,
                    &[(PhysicalSensitiveCall::PreparedObserverFactory, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_f32_physical_graph_observer", "")?,
                    &[(PhysicalSensitiveCall::PreparedObserverFactory, 1)][..],
                ),
                (
                    audited_function_scope(source, "record_half_physical_trace", "")?,
                    &[(PhysicalSensitiveCall::ObserverFinalizationAuthority, 1)][..],
                ),
                (
                    audited_function_scope(source, "record_prepared_f32_physical_trace", "")?,
                    &[(PhysicalSensitiveCall::ObserverFinalizationAuthority, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "BoundPhysicalGraphLaunches",
                        "enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::PreparedSubmission, 2)][..],
                ),
                (
                    audited_function_scope(source, "prepare_conversion_graph_launch", "")?,
                    &[(PhysicalSensitiveCall::ObservationResolution, 1)][..],
                ),
            ] {
                validate_sensitive_scope(source, scope, expected, &calls, &mut covered)?;
            }
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_triad/launch.rs") {
            for (scope, expected) in [
                (
                    audited_function_scope(
                        source,
                        "prepare_sm89_tf32_tn_pre_rna_graph_sequence",
                        "",
                    )?,
                    &[
                        (PhysicalSensitiveCall::ConversionArguments, 1),
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::InputTransformObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 2),
                    ][..],
                ),
                (
                    audited_function_scope(source, "enqueue_sm89_tf32_tn_pre_rna", "")?,
                    &[
                        (PhysicalSensitiveCall::ConversionArguments, 1),
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::InputTransformObservation, 1),
                        (PhysicalSensitiveCall::Submission, 2),
                    ][..],
                ),
                (
                    audited_function_scope(source, "prepare_physical_observer", "")?,
                    &[(PhysicalSensitiveCall::ObserverConstructionAuthority, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "ScalarLaunchControl",
                        "enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "PhysicalScalarLaunchControl",
                        "enqueue",
                    )?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::Submission, 1),
                    ][..],
                ),
                (
                    // The first-use proof drives a scalar body through its
                    // own controller, and submits once like the others.
                    active_inlined_production_method_scope(
                        source,
                        "ProvenScalarLaunch",
                        "enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_scalar_zero_f32", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_tf32_raw", "")?,
                    &[(PhysicalSensitiveCall::Submission, 4)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_tf32_splitk_f32", "")?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::Submission, 1),
                    ][..],
                ),
                (
                    audited_function_scope(
                        source,
                        "enqueue_validated_prepared_f32_triad_observed",
                        "",
                    )?,
                    &[(PhysicalSensitiveCall::GemmObservation, 2)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_scalar_forward", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_scalar_backward", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    // The half observation is built where the policy guards
                    // live now: the shared helper that the eager path, the
                    // prepared graph path and the retained half routes all call.
                    audited_function_scope(
                        source,
                        "resolve_half_gemm_observation_with_context",
                        "",
                    )?,
                    &[(PhysicalSensitiveCall::GemmObservation, 1)][..],
                ),
                (
                    audited_function_scope(source, "enqueue_half_gemm", "#[inline(always)]")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "BoundTriadPhysicalGraphSequence",
                        "enqueue",
                    )?,
                    &[(PhysicalSensitiveCall::PreparedSubmission, 1)][..],
                ),
                (
                    active_inlined_production_method_scope(
                        source,
                        "PreparedPhysicalScalarLaunchControl",
                        "enqueue",
                    )?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    audited_function_scope(
                        source,
                        "prepare_prepared_f32_direct_graph_sequence",
                        "",
                    )?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    audited_function_scope(
                        source,
                        "prepare_tf32_splitk_direct_graph_sequence",
                        "",
                    )?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    audited_function_scope(source, "enqueue_sm120_tma_prepared_observed", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "launch_sm120_auto_observed", "")?,
                    &[(PhysicalSensitiveCall::GemmObservation, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_sm120_auto_graph_sequence", "")?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    audited_function_scope(source, "enqueue_sm100_tcgen_prepared_observed", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "launch_sm100_auto_observed", "")?,
                    &[(PhysicalSensitiveCall::GemmObservation, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_sm100_auto_graph_sequence", "")?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    audited_function_scope(source, "enqueue_sm90a_wgmma_prepared_observed", "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)][..],
                ),
                (
                    audited_function_scope(source, "launch_sm90a_auto_observed", "")?,
                    &[(PhysicalSensitiveCall::GemmObservation, 1)][..],
                ),
                (
                    audited_function_scope(source, "prepare_sm90a_auto_graph_sequence", "")?,
                    &[
                        (PhysicalSensitiveCall::GemmObservation, 1),
                        (PhysicalSensitiveCall::ObservationResolution, 1),
                    ][..],
                ),
                (
                    // The prepared half graph node is resolved in the helper
                    // shared by the native path and the retained half routes,
                    // not in prepare_native_half_graph_identity any more.
                    audited_function_scope(
                        source,
                        "resolve_prepared_half_graph_node_with_context",
                        "",
                    )?,
                    &[(PhysicalSensitiveCall::ObservationResolution, 1)][..],
                ),
                (
                    // The retained small16 test resolves one physical node of
                    // its own to compare the eager and prepared identities.
                    direct_test_function_scope(
                        active_test_module_scope(source, "half_physical_trace_tests")?,
                        "triad_retained_half_small16_eager_and_prepared_physical_nodes_match",
                        "#[test]",
                    )?,
                    &[(PhysicalSensitiveCall::ObservationResolution, 1)][..],
                ),
            ] {
                validate_sensitive_scope(source, scope, expected, &calls, &mut covered)?;
            }
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference/identity.rs") {
            validate_sensitive_scope(
                source,
                audited_function_scope(source, "observation", "")?,
                &[(PhysicalSensitiveCall::InferenceObservation, 1)],
                &calls,
                &mut covered,
            )?;
            let tests = active_test_module_scope(source, "tests")?;
            for (name, count) in [
                (
                    "every_terminal_has_its_exact_storage_geometry_and_arithmetic_tuple",
                    1,
                ),
                (
                    "every_terminal_rejects_individually_changed_route_and_abi_before_enqueue",
                    2,
                ),
                (
                    "pair_store_symbol_is_bound_even_when_geometry_is_identical",
                    1,
                ),
                (
                    "allocation_spans_and_zero_reduction_use_actual_required_operands",
                    3,
                ),
                (
                    "allocation_bound_digest_updates_the_contained_route_and_rejects_stale_copy",
                    1,
                ),
            ] {
                validate_sensitive_scope(
                    source,
                    direct_test_function_scope(tests, name, "#[test]")?,
                    &[
                        (PhysicalSensitiveCall::InferenceObservation, count),
                        (PhysicalSensitiveCall::ObservationResolution, count),
                    ],
                    &calls,
                    &mut covered,
                )?;
            }
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference/runtime_bundle.rs") {
            // The retained-route runtime bundle is a launch owner in its own
            // right: one submission from its single launch function, the same
            // shape as every launcher in gemm_bi_inference.rs.
            validate_sensitive_scope(
                source,
                audited_function_scope(source, "launch_inference_bundle", "")?,
                &[(PhysicalSensitiveCall::Submission, 1)],
                &calls,
                &mut covered,
            )?;
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference.rs") {
            for name in [
                "launch_sm89_exact_n64",
                "launch_sm120_exact_n64",
                "launch_sm120_copyplan_t256",
                "launch_sm120_sliced",
                "launch_sm120_tma_postbias",
                "launch_f32_n128_s2",
                "launch_tf32",
                "launch_tf32_rna_n96",
                "launch_tf32_wide",
                "launch_sm120_tf32",
                "launch_sm120_half",
                "launch_sm89_half_pipeline",
                "launch_sm89_half_swizzle",
                "launch_sm89_half_s3",
                "launch_sm89_half_n64",
                "launch_sm89_cell",
                "launch_ladder",
                "launch_f32out_ladder",
            ] {
                validate_sensitive_scope(
                    source,
                    audited_function_scope(source, name, "")?,
                    &[(PhysicalSensitiveCall::Submission, 1)],
                    &calls,
                    &mut covered,
                )?;
            }
            let inventory_tests =
                active_test_module_scope(source, "observed_inventory_cuda_tests")?;
            validate_sensitive_scope(
                source,
                direct_test_function_scope(inventory_tests, "run_case", "")?,
                &[
                    (PhysicalSensitiveCall::PreparedObserverFactory, 1),
                    (PhysicalSensitiveCall::ObserverFinalizationAuthority, 1),
                ],
                &calls,
                &mut covered,
            )?;
        } else if path.ends_with("mamba_ssm/gpu/graph_capture.rs") {
            validate_sensitive_scope(
                source,
                audited_function_scope(source, "capture_into_graph_with_physical_plan", "")?,
                &[(
                    PhysicalSensitiveCall::ObserverCaptureFinalizationAuthority,
                    1,
                )],
                &calls,
                &mut covered,
            )?;
        }
        covered.sort_unstable();
        covered.dedup();
        let uncovered = calls
            .iter()
            .filter(|(offset, _)| covered.binary_search(offset).is_err())
            .collect::<Vec<_>>();
        let actual_uncovered = uncovered
            .iter()
            .fold(BTreeMap::new(), |mut counts, (_, kind)| {
                *counts.entry(*kind).or_insert(0_usize) += 1;
                counts
            });
        let expected_uncovered = if path.ends_with("mamba_ssm/gpu/blas.rs") {
            [
                (PhysicalSensitiveCall::PreparedObserverFactory, 1),
                (PhysicalSensitiveCall::ObserverFinalizationAuthority, 1),
                (PhysicalSensitiveCall::Submission, 1),
                (PhysicalSensitiveCall::PreparedSubmission, 1),
                (PhysicalSensitiveCall::ObservationResolution, 1),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_triad/launch.rs") {
            [
                (PhysicalSensitiveCall::ObserverConstructionAuthority, 1),
                (PhysicalSensitiveCall::Submission, 1),
                (PhysicalSensitiveCall::PreparedSubmission, 1),
                (PhysicalSensitiveCall::ObservationResolution, 1),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference/identity.rs") {
            [(PhysicalSensitiveCall::ObservationResolution, 1)]
                .into_iter()
                .collect::<BTreeMap<_, _>>()
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference/runtime_bundle.rs") {
            // The import of the submission function sits outside any audited
            // scope, as it does in gemm_bi_inference.rs.
            [(PhysicalSensitiveCall::Submission, 1)]
                .into_iter()
                .collect::<BTreeMap<_, _>>()
        } else if path.ends_with("mamba_ssm/gpu/gemm_bi_inference.rs") {
            [
                (PhysicalSensitiveCall::Submission, 1),
                (PhysicalSensitiveCall::PreparedObserverFactory, 1),
                (PhysicalSensitiveCall::ObserverFinalizationAuthority, 1),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        } else if path.ends_with("mamba_ssm/gpu/graph_capture.rs") {
            [(
                PhysicalSensitiveCall::ObserverCaptureFinalizationAuthority,
                1,
            )]
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };
        if actual_uncovered != expected_uncovered {
            return Err(format!(
                "{} has unaudited physical-sensitive references: actual={actual_uncovered:?}, expected={expected_uncovered:?}",
                path.display(),
            ));
        }
    }
    if !found_identity_owner {
        return Err(
            "whole-source physical ownership census did not find kernel_identity.rs".into(),
        );
    }
    Ok(())
}

fn validate_physical_owner_exports(identity_source: &str) -> Result<(), String> {
    let tokens = rust_code_tokens(identity_source);
    let sensitive = [
        "PhysicalLaunchObservation",
        "PhysicalConversionArguments",
        "RecordingPhysicalObserver",
        "prepare_recording_physical_observer",
        "finish_recording_physical_observer",
        "finish_recording_physical_capture",
        "finish_capture",
        "resolve",
        "record",
        "record_before_enqueue",
        "invalidate_enqueue",
        "finish",
        "enqueue_with_physical_observation",
        "enqueue_prepared_physical_launch",
        "resolve_physical_launch_observation",
    ];
    let mut macro_bodies = BTreeMap::<String, Vec<String>>::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != "macro_rules" {
            continue;
        }
        let Some(name) = tokens.get(index + 2).map(|token| token.text.clone()) else {
            continue;
        };
        let Some(open) = tokens[index + 3..]
            .iter()
            .position(|token| token.text == "{")
            .map(|offset| index + 3 + offset)
        else {
            continue;
        };
        let mut depth = 0_u32;
        let mut close = open;
        for (offset, candidate) in tokens[open..].iter().enumerate() {
            match candidate.text.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        macro_bodies.insert(
            name,
            tokens[open + 1..close]
                .iter()
                .map(|token| token.text.clone())
                .collect(),
        );
    }
    let mut sensitive_macros = BTreeSet::new();
    loop {
        let before = sensitive_macros.len();
        for (name, body) in &macro_bodies {
            if body.iter().any(|token| {
                sensitive.contains(&token.as_str()) || sensitive_macros.contains(token)
            }) {
                sensitive_macros.insert(name.clone());
            }
        }
        if sensitive_macros.len() == before {
            break;
        }
    }
    let compact_identity = compact_code(identity_source);
    for name in &sensitive_macros {
        let declaration = format!("macro_rules!{name}");
        let Some(start) = compact_identity.find(&declaration) else {
            continue;
        };
        let prefix = &compact_identity[..start];
        if prefix.ends_with("#[macro_export]") {
            return Err(format!(
                "physical observation authority is exported through macro {name}"
            ));
        }
    }
    for (index, token) in tokens.iter().enumerate() {
        if !matches!(token.text.as_str(), "use" | "type") {
            continue;
        }
        let start = tokens[..index]
            .iter()
            .rposition(|candidate| matches!(candidate.text.as_str(), ";" | "{" | "}"))
            .map_or(0, |offset| offset + 1);
        let end = tokens[index + 1..]
            .iter()
            .position(|candidate| candidate.text == ";")
            .map(|offset| index + 1 + offset)
            .unwrap_or(tokens.len());
        let public = tokens[start..index]
            .iter()
            .any(|candidate| candidate.text == "pub");
        if public
            && tokens[index + 1..end].iter().any(|candidate| {
                sensitive.contains(&candidate.text.as_str())
                    || sensitive_macros.contains(&candidate.text)
            })
        {
            return Err(format!(
                "physical observation authority has an owner-side public alias: {}",
                tokens[index + 1..end]
                    .iter()
                    .map(|candidate| candidate.text.as_str())
                    .collect::<String>()
            ));
        }
    }
    Ok(())
}

fn has_recorded_physical_trace_literal(source: &str) -> bool {
    let mask = source_mask(source);
    let tokens = rust_code_tokens(source);
    let function_bodies = tokens
        .windows(2)
        .filter_map(|pair| {
            (pair[0].text == "fn"
                && pair[1]
                    .text
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphabetic))
            .then(|| function_body_open(&mask, pair[0].start, &pair[1].text).ok())
            .flatten()
        })
        .collect::<BTreeSet<_>>();
    tokens.windows(2).any(|pair| {
        pair[0].text == "RecordedPhysicalTrace"
            && pair[1].text == "{"
            // `fn helper() -> RecordedPhysicalTrace {` opens the function body,
            // whereas a literal inside that body has a distinct opening token.
            && !function_bodies.contains(&pair[1].start)
    })
}

fn validate_physical_trace_ownership_boundary(
    identity_source: &str,
    sources: &[(PathBuf, String)],
) -> Result<(), String> {
    validate_physical_owner_exports(identity_source)?;
    validate_owner_sensitive_function_topology(identity_source)?;
    validate_physical_sensitive_call_site_ownership(identity_source, sources)?;
    let identity = compact_code(identity_source);
    validate_concrete_physical_enqueue_primitive(identity_source)?;
    validate_prepared_physical_enqueue_primitive(identity_source)?;
    for forbidden in [
        "pub(crate)fnphysical_enqueue_event_digest",
        "pubfnphysical_enqueue_event_digest",
        "pub(crate)fnphysical_enqueue_provenance_digest",
        "pubfnphysical_enqueue_provenance_digest",
        "from_observed_nodes",
    ] {
        if identity.contains(forbidden) {
            return Err(format!(
                "forgeable physical provenance API remains: {forbidden}"
            ));
        }
    }
    for forbidden in [
        "pubstructPhysicalTraceRecorder",
        "pub(crate)structPhysicalTraceRecorder",
        "pub(super)structPhysicalTraceRecorder",
    ] {
        if identity.contains(forbidden) {
            return Err("the physical trace recorder core must remain private".into());
        }
    }
    let recorder_struct = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalTraceRecorder",
        0,
    )?);
    if recorder_struct
        != "structPhysicalTraceRecorder{capacity:usize,nodes:Vec<ResolvedPhysicalKernelLaunch>,enqueue_events:Vec<PhysicalEnqueueEvent>,overflowed:bool,enqueue_failed:bool,}"
    {
        return Err("the private recorder fields must remain exact and inaccessible".into());
    }
    let enqueue_event = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalEnqueueEvent",
        0,
    )?);
    if enqueue_event != "structPhysicalEnqueueEvent{launch:ResolvedPhysicalKernelLaunch,}"
        || identity.contains("pubstructPhysicalEnqueueEvent")
        || identity.contains("pub(crate)structPhysicalEnqueueEvent")
        || identity.contains("pub(super)structPhysicalEnqueueEvent")
    {
        return Err("physical enqueue events must remain private exact launch snapshots".into());
    }
    let observer_trait = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "trait",
        "PhysicalLaunchObserver",
        0,
    )?);
    if !observer_trait.starts_with(
        "traitPhysicalLaunchObserver:physical_observer_private::Sealed{constENABLED:bool;",
    ) || !identity
        .contains("pub(super)traitPhysicalLaunchObserver:physical_observer_private::Sealed{")
        || observer_trait.contains("observe(")
        || observer_trait.contains("ResolvedPhysicalKernelLaunch")
        || observer_trait.contains("record_before_enqueue")
    {
        return Err(
            "the visible observer trait must be sealed and expose no node/event mutation method"
                .into(),
        );
    }
    let private_module = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "mod",
        "physical_observer_private",
        0,
    )?);
    if !private_module.starts_with("modphysical_observer_private{")
        || !private_module.contains("pubstructAuthority(());")
        || !private_module.contains("pubtraitSealed{")
        || !private_module.contains(
            "fnrecord_before_enqueue(&mutself,authority:&mutAuthority,launch:ResolvedPhysicalKernelLaunch,)->Result<(),String>;",
        )
        || !private_module
            .contains("fninvalidate_enqueue(&mutself,authority:&mutAuthority);")
    {
        return Err(
            "recorder mutation must require the owning module's private sealed authority"
                .into(),
        );
    }
    let observation_struct = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalLaunchObservation",
        0,
    )?);
    if observation_struct
        != "structPhysicalLaunchObservation{gemm:Option<PhysicalGemmObservation>,conversion:Option<PhysicalConversionObservation>,input_transform:Option<PhysicalInputTransformObservation>,}"
        || identity
            .matches("pub(super)structPhysicalLaunchObservation{")
            .count()
            != 1
        || identity.contains("enumPhysicalLaunchObservation")
    {
        return Err("physical launch observations must remain opaque outside their owner".into());
    }
    let gemm_observation = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalGemmObservation",
        0,
    )?);
    let conversion_observation = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalConversionObservation",
        0,
    )?);
    if gemm_observation
        != "structPhysicalGemmObservation{logical_dtype:PolicyDtype,resources_digest:Option<Sha256Digest>,route:ResolvedGemmRoute,inference:Option<super::gemm_bi_inference::identity::Arguments>,}"
        || conversion_observation
            != "structPhysicalConversionObservation{kind:PhysicalLaunchKind,logical_op:ResolvedGemmOp,logical_dtype:PolicyDtype,shape:(usize,usize,usize),strides:(usize,usize,usize),element_count:u64,arguments:PhysicalConversionArguments,}"
        || identity.contains("pubstructPhysicalGemmObservation")
        || identity.contains("pub(crate)structPhysicalGemmObservation")
        || identity.contains("pub(super)structPhysicalGemmObservation")
        || identity.contains("pubstructPhysicalConversionObservation")
        || identity.contains("pub(crate)structPhysicalConversionObservation")
        || identity.contains("pub(super)structPhysicalConversionObservation")
    {
        return Err("physical observation payloads must remain private and exact".into());
    }
    let transform_observation = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "PhysicalInputTransformObservation",
        0,
    )?);
    if transform_observation
        != "structPhysicalInputTransformObservation{identity:ResolvedInputTransform,logical_op:ResolvedGemmOp,logical_dtype:PolicyDtype,shape:(usize,usize,usize),strides:(usize,usize,usize),arguments:PhysicalConversionArguments,symbol:&'staticstr,}"
        || [
            "pubstruct",
            "pub(crate)struct",
            "pub(super)struct",
            "pub(incrate::mamba_ssm::gpu)struct",
        ]
        .iter()
        .any(|visibility| {
            identity.contains(&format!("{visibility}PhysicalInputTransformObservation"))
        })
    {
        return Err("physical input-transform payload must remain private and exact".into());
    }
    let observation_implementations =
        impl_scopes_for_type(identity_source, "PhysicalLaunchObservation");
    if observation_implementations.len() != 1 {
        return Err("PhysicalLaunchObservation must have one owning implementation".into());
    }
    let observation_implementation = observation_implementations[0];
    if direct_function_names_at_depth(observation_implementation, 1)
        != [
            "gemm",
            "inference",
            "conversion",
            "input_transform",
            "resolve",
        ]
    {
        return Err("PhysicalLaunchObservation gained a constructor or resolution wrapper".into());
    }
    let gemm_constructor = compact_code(active_production_method_scope(
        observation_implementation,
        "PhysicalLaunchObservation",
        "gemm",
    )?);
    let conversion_constructor = compact_code(active_production_method_scope(
        observation_implementation,
        "PhysicalLaunchObservation",
        "conversion",
    )?);
    let observation_resolver = compact_code(active_production_method_scope(
        observation_implementation,
        "PhysicalLaunchObservation",
        "resolve",
    )?);
    let inference_constructor = compact_code(active_production_method_scope(
        observation_implementation,
        "PhysicalLaunchObservation",
        "inference",
    )?);
    let transform_constructor = compact_code(active_production_method_scope(
        observation_implementation,
        "PhysicalLaunchObservation",
        "input_transform",
    )?);
    if !gemm_constructor.contains(
        "Self{gemm:Some(PhysicalGemmObservation{logical_dtype,resources_digest,route,inference:None,}),conversion:None,input_transform:None,}",
    ) || !conversion_constructor.contains(
        "Self{gemm:None,conversion:Some(PhysicalConversionObservation{kind,logical_op,logical_dtype,shape,strides,element_count,arguments,}),input_transform:None,}",
    ) || !inference_constructor.contains(
        "Self{gemm:Some(PhysicalGemmObservation{logical_dtype:arguments.storage[0],resources_digest:None,route,inference:Some(arguments),}),conversion:None,input_transform:None,}",
    ) || !transform_constructor.contains(
        "Self{gemm:None,conversion:None,input_transform:Some(PhysicalInputTransformObservation{identity,logical_op,logical_dtype,shape,strides,arguments,symbol,}),}",
    ) || !observation_resolver.starts_with("fnresolve<")
        || !observation_resolver.contains("match(self.gemm,self.conversion,self.input_transform){")
        || !observation_resolver.contains("Some(PhysicalGemmObservation{")
        || !observation_resolver.contains("Some(PhysicalConversionObservation{")
        || !observation_resolver.contains("Some(PhysicalInputTransformObservation{")
        || !observation_resolver.contains("_=>Err(")
    {
        return Err(
            "physical observations must carry exactly one private constructor-owned payload".into(),
        );
    }
    let trace_struct = compact_code(unique_named_item_scope_at_depth(
        identity_source,
        "struct",
        "RecordedPhysicalTrace",
        0,
    )?);
    if trace_struct
        != "structRecordedPhysicalTrace{context:GemmRouteIdentity,binding:PhysicalGraphBinding,launches:ResolvedPhysicalLaunchSet,enqueue_provenance:Sha256Digest,nodes:Box<[ResolvedPhysicalKernelLaunch]>,}"
    {
        return Err("recorded physical trace fields must remain private and exact".into());
    }
    let core_implementations = impl_scopes_for_type(identity_source, "PhysicalTraceRecorder");
    if core_implementations.len() != 1 {
        return Err("the private physical recorder core must have one implementation".into());
    }
    if direct_function_names_at_depth(core_implementations[0], 1)
        != [
            "with_capacity",
            "validate_start",
            "validate_complete",
            "record",
            "invalidate_enqueue",
            "finish",
        ]
    {
        return Err("the private physical recorder core gained a forwarding method".into());
    }
    let core = compact_code(core_implementations[0]);
    for forbidden in [
        "pubfnrecord(",
        "pub(crate)fnrecord(",
        "pub(super)fnrecord(",
        "pubfnfinish(",
        "pub(crate)fnfinish(",
        "pub(super)fnfinish(",
    ] {
        if core.contains(forbidden) {
            return Err("physical recorder mutation/finalization must remain private".into());
        }
    }
    let observer_implementations =
        impl_scopes_for_type(identity_source, "RecordingPhysicalObserver")
            .into_iter()
            .filter(|implementation| {
                compact_code(&source_mask(implementation))
                    .starts_with("implRecordingPhysicalObserver{")
            })
            .collect::<Vec<_>>();
    if observer_implementations.len() != 1 {
        return Err("RecordingPhysicalObserver must have one exact inherent implementation".into());
    }
    let observer_impl = observer_implementations[0];
    if direct_function_names_at_depth(observer_impl, 1)
        != [
            "with_argument_identity",
            "with_replay_provenance",
            "validate_start",
            "validate_capture_start",
            "validate_capture_binding",
            "finish",
            "finish_capture",
        ]
    {
        return Err("RecordingPhysicalObserver gained a private authority wrapper".into());
    }
    let finish = compact_code(active_production_method_scope(
        observer_impl,
        "RecordingPhysicalObserver",
        "finish",
    )?);
    if !finish.starts_with(
        "fnfinish(self,context:GemmRouteIdentity)->Result<RecordedPhysicalTrace,String>{",
    ) || identity.contains(
        "pub(super)fnfinish(self,context:GemmRouteIdentity)->Result<RecordedPhysicalTrace,String>{",
    ) || !identity.contains(
        "pub(super)fnfinish_recording_physical_observer(observer:RecordingPhysicalObserver,context:GemmRouteIdentity,)->Result<RecordedPhysicalTrace,String>{observer.finish(context)}",
    ) || finish.contains("ResolvedPhysicalKernelLaunch")
        || finish.contains("Sha256Digest")
        || !finish.contains("letbinding=self.replay_provenance.as_ref().ok_or_else(")
        || !finish.ends_with("self.recorder.finish(context,binding)}")
    {
        return Err("observer finalization must consume only its privately recorded state".into());
    }
    let trace_implementations = impl_scopes_for_type(identity_source, "RecordedPhysicalTrace");
    if trace_implementations.len() != 1 {
        return Err("RecordedPhysicalTrace must have one constructor-free implementation".into());
    }
    if direct_function_names_at_depth(trace_implementations[0], 1)
        != [
            "context",
            "nodes",
            "launches",
            "validate_integrity",
            "manifest",
        ]
    {
        return Err("RecordedPhysicalTrace gained a trace-minting method".into());
    }
    for visibility in ["pubfn", "pub(crate)fn", "pub(super)fn"] {
        let mut remaining = identity.as_str();
        while let Some(offset) = remaining.find(visibility) {
            let signature = &remaining[offset..];
            let signature = &signature[..signature.find('{').unwrap_or(signature.len())];
            let owned_finish = signature.starts_with(
                "pub(super)fnfinish_recording_physical_observer(observer:RecordingPhysicalObserver,context:GemmRouteIdentity,)->Result<RecordedPhysicalTrace,String>",
            );
            if signature.contains("RecordedPhysicalTrace") && !owned_finish {
                return Err("a public function can mint a recorded physical trace".into());
            }
            remaining = &remaining[offset + visibility.len()..];
        }
    }
    for (path, source) in sources {
        let mask = source_mask(source);
        if path.ends_with("kernel_identity.rs") {
            continue;
        }
        if has_recorded_physical_trace_literal(source)
            || mask.contains("RecordedPhysicalTrace::from_")
            || mask.contains("physical_enqueue_event_digest(")
            || mask.contains("physical_enqueue_provenance_digest(")
            || mask.contains("PhysicalTraceRecorder")
            || mask.contains("PhysicalLaunchObserver::observe(")
            || mask.contains("PhysicalLaunchObserver>::observe(")
            || mask.contains("observer.observe(")
            || mask.contains("observe_then_enqueue(")
            || mask.contains("record_before_enqueue(")
        {
            return Err(format!(
                "{} can bypass the opaque physical recorder",
                path.display()
            ));
        }
    }
    Ok(())
}

fn replace_nth_in_function(
    source: &str,
    function: &str,
    needle: &str,
    replacement: &str,
    occurrence: usize,
) -> String {
    let scope = active_production_function_scope(source, function)
        .unwrap_or_else(|error| panic!("locate {function} mutation scope: {error}"));
    let scope_start = scope.as_ptr() as usize - source.as_ptr() as usize;
    let relative = scope
        .match_indices(needle)
        .nth(occurrence)
        .map(|(offset, _)| offset)
        .unwrap_or_else(|| panic!("{function} has no occurrence {occurrence} of {needle}"));
    let start = scope_start + relative;
    let end = start + needle.len();
    format!("{}{}{}", &source[..start], replacement, &source[end..])
}

fn replace_nth_in_inlined_function(
    source: &str,
    function: &str,
    needle: &str,
    replacement: &str,
    occurrence: usize,
) -> String {
    let scope = active_inlined_production_function_scope(source, function)
        .unwrap_or_else(|error| panic!("locate {function} mutation scope: {error}"));
    let scope_start = scope.as_ptr() as usize - source.as_ptr() as usize;
    let relative = scope
        .match_indices(needle)
        .nth(occurrence)
        .map(|(offset, _)| offset)
        .unwrap_or_else(|| panic!("{function} has no occurrence {occurrence} of {needle}"));
    let start = scope_start + relative;
    let end = start + needle.len();
    format!("{}{}{}", &source[..start], replacement, &source[end..])
}

fn validate_physical_observation_topology(
    identity_source: &str,
    blas_source: &str,
    launch_source: &str,
) -> Result<(), String> {
    validate_concrete_physical_enqueue_primitive(identity_source)?;
    validate_prepared_physical_enqueue_primitive(identity_source)?;
    if source_mask(launch_source).contains("observe_then_enqueue")
        || source_mask(blas_source).contains("observe_then_enqueue")
    {
        return Err("the closure-based physical enqueue boundary must stay deleted".into());
    }

    for (entry, shared) in [
        ("gemm_bi_forward_typed", "gemm_bi_forward_typed_in"),
        ("gemm_bi_backward_dw_typed", "gemm_bi_backward_dw_typed_in"),
        ("gemm_bi_backward_dx_typed", "gemm_bi_backward_dx_typed_in"),
    ] {
        assert_unique_live_delegation(blas_source, entry, shared)?;
    }

    for function in ["bi_upcast_to_f32", "bi_downcast_from_f32"] {
        let scope = active_production_function_scope(blas_source, function)?;
        assert_no_raw_cuda_launch(scope, function)?;
        assert_calls_with_config(
            scope,
            function,
            "enqueue_with_physical_observation",
            1,
            2,
            "config",
        )?;
        let code = compact_code(&source_mask(scope));
        if !code.contains("ifO::ENABLED{Some(conversion_observation(") {
            return Err(format!(
                "{function} must derive conversion identity only in the observer-enabled arm"
            ));
        }
    }

    let half_enqueue =
        active_inlined_production_function_scope(launch_source, "enqueue_half_gemm")?;
    assert_no_raw_cuda_launch(half_enqueue, "enqueue_half_gemm")?;
    assert_calls_with_config(
        half_enqueue,
        "enqueue_half_gemm",
        "enqueue_with_physical_observation",
        1,
        2,
        "config",
    )?;
    let half_code = compact_code(&source_mask(half_enqueue));
    if !half_code.contains("ifO::ENABLED{Some(resolve_half_gemm_observation(") {
        return Err("native-half identity must be observer-gated before concrete enqueue".into());
    }

    for (function, expected_calls) in [
        ("gemm_bi_forward_tc_with_tile_in", 1),
        ("gemm_bi_backward_dw_tc_with_tile_in", 1),
        ("gemm_bi_backward_dx_tc_with_tile_in", 1),
        ("gemm_bi_forward_typed_in", 4),
        ("gemm_bi_backward_dw_typed_in", 3),
        ("gemm_bi_backward_dx_typed_in", 3),
    ] {
        let scope = active_production_function_scope(launch_source, function)?;
        assert_no_raw_cuda_launch(scope, function)?;
        assert_calls_with_config(
            scope,
            function,
            "enqueue_half_gemm",
            expected_calls,
            3,
            "cfg",
        )?;
    }

    for (function, wrapper) in [
        ("gemm_bi_forward_sub_with_control", "enqueue_scalar_forward"),
        (
            "gemm_bi_backward_dw_with_control",
            "enqueue_scalar_backward",
        ),
        (
            "gemm_bi_backward_dx_with_control",
            "enqueue_scalar_backward",
        ),
    ] {
        let scope = active_production_function_scope(launch_source, function)?;
        assert_no_raw_cuda_launch(scope, function)?;
        if matching_call_ranges(scope, wrapper)?.is_empty() {
            return Err(format!("{function} has no controlled scalar enqueue"));
        }
    }

    for function in ["enqueue_scalar_forward", "enqueue_scalar_backward"] {
        let scope = active_production_function_scope(launch_source, function)?;
        assert_no_raw_cuda_launch(scope, function)?;
        assert_calls_with_config(
            scope,
            function,
            "enqueue_with_physical_observation",
            1,
            2,
            "config",
        )?;
    }

    for controller in ["ScalarLaunchControl", "PhysicalScalarLaunchControl"] {
        let enqueue = active_inlined_production_method_scope(launch_source, controller, "enqueue")?;
        assert_no_raw_cuda_launch(enqueue, &format!("{controller}::enqueue"))?;
        assert_calls_with_config(
            enqueue,
            &format!("{controller}::enqueue"),
            "enqueue_with_physical_observation",
            1,
            2,
            "config",
        )?;
    }

    let ordinary =
        active_production_function_scope(launch_source, "enqueue_validated_prepared_f32_triad")?;
    let observed = active_production_function_scope(
        launch_source,
        "enqueue_validated_prepared_f32_triad_observed",
    )?;
    for (owner, scope, observation) in [
        ("enqueue_validated_prepared_f32_triad", ordinary, "None"),
        (
            "enqueue_validated_prepared_f32_triad_observed",
            observed,
            "Some(PhysicalLaunchObservation::gemm(",
        ),
    ] {
        assert_no_raw_cuda_launch(scope, owner)?;
        assert_calls_with_config(scope, owner, "enqueue_scalar_zero_f32", 1, 3, "*config")?;
        let scalar_zero = matching_call_ranges(scope, "enqueue_scalar_zero_f32")?[0];
        let scalar_arguments = top_level_arguments(&scope[scalar_zero.0..scalar_zero.1])?;
        if !compact_code(&source_mask(scalar_arguments[5])).starts_with(observation) {
            return Err(format!(
                "{owner} ScalarZero arm has the wrong observation capability"
            ));
        }
        let tf32 = matching_call_ranges(scope, "enqueue_tf32_f32")?;
        if tf32.len() != 1 {
            return Err(format!("{owner} must delegate TF32 exactly once"));
        }
        let tf32_arguments = top_level_arguments(&scope[tf32[0].0..tf32[0].1])?;
        if tf32_arguments.len() != 4 {
            return Err(format!("{owner} TF32 delegation changed shape"));
        }
        let raw = compact_code(&source_mask(tf32_arguments[2]));
        if !raw.contains("config:*config") || !raw.contains(&format!("observation:{observation}")) {
            return Err(format!(
                "{owner} TF32 arm must carry the exact config and observation through its raw launch package"
            ));
        }
    }

    let scalar_zero = active_production_function_scope(launch_source, "enqueue_scalar_zero_f32")?;
    assert_no_raw_cuda_launch(scalar_zero, "enqueue_scalar_zero_f32")?;
    assert_calls_with_config(
        scalar_zero,
        "enqueue_scalar_zero_f32",
        "enqueue_with_physical_observation",
        1,
        2,
        "config",
    )?;

    let tf32 = active_production_function_scope(launch_source, "enqueue_tf32_f32")?;
    assert_no_raw_cuda_launch(tf32, "enqueue_tf32_f32")?;
    if matching_call_ranges(tf32, "enqueue_tf32_raw")?.len() != 1
        || !matching_call_ranges(tf32, "enqueue_with_physical_observation")?.is_empty()
    {
        return Err("enqueue_tf32_f32 must have one transitive raw-ABI delegation".into());
    }

    let tf32_raw = active_production_function_scope(launch_source, "enqueue_tf32_raw")?;
    assert_no_raw_cuda_launch(tf32_raw, "enqueue_tf32_raw")?;
    assert_calls_with_config(
        tf32_raw,
        "enqueue_tf32_raw",
        "enqueue_with_physical_observation",
        4,
        2,
        "launch.config",
    )?;
    for (start, end) in matching_call_ranges(tf32_raw, "enqueue_with_physical_observation")? {
        let arguments = top_level_arguments(&tf32_raw[start..end])?;
        if arguments.len() != 4 || compact_code(&source_mask(arguments[3])) != "launch.observation"
        {
            return Err(
                "every TF32 raw ABI alternative must carry the same opaque observation".into(),
            );
        }
    }

    let raw_helpers = direct_function_names_at_depth(launch_source, 0)
        .into_iter()
        .filter(|name| name.starts_with("enqueue_tf32_raw"))
        .collect::<Vec<_>>();
    if raw_helpers != ["enqueue_tf32_raw"] {
        return Err(format!(
            "TF32 raw enqueue helper census changed: {raw_helpers:?}"
        ));
    }
    Ok(())
}

#[test]
fn physical_observation_uses_one_inseparable_enqueue_primitive() {
    validate_physical_observation_topology(IDENTITY_SOURCE, BLAS_SOURCE, LAUNCH_SOURCE).unwrap();
}

#[test]
fn physical_observation_oracle_rejects_zero_tf32_transitive_mutations() {
    let mutations = [
        (
            "observed scalar-zero wrapper deletion",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_validated_prepared_f32_triad_observed",
                "enqueue_scalar_zero_f32",
                "enqueue_scalar_zero_f32_without_observation",
                0,
            ),
        ),
        (
            "observed TF32 wrapper deletion",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_validated_prepared_f32_triad_observed",
                "enqueue_tf32_f32",
                "enqueue_tf32_f32_without_observation",
                0,
            ),
        ),
        (
            "scalar-zero helper wrong config",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_scalar_zero_f32",
                "&mut builder, config, observation",
                "&mut builder, wrong_config, observation",
                0,
            ),
        ),
        (
            "TF32 helper wrong config",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_tf32_raw",
                "&mut builder,\n                    launch.config,",
                "&mut builder,\n                    wrong_config,",
                0,
            ),
        ),
        (
            "scalar-zero second raw launch",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_scalar_zero_f32",
                ".map_err(|error|",
                ".and_then(|_| unsafe { builder.launch(config) }).map_err(|error|",
                0,
            ),
        ),
        (
            "scalar-zero token-separated raw launch",
            replace_nth_in_function(
                LAUNCH_SOURCE,
                "enqueue_scalar_zero_f32",
                ".map_err(|error|",
                ".and_then(|_| unsafe { builder . launch /* bypass */ (config) }).map_err(|error|",
                0,
            ),
        ),
        (
            "new TF32 raw helper alternative",
            format!(
                "{LAUNCH_SOURCE}\nunsafe fn enqueue_tf32_raw_alternative(builder: &mut Builder, config: LaunchConfig) {{ let _ = unsafe {{ builder.launch(config) }}; }}"
            ),
        ),
    ];
    let accepted = mutations
        .iter()
        .filter_map(|(label, source)| {
            validate_physical_observation_topology(IDENTITY_SOURCE, BLAS_SOURCE, source)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        accepted.is_empty(),
        "round-4 enqueue oracle accepted zero/TF32 mutations: {accepted:?}"
    );
}

#[test]
fn physical_observation_oracle_rejects_control_flow_and_raw_launch_mutations() {
    validate_concrete_physical_enqueue_primitive(IDENTITY_SOURCE).unwrap();
    let mutations = [
        replace_nth_in_inlined_function(
            IDENTITY_SOURCE,
            "enqueue_with_physical_observation",
            "if O::ENABLED {",
            "if false {",
            0,
        ),
        replace_nth_in_inlined_function(
            IDENTITY_SOURCE,
            "enqueue_with_physical_observation",
            "match unsafe { builder.launch(config) }",
            "match unsafe { builder.launch(other_config) }",
            0,
        ),
        replace_nth_in_inlined_function(
            IDENTITY_SOURCE,
            "enqueue_with_physical_observation",
            "match unsafe { builder.launch(config) }",
            "let _ = unsafe { builder.launch(config) }; match unsafe { builder.launch(config) }",
            0,
        ),
        replace_nth_in_inlined_function(
            IDENTITY_SOURCE,
            "enqueue_with_physical_observation",
            "match unsafe { builder.launch(config) }",
            "let _ = unsafe { builder . launch /* bypass */ (config) }; match unsafe { builder.launch(config) }",
            0,
        ),
        replace_nth_in_inlined_function(
            IDENTITY_SOURCE,
            "enqueue_with_physical_observation",
            "observation: Option<PhysicalLaunchObservation>",
            "enqueue: impl FnOnce() -> Result<(), String>",
            0,
        ),
    ];
    for mutation in mutations {
        assert!(
            validate_concrete_physical_enqueue_primitive(&mutation).is_err(),
            "concrete physical enqueue mutation was accepted"
        );
    }

    let primitive = active_inlined_production_function_scope(
        IDENTITY_SOURCE,
        "enqueue_with_physical_observation",
    )
    .unwrap();
    let duplicate = format!("{IDENTITY_SOURCE}\n{primitive}");
    assert!(validate_concrete_physical_enqueue_primitive(&duplicate).is_err());
    let conditional = IDENTITY_SOURCE.replacen(
        "#[inline(always)]\npub(super) unsafe fn enqueue_with_physical_observation",
        "#[cfg(any())]\n#[inline(always)]\npub(super) unsafe fn enqueue_with_physical_observation",
        1,
    );
    assert!(validate_concrete_physical_enqueue_primitive(&conditional).is_err());

    let valid_call = "fn enqueue() { enqueue_with_physical_observation(observer, &mut builder, config, observation)?; }";
    assert_no_raw_cuda_launch(valid_call, "synthetic valid caller").unwrap();
    assert_calls_with_config(
        valid_call,
        "synthetic valid caller",
        "enqueue_with_physical_observation",
        1,
        2,
        "config",
    )
    .unwrap();
    for invalid in [
        "fn enqueue() { unsafe { builder.launch(config) }; }",
        "fn enqueue() { enqueue_with_physical_observation(observer, &mut builder, config, observation)?; unsafe { builder.launch(config) }; }",
        "fn enqueue() { enqueue_with_physical_observation(observer, &mut builder, wrong_config, observation)?; }",
        "fn enqueue() { enqueue_with_physical_observation(observer, &mut builder, config, observation)?; unsafe { builder . launch /* bypass */ (config) }; }",
    ] {
        let accepted = assert_no_raw_cuda_launch(invalid, "synthetic invalid caller").is_ok()
            && assert_calls_with_config(
                invalid,
                "synthetic invalid caller",
                "enqueue_with_physical_observation",
                1,
                2,
                "config",
            )
            .is_ok();
        assert!(
            !accepted,
            "raw/wrong-config caller mutation was accepted: {invalid}"
        );
    }
}

#[test]
fn prepared_physical_enqueue_oracle_rejects_body_and_topology_mutations() {
    validate_physical_observation_topology(IDENTITY_SOURCE, BLAS_SOURCE, LAUNCH_SOURCE).unwrap();
    let mutations = [
        (
            "missing recorder operation",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "physical_observer_private::Sealed::record_before_enqueue",
                "physical_observer_private::Sealed::record_before_enqueue_removed",
                0,
            ),
        ),
        (
            "wrong launch config",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "match unsafe { builder.launch(config) }",
                "match unsafe { builder.launch(other_config) }",
                0,
            ),
        ),
        (
            "extra raw launch",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "match unsafe { builder.launch(config) }",
                "let _ = unsafe { builder.launch(config) }; match unsafe { builder.launch(config) }",
                0,
            ),
        ),
        (
            "token-separated extra raw launch",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "match unsafe { builder.launch(config) }",
                "let _ = unsafe { builder . launch /* bypass */ (config) }; match unsafe { builder.launch(config) }",
                0,
            ),
        ),
        (
            "callback parameter",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "launch: ResolvedPhysicalKernelLaunch,",
                "launch: ResolvedPhysicalKernelLaunch, enqueue: impl FnOnce() -> Result<(), String>,",
                0,
            ),
        ),
        (
            "deferred closure launch",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "match unsafe { builder.launch(config) }",
                "let deferred = || unsafe { builder.launch(config) }; match deferred()",
                0,
            ),
        ),
        (
            "early return",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "let mut authority = physical_observer_private::Authority::new();",
                "if bypass { return Ok(()); } let mut authority = physical_observer_private::Authority::new();",
                0,
            ),
        ),
        (
            "missing Driver sticky invalidation",
            replace_nth_in_inlined_function(
                IDENTITY_SOURCE,
                "enqueue_prepared_physical_launch",
                "physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);",
                "",
                1,
            ),
        ),
    ];
    let accepted = mutations
        .iter()
        .filter_map(|(label, identity)| {
            validate_physical_observation_topology(identity, BLAS_SOURCE, LAUNCH_SOURCE)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        accepted.is_empty(),
        "prepared physical enqueue oracle accepted body mutations: {accepted:?}"
    );

    let primitive = active_inlined_production_function_scope(
        IDENTITY_SOURCE,
        "enqueue_prepared_physical_launch",
    )
    .unwrap();
    let duplicate = format!("{IDENTITY_SOURCE}\n{primitive}");
    assert!(
        validate_physical_observation_topology(&duplicate, BLAS_SOURCE, LAUNCH_SOURCE).is_err(),
        "a duplicate prepared physical enqueue item was accepted"
    );
    let conditional = IDENTITY_SOURCE.replacen(
        "#[inline(always)]\npub(super) unsafe fn enqueue_prepared_physical_launch",
        "#[cfg(any())]\n#[inline(always)]\npub(super) unsafe fn enqueue_prepared_physical_launch",
        1,
    );
    assert!(
        validate_physical_observation_topology(&conditional, BLAS_SOURCE, LAUNCH_SOURCE).is_err(),
        "a conditionally disabled prepared physical enqueue item was accepted"
    );
}

#[test]
fn f32_preparation_and_cache_store_no_physical_observer_digest() {
    let prepared = compact_code(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) struct PreparedF32TriadLaunch",
    ));
    assert!(!prepared.contains("physical_resources_digest"));
    for function in [
        "prepare_scalar_f32",
        "prepare_scalar_zero_f32",
        "prepare_tf32_f32",
        "prepare_f32_triad",
        "prepare_exact_scalar_f32_triad",
    ] {
        let scope = active_production_function_scope(LAUNCH_SOURCE, function).unwrap();
        let code = source_mask(scope);
        assert!(
            !code.contains("physical_digest("),
            "{function} must not hash physical observer resources"
        );
    }
    let observed = active_production_function_scope(
        LAUNCH_SOURCE,
        "enqueue_validated_prepared_f32_triad_observed",
    )
    .unwrap();
    assert_eq!(
        source_mask(observed)
            .matches("prepared.resources.physical_digest()")
            .count(),
        1,
        "the observer-enabled enqueue path alone derives role-bound physical resources"
    );
}

#[test]
fn task2_physical_observer_sources_have_no_clippy_allowances() {
    for (label, source) in [
        ("physical identity", IDENTITY_SOURCE),
        ("BLAS physical enqueue", BLAS_SOURCE),
        ("triad physical enqueue", LAUNCH_SOURCE),
        ("physical contract", TF32_CONTRACT_TEST_SOURCE),
    ] {
        assert!(
            !compact_code(&source_mask(source)).contains("allow(clippy::"),
            "{label} may not suppress a Clippy diagnostic"
        );
    }
}

#[test]
fn half_physical_trace_authority_is_crate_private() {
    assert_code_contains_all(
        BLAS_SOURCE,
        &[
            "pub(in crate::mamba_ssm::gpu) struct HalfPhysicalTraceRequest",
            "pub(in crate::mamba_ssm::gpu) unsafe fn record_half_physical_trace",
        ],
        "half physical trace authority",
    );
    assert_code_excludes_all(
        KERNEL_IDENTITY_CUDA_SOURCE,
        &["HalfPhysicalTraceRequest", "record_half_physical_trace"],
        "external CUDA identity tests",
    );
}

#[test]
fn physical_cuda_launch_prepared_error_is_test_only() {
    let launch_error = compact_code(
        unique_named_item_scope_at_depth(IDENTITY_SOURCE, "enum", "PhysicalCudaLaunchError", 0)
            .unwrap(),
    );
    assert!(
        launch_error.contains("#[cfg(test)]Prepared(&'staticstr),"),
        "the synthetic prepared body error must not exist in production builds"
    );
    let launch_error_impl = impl_scopes_for_type(IDENTITY_SOURCE, "PhysicalCudaLaunchError")
        .into_iter()
        .find(|implementation| {
            compact_code(&source_mask(implementation)).starts_with("implPhysicalCudaLaunchError{")
        })
        .expect("PhysicalCudaLaunchError inherent implementation");
    let contextual_formatter = compact_code(
        active_production_method_scope(
            launch_error_impl,
            "PhysicalCudaLaunchError",
            "with_driver_context",
        )
        .unwrap(),
    );
    assert!(
        contextual_formatter.contains("#[cfg(test)]Self::Prepared(error)=>error.to_string(),"),
        "the contextual synthetic prepared error formatter must use the same test-only gate"
    );
    let formatter = compact_code(
        active_production_function_scope(GRAPH_CAPTURE_SOURCE, "physical_capture_body_error")
            .unwrap(),
    );
    assert!(
        formatter
            .contains("#[cfg(test)]PhysicalCudaLaunchError::Prepared(error)=>error.to_string(),"),
        "the synthetic prepared body error formatter must use the same test-only gate"
    );
    let package_enqueue = compact_code(
        active_inlined_production_method_scope(
            BLAS_SOURCE,
            "BoundPhysicalGraphLaunches",
            "enqueue",
        )
        .unwrap(),
    );
    assert!(
        package_enqueue.contains(
            "#[cfg(test)]ifself.fail_before_enqueue{returnErr(PhysicalCudaLaunchError::Prepared("
        ),
        "the synthetic prepared body error must be constructed only by the test injection"
    );
}

#[test]
fn physical_trace_provenance_has_no_crate_visible_mint_or_src_bypass() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect_rust_sources(&root, &mut sources);
    validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &sources).unwrap();

    for mutation in [
        "pub(crate) fn physical_enqueue_event_digest() {}",
        "pub fn physical_enqueue_provenance_digest() {}",
        "impl RecordedPhysicalTrace { pub(crate) fn from_observed_nodes() {} }",
        "pub(crate) struct PhysicalTraceRecorder {}",
        "impl PhysicalTraceRecorder { pub(crate) fn record(&mut self, _: ResolvedPhysicalKernelLaunch) {} }",
        "pub fn mint_trace(_: GemmRouteIdentity, _: Vec<ResolvedPhysicalKernelLaunch>, _: Sha256Digest) -> RecordedPhysicalTrace { unreachable!() }",
    ] {
        let identity = format!("{IDENTITY_SOURCE}\n{mutation}");
        assert!(
            validate_physical_trace_ownership_boundary(&identity, &sources).is_err(),
            "forgeable provenance mutation was accepted: {mutation}"
        );
    }

    let mut bypass = sources.clone();
    bypass.push((
        PathBuf::from("src/forged.rs"),
        r#"
        fn forged(mut node: ResolvedPhysicalKernelLaunch) {
            let route = node.gemm_route().unwrap();
            node.gemm_route = Some(route);
            let event = physical_enqueue_event_digest(&node).unwrap();
            let provenance = physical_enqueue_provenance_digest(&[event]);
            let _ = RecordedPhysicalTrace::from_observed_nodes(
                context,
                vec![node],
                vec![event],
                provenance,
            );
        }
        "#
        .into(),
    ));
    assert!(validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &bypass).is_err());

    let mut core_bypass = sources.clone();
    core_bypass.push((
        PathBuf::from("src/forged_core.rs"),
        "fn forged(node: ResolvedPhysicalKernelLaunch) { let mut core = PhysicalTraceRecorder::with_capacity(1); core.record(node); core.finish(context); }"
            .into(),
    ));
    assert!(validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &core_bypass).is_err());

    let crate_visible_bypasses = [
        (
            "ordinary observer method",
            "fn forged(mut observer: RecordingPhysicalObserver, node: ResolvedPhysicalKernelLaunch) { observer.observe(node).unwrap(); let _ = observer.finish(context); }",
        ),
        (
            "UFCS observer method",
            "fn forged(mut observer: RecordingPhysicalObserver, node: ResolvedPhysicalKernelLaunch) { <RecordingPhysicalObserver as PhysicalLaunchObserver>::observe(&mut observer, node).unwrap(); let _ = observer.finish(context); }",
        ),
        (
            "fake-success enqueue closure",
            "fn forged(mut observer: RecordingPhysicalObserver, node: ResolvedPhysicalKernelLaunch) { observe_then_enqueue(&mut observer, config, |_, _| Ok(node), |_| Ok(())).unwrap(); let _ = observer.finish(context); }",
        ),
        (
            "sibling GEMM semantic constructor",
            "fn forged(route: ResolvedGemmRoute) { let _ = PhysicalLaunchObservation :: gemm(dtype, None, route); }",
        ),
        (
            "sibling conversion semantic constructor",
            "fn forged(arguments: PhysicalConversionArguments) { let _ = PhysicalLaunchObservation :: conversion(kind, op, dtype, shape, strides, count, arguments); }",
        ),
        (
            "sibling Inference semantic constructor",
            "fn forged() { let _ = PhysicalLaunchObservation::inference(arguments, route); }",
        ),
        (
            "aliased sibling Inference semantic constructor",
            "use PhysicalLaunchObservation::inference as mint; fn forged() { let _ = mint(arguments, route); }",
        ),
        (
            "sibling input-transform semantic constructor",
            "fn forged() { let _ = PhysicalLaunchObservation::input_transform(symbol, op, dtype, shape, strides, identity, arguments); }",
        ),
        (
            "aliased sibling input-transform semantic constructor",
            "type Alias = PhysicalLaunchObservation; fn forged() { let mint = Alias::input_transform; let _ = mint(symbol, op, dtype, shape, strides, identity, arguments); }",
        ),
        (
            "sibling conversion argument constructor",
            "fn forged() { let _ = PhysicalConversionArguments :: new(source, source_bytes, destination, destination_bytes); }",
        ),
        (
            "sibling recording observer constructor",
            "fn forged() { let _ = RecordingPhysicalObserver :: with_argument_identity(1, Some(context), |_, _| Ok([1; 32])); }",
        ),
        (
            "sibling recording observer construction authority",
            "fn forged() { let _ = prepare_recording_physical_observer (1, Some(context), |_, _| Ok([1; 32])); }",
        ),
        (
            "sibling prepared observer factory",
            "fn forged(ctx: &GpuCtx, ranges: &[PhysicalArgumentRange]) { let _ = prepare_physical_observer (ctx, 1, ranges); }",
        ),
        (
            "sibling recording observer finalizer",
            "fn forged(observer: RecordingPhysicalObserver) { let _ = observer . finish(context); }",
        ),
        (
            "sibling UFCS recording observer finalizer",
            "fn forged(observer: RecordingPhysicalObserver) { let _ = RecordingPhysicalObserver :: finish(observer, context); }",
        ),
        (
            "sibling recording observer finalization authority",
            "fn forged(observer: RecordingPhysicalObserver) { let _ = finish_recording_physical_observer (observer, context); }",
        ),
        (
            "sibling recording observer capture finalization authority",
            "fn forged(observer: RecordingPhysicalObserver) { let _ = finish_recording_physical_capture (observer, ctx, manifest); }",
        ),
        (
            "sibling physical submission",
            "fn forged(mut observer: RecordingPhysicalObserver, mut builder: LaunchArgs<'_>) { let _ = enqueue_with_physical_observation (&mut observer, &mut builder, config, observation); }",
        ),
        (
            "sibling prepared physical submission",
            "fn forged(mut observer: RecordingPhysicalObserver, mut builder: LaunchArgs<'_>) { let _ = enqueue_prepared_physical_launch (&mut observer, &mut builder, config, node); }",
        ),
        (
            "sibling physical observation resolution",
            "fn forged(observer: RecordingPhysicalObserver, observation: PhysicalLaunchObservation) { let _ = resolve_physical_launch_observation (&observer, observation, config); }",
        ),
        (
            "cfg-hidden sibling physical submission",
            "#[cfg(any())] fn forged(mut observer: RecordingPhysicalObserver, mut builder: LaunchArgs<'_>) { let _ = enqueue_with_physical_observation (&mut observer, &mut builder, config, observation); }",
        ),
        (
            "aliased sibling physical submission",
            "use crate::mamba_ssm::gpu::kernel_identity::enqueue_with_physical_observation as submit; fn forged() { let _ = submit(observer, builder, config, observation); }",
        ),
        (
            "function-pointer sibling observer factory",
            "fn forged() { let factory = prepare_physical_observer; let _ = factory(ctx, 1, ranges); }",
        ),
        (
            "aliased sibling GEMM semantic constructor",
            "use PhysicalLaunchObservation :: gemm as observe_gemm; fn forged() { let _ = observe_gemm(dtype, None, route); }",
        ),
        (
            "type-aliased sibling GEMM semantic constructor",
            "type Alias = PhysicalLaunchObservation; fn forged() { let mint = Alias::gemm; let _ = mint(dtype, None, route); }",
        ),
        (
            "import-aliased sibling conversion semantic constructor",
            "use crate::mamba_ssm::gpu::kernel_identity::PhysicalLaunchObservation as Alias; fn forged() { let mint = Alias::conversion; let _ = mint(kind, op, dtype, shape, strides, count, arguments); }",
        ),
        (
            "UFCS through conversion-argument type alias",
            "type Alias = PhysicalConversionArguments; fn forged() { let _ = <Alias>::new(source, source_bytes, destination, destination_bytes); }",
        ),
        (
            "alias-derived observer-constructor function pointer",
            "type Alias = RecordingPhysicalObserver; fn forged() { let mint = Alias::with_argument_identity; let _ = mint(1, Some(context), resolver); }",
        ),
        (
            "macro-expanded semantic constructor through alias",
            "type Alias = PhysicalLaunchObservation; macro_rules! mint { () => { Alias::gemm(dtype, None, route) } } fn forged() { let _ = mint!(); }",
        ),
        (
            "aliased sibling capture finalization authority",
            "use crate::mamba_ssm::gpu::kernel_identity::finish_recording_physical_capture as finalize; fn forged() { let _ = finalize(observer, ctx, manifest); }",
        ),
        (
            "alias-derived prepared submission function pointer",
            "use crate::mamba_ssm::gpu::kernel_identity::enqueue_prepared_physical_launch as submit; fn forged() { let enqueue = submit; let _ = enqueue(observer, builder, config, node); }",
        ),
        (
            "macro-expanded observation resolution alias",
            "use crate::mamba_ssm::gpu::kernel_identity::resolve_physical_launch_observation as resolve; macro_rules! mint { () => { resolve(observer, observation, config) } } fn forged() { let _ = mint!(); }",
        ),
    ];
    let accepted = crate_visible_bypasses
        .iter()
        .filter_map(|(label, bypass_source)| {
            let mut bypass = sources.clone();
            bypass.push((
                PathBuf::from(format!("src/forged_{label}.rs")),
                (*bypass_source).into(),
            ));
            validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &bypass)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        accepted.is_empty(),
        "crate-visible APIs can mint a trace without CUDA enqueue: {accepted:?}"
    );

    for (authority, alias, caller) in [
        (
            "enqueue_with_physical_observation",
            "submit",
            "fn forged() { let _ = submit(observer, builder, config, observation); }",
        ),
        (
            "prepare_recording_physical_observer",
            "prepare",
            "fn forged() { let _ = prepare(ctx, 1, epoch, resolver); }",
        ),
        (
            "finish_recording_physical_observer",
            "finish_eager",
            "fn forged() { let _ = finish_eager(observer, context); }",
        ),
        (
            "finish_recording_physical_capture",
            "finish_capture",
            "fn forged() { let _ = finish_capture(observer, ctx, manifest); }",
        ),
        (
            "enqueue_prepared_physical_launch",
            "submit_prepared",
            "fn forged() { let _ = submit_prepared(observer, builder, config, node); }",
        ),
        (
            "resolve_physical_launch_observation",
            "resolve_observation",
            "fn forged() { let _ = resolve_observation(observer, observation, config); }",
        ),
    ] {
        let owner_reexport =
            format!("{IDENTITY_SOURCE}\npub(super) use self::{authority} as {alias};");
        let mut bypass = sources
            .iter()
            .map(|(path, source)| {
                if path.ends_with("mamba_ssm/gpu/kernel_identity.rs") {
                    (path.clone(), owner_reexport.clone())
                } else {
                    (path.clone(), source.clone())
                }
            })
            .collect::<Vec<_>>();
        bypass.push((
            PathBuf::from(format!("src/owner_reexport_{alias}.rs")),
            caller.into(),
        ));
        assert!(
            validate_physical_trace_ownership_boundary(&owner_reexport, &bypass).is_err(),
            "owner-side re-export of {authority} as {alias} was accepted"
        );
    }

    let owner_macro_reexport = format!(
        "{IDENTITY_SOURCE}\nmacro_rules! submit_macro {{ () => {{ enqueue_prepared_physical_launch(observer, builder, config, node) }} }} pub(super) use submit_macro as exported_submit;"
    );
    let mut macro_bypass = sources
        .iter()
        .map(|(path, source)| {
            if path.ends_with("mamba_ssm/gpu/kernel_identity.rs") {
                (path.clone(), owner_macro_reexport.clone())
            } else {
                (path.clone(), source.clone())
            }
        })
        .collect::<Vec<_>>();
    macro_bypass.push((
        PathBuf::from("src/owner_macro_reexport.rs"),
        "fn forged() { let _ = exported_submit!(); }".into(),
    ));
    assert!(
        validate_physical_trace_ownership_boundary(&owner_macro_reexport, &macro_bypass).is_err(),
        "owner-side macro re-export of prepared submission authority was accepted"
    );

    let mut decoys = sources.clone();
    decoys.push((
        PathBuf::from("src/physical_observer_decoys.rs"),
        r#"
        // PhysicalLaunchObservation::gemm(dtype, None, route);
        // RecordingPhysicalObserver::with_argument_identity(1, None, resolver);
        const DECOYS: &str = "PhysicalConversionArguments::new(a, b, c, d); observer.finish(context); enqueue_with_physical_observation(observer, builder, config, observation);";
        "#
        .into(),
    ));
    validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &decoys)
        .expect("comment and string decoys are not executable ownership uses");

    let duplicate_semantics = replace_nth_in_function(
        LAUNCH_SOURCE,
        "resolve_half_gemm_observation_with_context",
        "Ok(PhysicalLaunchObservation::gemm(",
        "let _duplicate = PhysicalLaunchObservation :: gemm(half_policy_dtype(observation.dtype)?, None, route); Ok(PhysicalLaunchObservation::gemm(",
        0,
    );
    let duplicate_sources = sources
        .iter()
        .map(|(path, source)| {
            if path.ends_with("mamba_ssm/gpu/gemm_bi_triad/launch.rs") {
                (path.clone(), duplicate_semantics.clone())
            } else {
                (path.clone(), source.clone())
            }
        })
        .collect::<Vec<_>>();
    assert!(
        validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &duplicate_sources).is_err(),
        "an extra semantic constructor inside an otherwise approved function was accepted"
    );

    let conditional_constructor = LAUNCH_SOURCE.replacen(
        "pub(in crate::mamba_ssm::gpu) fn prepare_physical_observer(",
        "#[cfg(any())]\npub(in crate::mamba_ssm::gpu) fn prepare_physical_observer(",
        1,
    );
    let conditional_sources = sources
        .iter()
        .map(|(path, source)| {
            if path.ends_with("mamba_ssm/gpu/gemm_bi_triad/launch.rs") {
                (path.clone(), conditional_constructor.clone())
            } else {
                (path.clone(), source.clone())
            }
        })
        .collect::<Vec<_>>();
    assert!(
        validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &conditional_sources).is_err(),
        "a cfg-hidden approved observer constructor was accepted"
    );
}

#[test]
fn physical_904_census_keeps_test_authorities_gated_private_and_exact() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect_rust_sources(&root, &mut sources);
    validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &sources).unwrap();
    for signature in [
        "fn helper() -> RecordedPhysicalTrace { unreachable!() }",
        "fn helper() -> crate::mamba_ssm::gpu::kernel_identity::RecordedPhysicalTrace { unreachable!() }",
    ] {
        assert!(!has_recorded_physical_trace_literal(signature));
        let mut with_return_type = sources.clone();
        with_return_type.push((
            PathBuf::from("src/return_type_fixture.rs"),
            signature.into(),
        ));
        validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &with_return_type).unwrap();
    }
    for literal in [
        "fn forged() -> RecordedPhysicalTrace { RecordedPhysicalTrace { context, binding, launches, enqueue_provenance, nodes } }",
        "fn forged() { let _ = RecordedPhysicalTrace\n{ context, binding, launches, enqueue_provenance, nodes }; }",
        "fn forged() { let _ = crate::kernel_identity::RecordedPhysicalTrace/* gap */{ context, binding, launches, enqueue_provenance, nodes }; }",
    ] {
        assert!(has_recorded_physical_trace_literal(literal));
        let mut with_literal = sources.clone();
        with_literal.push((
            PathBuf::from("src/forged_literal_fixture.rs"),
            literal.into(),
        ));
        assert!(
            validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &with_literal).is_err()
        );
    }
    for (label, from, to) in [
        (
            "missing test gate",
            "#[cfg(test)]\npub(in crate::mamba_ssm::gpu) mod inference_test_support",
            "pub(in crate::mamba_ssm::gpu) mod inference_test_support",
        ),
        (
            "public test module",
            "pub(in crate::mamba_ssm::gpu) mod inference_test_support",
            "pub mod inference_test_support",
        ),
        (
            "public observer factory",
            "pub(in crate::mamba_ssm::gpu) fn observer(",
            "pub fn observer(",
        ),
        (
            "extra observer factory",
            "RecordingPhysicalObserver::with_argument_identity(128, None, resolve).unwrap()",
            "let _extra = RecordingPhysicalObserver::with_argument_identity(128, None, resolve); RecordingPhysicalObserver::with_argument_identity(128, None, resolve).unwrap()",
        ),
        (
            "public resolver",
            "    fn resolve<O: PhysicalLaunchObserver>(",
            "    pub(super) fn resolve<O: PhysicalLaunchObserver>(",
        ),
        (
            "extra nested resolver",
            "launch.arguments_digest = arguments.resolve(observer, &route, config)?;",
            "let _extra = arguments.resolve(observer, &route, config)?; launch.arguments_digest = arguments.resolve(observer, &route, config)?;",
        ),
        (
            "public transform payload",
            "struct PhysicalInputTransformObservation {",
            "pub(super) struct PhysicalInputTransformObservation {",
        ),
        (
            "wrong Inference payload",
            "inference: Some(arguments),",
            "inference: None,",
        ),
    ] {
        assert!(
            IDENTITY_SOURCE.contains(from),
            "missing mutation target: {label}"
        );
        let identity = IDENTITY_SOURCE.replacen(from, to, 1);
        assert!(
            validate_physical_trace_ownership_boundary(&identity, &sources).is_err(),
            "accepted {label}"
        );
    }
    for (path_suffix, module) in [
        ("mamba_ssm/gpu/blas.rs", "matvec_inventory_cuda_tests"),
        (
            "mamba_ssm/gpu/gemm_bi_inference.rs",
            "observed_inventory_cuda_tests",
        ),
    ] {
        let mut changed = sources.clone();
        let (_, source) = changed
            .iter_mut()
            .find(|(path, _)| path.ends_with(path_suffix))
            .unwrap();
        let from = format!("#[cfg(test)]\nmod {module}");
        assert!(source.contains(&from));
        *source = source.replacen(&from, &format!("mod {module}"), 1);
        assert!(
            validate_physical_trace_ownership_boundary(IDENTITY_SOURCE, &changed).is_err(),
            "ungated {module} accepted"
        );
    }
}

#[test]
fn physical_owner_rejects_forwarders_and_visible_function_pointers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect_rust_sources(&root, &mut sources);

    let forwarding_mutations = [
        (
            "GEMM semantic constructor",
            "pub(super) fn owner_forward_gemm(route: ResolvedGemmRoute) -> PhysicalLaunchObservation { PhysicalLaunchObservation::gemm(PolicyDtype::F32, None, route) }",
        ),
        (
            "conversion semantic constructor",
            "pub(super) fn owner_forward_conversion(arguments: PhysicalConversionArguments) -> PhysicalLaunchObservation { PhysicalLaunchObservation::conversion(kind, op, dtype, shape, strides, count, arguments) }",
        ),
        (
            "conversion argument constructor",
            "pub(super) fn owner_forward_conversion_arguments() -> PhysicalConversionArguments { PhysicalConversionArguments::new(source, source_bytes, destination, destination_bytes) }",
        ),
        (
            "observer constructor",
            "pub(super) fn owner_forward_observer() -> Result<RecordingPhysicalObserver, String> { RecordingPhysicalObserver::with_argument_identity(1, Some(context), resolver) }",
        ),
        (
            "observer construction authority",
            "pub(super) fn owner_forward_prepare(ctx: &GpuCtx) -> Result<RecordingPhysicalObserver, String> { prepare_recording_physical_observer(ctx, 1, epoch, resolver) }",
        ),
        (
            "prepared observer factory",
            "pub(super) fn owner_forward_prepared_observer(ctx: &GpuCtx, ranges: &[PhysicalArgumentRange]) -> Result<RecordingPhysicalObserver, String> { prepare_physical_observer(ctx, 1, ranges) }",
        ),
        (
            "observer method finalizer",
            "pub(super) fn owner_forward_observer_finish(observer: RecordingPhysicalObserver) -> Result<RecordedPhysicalTrace, String> { observer.finish(context) }",
        ),
        (
            "observer finalization authority",
            "pub(super) fn owner_forward_finish(observer: RecordingPhysicalObserver) -> Result<RecordedPhysicalTrace, String> { finish_recording_physical_observer(observer, context) }",
        ),
        (
            "capture finalization authority",
            "pub(super) fn owner_forward_capture_finish(observer: RecordingPhysicalObserver, ctx: &GpuCtx, manifest: &PreparedPhysicalCaptureManifest) -> Result<CapturedPhysicalGraphPlan, String> { finish_recording_physical_capture(observer, ctx, manifest) }",
        ),
        (
            "physical submission",
            "pub(super) unsafe fn owner_forward_submit(observer: &mut RecordingPhysicalObserver, builder: &mut cudarc::driver::LaunchArgs<'_>, config: cudarc::driver::LaunchConfig, observation: Option<PhysicalLaunchObservation>) -> Result<(), PhysicalCudaLaunchError> { enqueue_with_physical_observation(observer, builder, config, observation) }",
        ),
        (
            "prepared physical submission",
            "pub(super) unsafe fn owner_forward_prepared_submit(observer: &mut RecordingPhysicalObserver, builder: &mut cudarc::driver::LaunchArgs<'_>, config: cudarc::driver::LaunchConfig, node: ResolvedPhysicalKernelLaunch) -> Result<(), PhysicalCudaLaunchError> { enqueue_prepared_physical_launch(observer, builder, config, node) }",
        ),
        (
            "observation resolution authority",
            "pub(super) fn owner_forward_resolve(observer: &RecordingPhysicalObserver, observation: PhysicalLaunchObservation, config: cudarc::driver::LaunchConfig) -> Result<ResolvedPhysicalKernelLaunch, String> { resolve_physical_launch_observation(observer, observation, config) }",
        ),
        (
            "renamed GEMM semantic forwarding authority",
            "type OwnerObservationAlias = PhysicalLaunchObservation; pub(super) fn owner_forward_renamed_gemm(route: ResolvedGemmRoute) -> PhysicalLaunchObservation { OwnerObservationAlias::gemm(PolicyDtype::F32, None, route) }",
        ),
        (
            "renamed prepared submission forwarding authority",
            "use self::enqueue_prepared_physical_launch as renamed_owner_submit; pub(super) unsafe fn owner_forward_renamed_submit(observer: &mut RecordingPhysicalObserver, builder: &mut cudarc::driver::LaunchArgs<'_>, config: cudarc::driver::LaunchConfig, node: ResolvedPhysicalKernelLaunch) -> Result<(), PhysicalCudaLaunchError> { renamed_owner_submit(observer, builder, config, node) }",
        ),
    ];
    let accepted_forwarders = forwarding_mutations
        .iter()
        .filter_map(|(label, mutation)| {
            let identity = format!("{IDENTITY_SOURCE}\n{mutation}");
            validate_physical_trace_ownership_boundary(&identity, &sources)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();
    let pointer_mutations = [
        (
            "renamed GEMM constructor constant",
            "type OwnerObservation = PhysicalLaunchObservation; type OwnerGemmConstructor = fn(PolicyDtype, Option<Sha256Digest>, ResolvedGemmRoute) -> PhysicalLaunchObservation; pub(super) const OWNER_GEMM: OwnerGemmConstructor = OwnerObservation::gemm;",
        ),
        (
            "conversion constructor constant",
            "type OwnerConversionConstructor = fn(PhysicalLaunchKind, ResolvedGemmOp, PolicyDtype, (usize, usize, usize), (usize, usize, usize), u64, PhysicalConversionArguments) -> PhysicalLaunchObservation; pub(super) const OWNER_CONVERSION: OwnerConversionConstructor = PhysicalLaunchObservation::conversion;",
        ),
        (
            "conversion argument constructor static",
            "type OwnerConversionArguments = fn(cudarc::driver::sys::CUdeviceptr, u64, cudarc::driver::sys::CUdeviceptr, u64) -> PhysicalConversionArguments; pub(super) static OWNER_CONVERSION_ARGUMENTS: OwnerConversionArguments = PhysicalConversionArguments::new;",
        ),
        (
            "observer constructor constant",
            "type OwnerResolver = fn(cudarc::driver::sys::CUdeviceptr, u64) -> Result<Sha256Digest, String>; type OwnerObserverConstructor = fn(usize, Option<GemmRouteIdentity>, OwnerResolver) -> Result<RecordingPhysicalObserver, String>; pub(super) const OWNER_OBSERVER: OwnerObserverConstructor = RecordingPhysicalObserver::with_argument_identity::<OwnerResolver>;",
        ),
        (
            "observer construction authority constant",
            "type OwnerResolver = fn(cudarc::driver::sys::CUdeviceptr, u64) -> Result<Sha256Digest, String>; type OwnerPrepareObserver = fn(&GpuCtx, usize, Option<ManagedAllocationEpochStamp>, OwnerResolver) -> Result<RecordingPhysicalObserver, String>; pub(super) const OWNER_PREPARE_OBSERVER: OwnerPrepareObserver = prepare_recording_physical_observer::<OwnerResolver>;",
        ),
        (
            "prepared observer factory static",
            "type OwnerPreparedObserver = fn(&GpuCtx, usize, &[super::gemm_bi_triad::PhysicalArgumentRange]) -> Result<RecordingPhysicalObserver, String>; pub(super) static OWNER_PREPARED_OBSERVER: OwnerPreparedObserver = super::gemm_bi_triad::prepare_physical_observer;",
        ),
        (
            "observer method finalizer constant",
            "type OwnerObserverFinish = fn(RecordingPhysicalObserver, GemmRouteIdentity) -> Result<RecordedPhysicalTrace, String>; pub(super) const OWNER_OBSERVER_FINISH: OwnerObserverFinish = RecordingPhysicalObserver::finish;",
        ),
        (
            "renamed observer finish static",
            "use self::finish_recording_physical_observer as renamed_finish; type OwnerFinish = fn(RecordingPhysicalObserver, GemmRouteIdentity) -> Result<RecordedPhysicalTrace, String>; pub(super) static OWNER_FINISH: OwnerFinish = renamed_finish;",
        ),
        (
            "renamed capture finish constant",
            "use self::finish_recording_physical_capture as renamed_capture_finish; type OwnerCaptureFinish = fn(RecordingPhysicalObserver, &GpuCtx, &PreparedPhysicalCaptureManifest) -> Result<CapturedPhysicalGraphPlan, String>; pub(super) const OWNER_CAPTURE_FINISH: OwnerCaptureFinish = renamed_capture_finish;",
        ),
        (
            "physical submission constant",
            "type OwnerSubmit = for<'a> unsafe fn(&mut RecordingPhysicalObserver, &mut cudarc::driver::LaunchArgs<'a>, cudarc::driver::LaunchConfig, Option<PhysicalLaunchObservation>) -> Result<(), PhysicalCudaLaunchError>; pub(super) const OWNER_SUBMIT: OwnerSubmit = enqueue_with_physical_observation::<RecordingPhysicalObserver>;",
        ),
        (
            "renamed prepared submission static",
            "use self::enqueue_prepared_physical_launch as renamed_prepared_submit; type OwnerPreparedSubmit = for<'a> unsafe fn(&mut RecordingPhysicalObserver, &mut cudarc::driver::LaunchArgs<'a>, cudarc::driver::LaunchConfig, ResolvedPhysicalKernelLaunch) -> Result<(), PhysicalCudaLaunchError>; pub(super) static OWNER_PREPARED_SUBMIT: OwnerPreparedSubmit = renamed_prepared_submit;",
        ),
        (
            "renamed observation resolver constant",
            "use self::resolve_physical_launch_observation as renamed_resolve; type OwnerResolve = fn(&RecordingPhysicalObserver, PhysicalLaunchObservation, cudarc::driver::LaunchConfig) -> Result<ResolvedPhysicalKernelLaunch, String>; pub(super) const OWNER_RESOLVE: OwnerResolve = renamed_resolve::<RecordingPhysicalObserver>;",
        ),
    ];
    let accepted_pointers = pointer_mutations
        .iter()
        .filter_map(|(label, mutation)| {
            let identity = format!("{IDENTITY_SOURCE}\n{mutation}");
            validate_physical_trace_ownership_boundary(&identity, &sources)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();
    assert!(
        accepted_forwarders.is_empty() && accepted_pointers.is_empty(),
        "owner authorities were accepted: forwarders={accepted_forwarders:?}, visible_function_pointers={accepted_pointers:?}"
    );
}

#[test]
fn physical_owner_rejects_private_method_authorities() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect_rust_sources(&root, &mut sources);

    let forwarding_mutations = [
        (
            "direct capture finalizer",
            r#"
pub(super) fn owner_forward_direct_capture_finish(
    recording: RecordingPhysicalObserver,
    ctx: &GpuCtx,
    manifest: &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String> {
    recording.finish_capture(ctx, manifest)
}
"#,
        ),
        (
            "arbitrary receiver eager finalizer with result alias",
            r#"
type OwnerTraceResult = Result<RecordedPhysicalTrace, String>;
pub(super) fn owner_forward_arbitrary_finish(
    recording: RecordingPhysicalObserver,
    context: GemmRouteIdentity,
) -> OwnerTraceResult {
    recording.finish(context)
}
"#,
        ),
        (
            "renamed local eager finalizer with type and result aliases",
            r#"
type OwnerMovedObserverAlias = RecordingPhysicalObserver;
type OwnerMovedTraceResult = Result<RecordedPhysicalTrace, String>;
pub(super) fn owner_forward_moved_finish(
    recording: OwnerMovedObserverAlias,
    context: GemmRouteIdentity,
) -> OwnerMovedTraceResult {
    let renamed = recording;
    renamed.finish(context)
}
"#,
        ),
        (
            "untyped closure eager finalizer with type and result aliases",
            r#"
type OwnerClosureObserverAlias = RecordingPhysicalObserver;
type OwnerClosureTraceResult = Result<RecordedPhysicalTrace, String>;
pub(super) fn owner_forward_closure_finish(
    recording: OwnerClosureObserverAlias,
    context: GemmRouteIdentity,
) -> OwnerClosureTraceResult {
    (|renamed| renamed.finish(context))(recording)
}
"#,
        ),
        (
            "UFCS eager finalizer through type and result aliases",
            r#"
type OwnerObserverAlias = RecordingPhysicalObserver;
type OwnerUfcsTraceResult = Result<RecordedPhysicalTrace, String>;
pub(super) fn owner_forward_ufcs_finish(
    recording: OwnerObserverAlias,
    context: GemmRouteIdentity,
) -> OwnerUfcsTraceResult {
    <OwnerObserverAlias>::finish(recording, context)
}
"#,
        ),
        (
            "direct observation resolver",
            r#"
pub(super) fn owner_forward_direct_resolve(
    recording: &RecordingPhysicalObserver,
    observation: PhysicalLaunchObservation,
    config: cudarc::driver::LaunchConfig,
) -> Result<ResolvedPhysicalKernelLaunch, String> {
    observation.resolve(recording, config)
}
"#,
        ),
        (
            "direct private recorder mutation",
            r#"
pub(super) fn owner_forward_private_record(
    recorder: &mut PhysicalTraceRecorder,
    launch: ResolvedPhysicalKernelLaunch,
) -> Result<(), String> {
    recorder.record(launch)
}
"#,
        ),
        (
            "direct private recorder invalidation",
            r#"
pub(super) fn owner_forward_private_invalidation(recorder: &mut PhysicalTraceRecorder) {
    recorder.invalidate_enqueue();
}
"#,
        ),
        (
            "direct private recorder finalization with result alias",
            r#"
type OwnerRecorderTraceResult = Result<RecordedPhysicalTrace, String>;
pub(super) fn owner_forward_private_recorder_finish(
    recorder: PhysicalTraceRecorder,
    context: GemmRouteIdentity,
    binding: PhysicalGraphBinding,
) -> OwnerRecorderTraceResult {
    recorder.finish(context, binding)
}
"#,
        ),
        (
            "direct sealed recorder mutation",
            r#"
pub(super) fn owner_forward_sealed_record(
    recording: &mut RecordingPhysicalObserver,
    authority: &mut physical_observer_private::Authority,
    launch: ResolvedPhysicalKernelLaunch,
) -> Result<(), String> {
    physical_observer_private::Sealed::record_before_enqueue(
        recording,
        authority,
        launch,
    )
}
"#,
        ),
        (
            "direct sealed recorder invalidation",
            r#"
pub(super) fn owner_forward_sealed_invalidation(
    recording: &mut RecordingPhysicalObserver,
    authority: &mut physical_observer_private::Authority,
) {
    physical_observer_private::Sealed::invalidate_enqueue(recording, authority);
}
"#,
        ),
        (
            "owner inherent capture wrapper",
            r#"
impl RecordingPhysicalObserver {
    pub(super) fn owner_capture_wrapper(
        self,
        ctx: &GpuCtx,
        manifest: &PreparedPhysicalCaptureManifest,
    ) -> Result<CapturedPhysicalGraphPlan, String> {
        self.finish_capture(ctx, manifest)
    }
}
"#,
        ),
        (
            "owner inherent recorder wrapper",
            r#"
impl RecordingPhysicalObserver {
    pub(super) fn owner_record_wrapper(
        &mut self,
        launch: ResolvedPhysicalKernelLaunch,
    ) -> Result<(), String> {
        self.recorder.record(launch)
    }
}
"#,
        ),
    ];
    let accepted_forwarders = forwarding_mutations
        .iter()
        .filter_map(|(label, mutation)| {
            let identity = format!("{IDENTITY_SOURCE}\n{mutation}");
            validate_physical_trace_ownership_boundary(&identity, &sources)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();

    let pointer_mutations = [
        (
            "direct eager finalizer constant with result alias",
            r#"
type OwnerEagerPointerTraceResult = Result<RecordedPhysicalTrace, String>;
type OwnerEagerFinalizer = fn(
    RecordingPhysicalObserver,
    GemmRouteIdentity,
) -> OwnerEagerPointerTraceResult;
pub(super) const OWNER_EAGER_FINALIZER: OwnerEagerFinalizer =
    RecordingPhysicalObserver::finish;
"#,
        ),
        (
            "renamed eager finalizer static with result alias",
            r#"
type OwnerRenamedEagerTraceResult = Result<RecordedPhysicalTrace, String>;
type OwnerRenamedEagerFinalizer = fn(
    RecordingPhysicalObserver,
    GemmRouteIdentity,
) -> OwnerRenamedEagerTraceResult;
const RENAMED_OWNER_EAGER_FINALIZER: OwnerRenamedEagerFinalizer =
    RecordingPhysicalObserver::finish;
pub(super) static OWNER_RENAMED_EAGER_FINALIZER: OwnerRenamedEagerFinalizer =
    RENAMED_OWNER_EAGER_FINALIZER;
"#,
        ),
        (
            "direct capture finalizer constant",
            r#"
type OwnerCaptureFinalizer = fn(
    RecordingPhysicalObserver,
    &GpuCtx,
    &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String>;
pub(super) const OWNER_CAPTURE_FINALIZER: OwnerCaptureFinalizer =
    RecordingPhysicalObserver::finish_capture;
"#,
        ),
        (
            "renamed capture finalizer static",
            r#"
type OwnerRenamedCaptureFinalizer = fn(
    RecordingPhysicalObserver,
    &GpuCtx,
    &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String>;
const RENAMED_OWNER_CAPTURE_FINALIZER: OwnerRenamedCaptureFinalizer =
    RecordingPhysicalObserver::finish_capture;
pub(super) static OWNER_RENAMED_CAPTURE_FINALIZER: OwnerRenamedCaptureFinalizer =
    RENAMED_OWNER_CAPTURE_FINALIZER;
"#,
        ),
        (
            "capture finalizer closure constant",
            r#"
type OwnerCaptureClosure = fn(
    RecordingPhysicalObserver,
    &GpuCtx,
    &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String>;
pub(super) const OWNER_CAPTURE_CLOSURE: OwnerCaptureClosure =
    |renamed, ctx, manifest| renamed.finish_capture(ctx, manifest);
"#,
        ),
        (
            "capture finalizer closure static",
            r#"
type OwnerStaticCaptureClosure = fn(
    RecordingPhysicalObserver,
    &GpuCtx,
    &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String>;
pub(super) static OWNER_STATIC_CAPTURE_CLOSURE: OwnerStaticCaptureClosure =
    |renamed, ctx, manifest| renamed.finish_capture(ctx, manifest);
"#,
        ),
        (
            "direct observation resolver constant",
            r#"
type OwnerObservationResolver = fn(
    PhysicalLaunchObservation,
    &RecordingPhysicalObserver,
    cudarc::driver::LaunchConfig,
) -> Result<ResolvedPhysicalKernelLaunch, String>;
pub(super) const OWNER_OBSERVATION_RESOLVER: OwnerObservationResolver =
    PhysicalLaunchObservation::resolve::<RecordingPhysicalObserver>;
"#,
        ),
        (
            "renamed observation resolver static",
            r#"
type OwnerRenamedObservationResolver = fn(
    PhysicalLaunchObservation,
    &RecordingPhysicalObserver,
    cudarc::driver::LaunchConfig,
) -> Result<ResolvedPhysicalKernelLaunch, String>;
const RENAMED_OWNER_OBSERVATION_RESOLVER: OwnerRenamedObservationResolver =
    PhysicalLaunchObservation::resolve::<RecordingPhysicalObserver>;
pub(super) static OWNER_RENAMED_OBSERVATION_RESOLVER: OwnerRenamedObservationResolver =
    RENAMED_OWNER_OBSERVATION_RESOLVER;
"#,
        ),
        (
            "direct recorder mutation constant",
            r#"
type OwnerRecorderMutation =
    fn(&mut PhysicalTraceRecorder, ResolvedPhysicalKernelLaunch) -> Result<(), String>;
pub(super) const OWNER_RECORDER_MUTATION: OwnerRecorderMutation = PhysicalTraceRecorder::record;
"#,
        ),
        (
            "renamed recorder invalidation static",
            r#"
type OwnerRecorderInvalidation = fn(&mut PhysicalTraceRecorder);
const RENAMED_OWNER_INVALIDATION: OwnerRecorderInvalidation =
    PhysicalTraceRecorder::invalidate_enqueue;
pub(super) static OWNER_RECORDER_INVALIDATION: OwnerRecorderInvalidation =
    RENAMED_OWNER_INVALIDATION;
"#,
        ),
        (
            "direct recorder finalizer constant with result alias",
            r#"
type OwnerPointerTraceResult = Result<RecordedPhysicalTrace, String>;
type OwnerRecorderFinalizer = fn(
    PhysicalTraceRecorder,
    GemmRouteIdentity,
    PhysicalGraphBinding,
) -> OwnerPointerTraceResult;
pub(super) const OWNER_RECORDER_FINALIZER: OwnerRecorderFinalizer =
    PhysicalTraceRecorder::finish;
"#,
        ),
        (
            "renamed recorder finalizer static",
            r#"
type OwnerRenamedRecorderFinalizer = fn(
    PhysicalTraceRecorder,
    GemmRouteIdentity,
    PhysicalGraphBinding,
) -> Result<RecordedPhysicalTrace, String>;
const RENAMED_OWNER_RECORDER_FINALIZER: OwnerRenamedRecorderFinalizer =
    PhysicalTraceRecorder::finish;
pub(super) static OWNER_RENAMED_RECORDER_FINALIZER: OwnerRenamedRecorderFinalizer =
    RENAMED_OWNER_RECORDER_FINALIZER;
"#,
        ),
        (
            "direct sealed recorder mutation constant",
            r#"
type OwnerSealedMutation = fn(
    &mut RecordingPhysicalObserver,
    &mut physical_observer_private::Authority,
    ResolvedPhysicalKernelLaunch,
) -> Result<(), String>;
pub(super) const OWNER_SEALED_MUTATION: OwnerSealedMutation =
    <RecordingPhysicalObserver as physical_observer_private::Sealed>::record_before_enqueue;
"#,
        ),
        (
            "renamed sealed invalidation static",
            r#"
type OwnerSealedInvalidation = fn(
    &mut RecordingPhysicalObserver,
    &mut physical_observer_private::Authority,
);
const RENAMED_SEALED_INVALIDATION: OwnerSealedInvalidation =
    <RecordingPhysicalObserver as physical_observer_private::Sealed>::invalidate_enqueue;
pub(super) static OWNER_SEALED_INVALIDATION: OwnerSealedInvalidation =
    RENAMED_SEALED_INVALIDATION;
"#,
        ),
    ];
    let accepted_pointers = pointer_mutations
        .iter()
        .filter_map(|(label, mutation)| {
            let identity = format!("{IDENTITY_SOURCE}\n{mutation}");
            validate_physical_trace_ownership_boundary(&identity, &sources)
                .is_ok()
                .then_some(*label)
        })
        .collect::<Vec<_>>();

    assert!(
        accepted_forwarders.is_empty() && accepted_pointers.is_empty(),
        "private owner authorities were accepted: forwarders={accepted_forwarders:?}, visible_function_pointers={accepted_pointers:?}"
    );

    for harmless in [
        "pub(super) const OWNER_HARMLESS_LIMIT: usize = 1;",
        "pub(super) static OWNER_HARMLESS_LABEL: &str = \"physical graph\";",
    ] {
        let identity = format!("{IDENTITY_SOURCE}\n{harmless}");
        let result = validate_physical_trace_ownership_boundary(&identity, &sources);
        assert!(
            result.is_ok(),
            "a visible non-authority owner item was rejected: {harmless}: {result:?}"
        );
    }
}

#[test]
fn f32_triad_capture_path_is_prepared_and_records_scalar_fallback_nodes() {
    assert_contains_all(
        LAUNCH_SOURCE,
        &[
            "pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad(",
            "request: F32TriadRequest",
            "operands: F32TriadOperands",
            "Result<PreparedF32TriadLaunch, String>",
            "pub(in crate::mamba_ssm::gpu) unsafe fn launch_prepared_f32_triad(",
            "record_resolved_gemm_route",
        ],
        "prepared f32 triad launch path",
    );
    let preparation = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad",
    ));
    assert_contains_all(
        &preparation,
        &[
            "resolve_f32_triad_auto_with_operands(",
            "request,",
            "operands,",
            "f32_triad_availability()",
        ],
        "operand-aware production TF32 preparation",
    );
    assert!(
        !preparation.contains("resolve_f32_triad_auto("),
        "production preparation must not discard concrete operand evidence"
    );
    let resolver_calls = matching_call_ranges(&preparation, "resolve_f32_triad_auto_with_operands")
        .expect("scan production operand-aware resolver calls");
    assert_eq!(
        resolver_calls.len(),
        1,
        "production preparation must resolve AUTO exactly once"
    );
    let resolver_call = &preparation[resolver_calls[0].0..resolver_calls[0].1];
    let resolver_arguments = top_level_arguments(resolver_call)
        .expect("parse production operand-aware resolver arguments")
        .into_iter()
        .map(|argument| {
            source_mask(argument)
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        resolver_arguments,
        [
            "ctx.f32_triad_policy()",
            "request",
            "operands",
            "ctx.kernels.f32_triad_availability()",
        ],
        "production preparation changed the exact operand-aware resolver call"
    );
    let launch = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) unsafe fn launch_prepared_f32_triad",
    ));
    for forbidden in [
        "capture_status",
        "compile",
        "alloc",
        "encode",
        "fallback",
        "Instant",
        "CudaEvent",
    ] {
        assert!(
            !launch.contains(forbidden),
            "prepared capture launch contains forbidden side effect {forbidden}"
        );
    }
}

#[test]
fn production_f32_wrappers_preserve_the_complete_auto_operand_contract() {
    for (marker, required) in [
        (
            "fn launch_cached_f32_forward_ptrs_selected",
            &[
                "output,",
                "a: x_ptr",
                "b: w_ptr",
                "bias: (bias_ptr != 0).then_some(bias_ptr)",
                "alpha: 1.0",
                "beta: 0.0",
            ][..],
        ),
        (
            "fn launch_cached_f32_backward_dw_selected",
            &[
                "output: dw_ptr",
                "x_saved.raw_ptr(&ctx.stream)",
                "dy.raw_ptr(&ctx.stream)",
                "bias: None",
                "alpha: 1.0",
                "beta: 1.0",
            ][..],
        ),
        (
            "fn launch_cached_f32_backward_dx_ptrs_selected",
            &[
                "output: dx,",
                "a: if reduction_is_zero { 0 } else { dy }",
                "b: if reduction_is_zero { 0 } else { w_ptr }",
                "bias: None",
                "alpha: 1.0",
                "beta: 0.0",
            ][..],
        ),
    ] {
        let scope = source_mask(braced_scope_after(LAUNCH_SOURCE, marker));
        assert_contains_all(&scope, required, marker);
        let calls = matching_call_ranges(&scope, "launch_cached_f32_triad")
            .expect("scan production cached F32 launch calls");
        assert_eq!(calls.len(), 1, "{marker} must enqueue exactly once");
        let call = &scope[calls[0].0..calls[0].1];
        let arguments = top_level_arguments(call).expect("parse cached F32 launch arguments");
        assert_eq!(
            arguments[..4]
                .iter()
                .map(|argument| argument.trim())
                .collect::<Vec<_>>(),
            ["ctx", "selection", "request", "operands"],
            "{marker} must pass its exact request and operands to the cache"
        );
    }
}

#[test]
fn tf32_sources_export_the_exact_planned_symbol_inventories() {
    let inventories = [
        (SM80_SOURCE, "_sm80_mma_tf32_", expected_sm80_symbols(), 18),
        (
            SM90A_SOURCE,
            "_sm90a_wgmma_tf32_",
            expected_sm90a_symbols(),
            6,
        ),
        (
            SM100_SOURCE,
            "_sm100_tcgen_tf32_",
            expected_sm100_symbols(),
            36,
        ),
        (
            SM120_SOURCE,
            "_sm120_tma_mma_tf32_",
            expected_sm120_symbols(),
            18,
        ),
    ];

    for (source, family, expected, count) in inventories {
        let source = source_mask(source);
        // The split-K kernels share the source but belong to the extension
        // contract, which has its own inventory.
        let actual: BTreeSet<_> = op_kernel_identifiers(&source)
            .into_iter()
            .filter(|symbol| symbol.contains(family) && !symbol.contains("_splitk"))
            .collect();
        assert_eq!(expected.len(), count);
        assert_eq!(actual, expected, "wrong public inventory for {family}");
    }
}

#[test]
fn rust_contract_and_module_loader_own_the_same_exact_tf32_inventories() {
    let inventory_test = braced_scope_after(
        MODULE_SOURCE,
        "fn tf32_contract_and_loader_inventories_match_cuda_exports",
    );
    assert_code_contains_all(
        inventory_test,
        &[
            "ModuleKind::TriadSm80",
            "ModuleKind::TriadSm90a",
            "ModuleKind::TriadSm100",
            "ModuleKind::TriadSm120",
            "tf32_route_specs",
            "tf32_module_symbols",
            "load_function",
            "assert_eq!",
            "18",
            "6",
            "36",
        ],
        "direct contract/module/CUDA inventory behavior",
    );

    let validator = braced_scope_after(MODULE_SOURCE, "validate_tf32_ptx_inventory");
    assert_code_contains_all(
        validator,
        &["kernel_spec", "BTreeSet", "Err("],
        "compiled TF32 PTX inventory validator",
    );
    let module_code = source_mask(MODULE_SOURCE);
    assert!(
        module_code.contains("validate_tf32_ptx_inventory(module_kind, extensions, ptx")
            || module_code
                .contains("validate_tf32_ptx_inventory(request.module_kind, extensions, ptx"),
        "module admission must call the compiled-PTX inventory validator"
    );
    let module_tests = braced_scope_after(MODULE_SOURCE, "#[cfg(test)]\nmod tests");
    let partial = braced_scope_after(
        module_tests,
        "fn tf32_compiled_inventory_rejects_every_partial_module",
    );
    assert_code_contains_all(
        partial,
        &[
            "ModuleKind::TriadSm80",
            "ModuleKind::TriadSm90a",
            "ModuleKind::TriadSm100",
            "ModuleKind::TriadSm120",
            "validate_tf32_ptx_inventory",
            "remove",
            "is_err()",
        ],
        "behavioral partial-module rejection",
    );
}

#[test]
fn cuda_and_rust_kernel_parameter_layouts_match() {
    let common_code = compact_code(COMMON_SOURCE);
    for primitive in ["float", "int"] {
        assert!(
            common_code.contains(&format!("static_assert(sizeof({primitive})==4")),
            "shared CUDA parameter ABI must freeze {primitive} at four bytes"
        );
    }

    let portable_fields = [
        ("float", "alpha"),
        ("float", "beta"),
        ("int", "m"),
        ("int", "k"),
        ("int", "n"),
        ("int", "lda"),
        ("int", "ldb"),
        ("int", "ldc"),
    ];
    let specialized_fields = [
        ("int", "a_x"),
        ("int", "a_y"),
        ("int", "b_x"),
        ("int", "b_y"),
        ("float", "alpha"),
        ("float", "beta"),
        ("int", "m"),
        ("int", "k"),
        ("int", "n"),
        ("int", "ldc"),
    ];
    assert_cuda_bundle_layout(
        SCALAR_SOURCE,
        "SgbZeroReductionParams",
        &portable_fields,
        32,
    );
    assert_cuda_bundle_layout(SM80_SOURCE, "Sm80Tf32KernelParams", &portable_fields, 32);
    for (source, name) in [
        (SM90A_SOURCE, "Sm90aTf32KernelParams"),
        (SM100_SOURCE, "Sm100KernelParams"),
        (SM120_SOURCE, "Sm120KernelParams"),
    ] {
        assert_cuda_bundle_layout(source, name, &specialized_fields, 40);
    }

    let rust_code = compact_code(LAUNCH_SOURCE);
    for (name, bytes, fields) in [
        ("SgbZeroReductionParams", 32, portable_fields.as_slice()),
        ("Sm80Tf32KernelParams", 32, portable_fields.as_slice()),
        ("Sm90aTf32KernelParams", 40, specialized_fields.as_slice()),
        ("Sm100KernelParams", 40, specialized_fields.as_slice()),
        ("Sm120KernelParams", 40, specialized_fields.as_slice()),
    ] {
        assert!(
            rust_code.contains(&format!("#[repr(C)]struct{name}")),
            "Rust launch ABI must define repr(C) {name}"
        );
        let structure = compact_code(braced_scope_after(LAUNCH_SOURCE, &format!("struct {name}")));
        for (field_type, field) in fields {
            let rust_type = if *field_type == "float" { "f32" } else { "i32" };
            assert!(
                structure.contains(&format!("{field}:{rust_type}")),
                "Rust {name} field {field} must be {rust_type}"
            );
        }
        assert!(
            rust_code.contains(&format!("size_of::<{name}>(),{bytes}"))
                && rust_code.contains(&format!("align_of::<{name}>(),4")),
            "Rust {name} must compile-time assert size and alignment"
        );
        for (index, (_, field)) in fields.iter().enumerate() {
            assert!(
                rust_code.contains(&format!("offset_of!({name},{field}),{}", index * 4)),
                "Rust {name} must assert offset {} for {field}",
                index * 4
            );
        }
    }

    for (source, family) in [
        (SM80_SOURCE, "_sm80_mma_tf32_"),
        (SM90A_SOURCE, "_sm90a_wgmma_tf32_"),
        (SM100_SOURCE, "_sm100_tcgen_tf32_"),
        (SM120_SOURCE, "_sm120_tma_mma_tf32_"),
    ] {
        let code = source_mask(source);
        assert!(
            code.contains("TF32_ASSERT_KERNEL_SIGNATURE"),
            "{family} must compile decltype(&symbol) signature assertions"
        );
        let symbols = op_kernel_identifiers(&code)
            .into_iter()
            .filter(|symbol| symbol.contains(family) && !symbol.contains("_splitk"));
        for symbol in symbols {
            assert!(
                code.contains(&format!("TF32_ASSERT_KERNEL_SIGNATURE({symbol}")),
                "{symbol} is missing its C++ signature assertion"
            );
        }
    }
    assert_code_contains_all(
        braced_scope_after(LAUNCH_SOURCE, "fn tf32_kernel_param_abi_matches_cuda"),
        &["size_of", "align_of", "offset_of"],
        "production Rust/CUDA ABI unit test",
    );
}

#[test]
fn compiled_tf32_ptx_exports_exact_five_parameter_abis() {
    let families = [
        (
            "SM80",
            compile_tf32_ptx(tf32_cuda_blob(SM80_SOURCE, true), "sm_89"),
            expected_sm80_symbols(),
            32,
            "sm_89",
            SM80_SOURCE,
        ),
        (
            "SM90a",
            compile_tf32_ptx(tf32_cuda_blob(SM90A_SOURCE, false), "sm_90a"),
            expected_sm90a_symbols(),
            40,
            "sm_90a",
            SM90A_SOURCE,
        ),
        (
            "SM100",
            compile_tf32_ptx(tf32_cuda_blob(SM100_SOURCE, false), "compute_100a"),
            expected_sm100_symbols(),
            40,
            "sm_100a",
            SM100_SOURCE,
        ),
        (
            "SM120",
            compile_tf32_ptx(tf32_cuda_blob(SM120_SOURCE, false), "compute_120"),
            expected_sm120_symbols(),
            40,
            "sm_120",
            SM120_SOURCE,
        ),
    ];

    let tensor_map_alignment = if loaded_nvrtc_version().0 >= 13 {
        128
    } else {
        64
    };
    for (label, ptx, expected, bundle_bytes, sass_target, source) in families {
        let actual = ptx_entry_symbols(&ptx, "_tf32_")
            .into_iter()
            .filter(|symbol| !symbol.contains("_splitk"))
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected, "{label} compiled PTX export inventory");
        for symbol in &expected {
            assert_eq!(
                ptx.matches(&format!(".entry {symbol}(")).count(),
                1,
                "{label} must compile one entry for {symbol}"
            );
            let entry = ptx_entry(&ptx, symbol);
            let parameters = ptx_parameters(entry, symbol);
            assert_eq!(
                parse_ptx_parameters(parameters),
                expected_ptx_parameters(symbol, bundle_bytes, tensor_map_alignment),
                "{label} {symbol} PTX parameter types, widths, or alignment drifted: {parameters}"
            );
            assert_k0_cfg_dominates_entry(entry, symbol, bundle_bytes);
            assert!(
                !entry.contains("%ctaid.y") && !entry.contains("%ctaid.z"),
                "{label} {symbol} must map one CTA to each output tile"
            );
            assert!(
                !entry.contains(".local"),
                "{label} {symbol} PTX must not declare local memory"
            );
            if symbol == "tn_sm80_mma_tf32_m128n64_bk32_s2" {
                assert!(
                    entry.contains(".maxntid 256, 1, 1"),
                    "{label} {symbol} must retain its 256-thread CTA"
                );
                assert!(
                    entry.contains(".minnctapersm 1"),
                    "{label} {symbol} must retain its portable one-CTA launch bound"
                );
            }
            for forbidden in [
                "atom.",
                "red.",
                "atom::",
                "red::",
                "multicast",
                "cta_group::2",
                "fma.rz",
                "fma.rm",
                "fma.rp",
                "fma.rn.ftz",
            ] {
                let present = if matches!(forbidden, "atom." | "red." | "atom::" | "red::") {
                    contains_opcode_prefix(entry, forbidden)
                } else {
                    entry.contains(forbidden)
                };
                assert!(
                    !present,
                    "{label} {symbol} compiled forbidden PTX {forbidden}"
                );
            }
            assert!(
                !contains_float_mad_opcode(entry),
                "{label} {symbol} compiled forbidden floating-point mad"
            );
            if symbol.starts_with("nn_") {
                assert!(
                    entry.contains("mul.rn.f32") && entry.contains("fma.rn.f32"),
                    "{label} {symbol} NN epilogue must keep alpha multiply and beta FMA"
                );
            } else if symbol.starts_with("tn_") {
                assert!(
                    entry.contains("fma.rn.f32") && !entry.contains("mul.rn.f32"),
                    "{label} {symbol} TN epilogue must use only its RN alpha FMA"
                );
            } else if symbol.starts_with("nt_") {
                assert!(
                    entry.contains("mul.rn.f32") && !entry.contains("fma.rn.f32"),
                    "{label} {symbol} NT epilogue must use only its RN alpha multiply"
                );
            }
        }

        assert!(
            ptx.contains("fma.rn.f32") && ptx.contains("mul.rn.f32"),
            "{label} compiled epilogues must preserve explicit RN FMA and multiply"
        );
        let code = source_mask(source);
        for forbidden in [
            "split_k",
            "stream_k",
            "work_steal",
            "persistent_queue",
            "blockIdx.y",
            "blockIdx.z",
        ] {
            if label == "SM80" && forbidden == "blockIdx.y" {
                assert_eq!(
                    code.matches(forbidden).count(),
                    2,
                    "SM80 source may use blockIdx.y only for its separately qualified split-K candidates"
                );
                continue;
            }
            if label == "SM80" && forbidden == "blockIdx.z" {
                assert_eq!(
                    code.matches(forbidden).count(),
                    1,
                    "SM80 source may use blockIdx.z only for its separately qualified split-K candidates"
                );
                continue;
            }
            assert!(
                !code.contains(forbidden),
                "{label} source contains forbidden ownership mechanism {forbidden}"
            );
        }
        let (resources, object_resources, sass, sass_cfg) =
            assemble_and_disassemble_tf32(&ptx, sass_target, label);
        assert_per_entry_zero_resources(&resources, &expected, label);
        assert_cuobjdump_zero_resources(&object_resources, &expected, label);
        for symbol in &expected {
            assert_sass_entry_contract(&sass, symbol, label);
            assert_sass_cfg_corroboration(&sass_cfg, &sass, source, symbol, label);
            assert!(
                !sass_entry(&sass, symbol).contains("LDGSTS.CTA_GROUP_2"),
                "{label}/{symbol} must not use multi-CTA transport"
            );
        }
    }
}

#[test]
fn compiled_sm80_tf32_splitk_candidates_freeze_abi_and_ordered_reduction() {
    let ptx = compile_tf32_ptx(tf32_cuda_blob(SM80_SOURCE, true), "sm_89");
    let fused_symbols = [
        "nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4",
        "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
        "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3",
        "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
        "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
        "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
    ];
    let expected = fused_symbols
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        ptx_entry_symbols(&ptx, "_tf32_splitk"),
        expected,
        "SM80 split-K compiled PTX export inventory"
    );

    for symbol in fused_symbols {
        assert_eq!(
            ptx.matches(&format!(".entry {symbol}(")).count(),
            1,
            "SM80 split-K must compile one entry for {symbol}"
        );
        let entry = ptx_entry(&ptx, symbol);
        assert_eq!(
            parse_ptx_parameters(ptx_parameters(entry, symbol)),
            expected_ptx_parameters(symbol, 32, 128),
            "SM80 split-K {symbol} seven-parameter ABI drifted"
        );
        assert!(
            !entry.contains(".local"),
            "SM80 split-K {symbol} PTX must not declare local memory"
        );
        for forbidden in ["red.", "atom::", "red::", "redux.", "atom.global.add.f32"] {
            assert!(
                !contains_opcode_prefix(entry, forbidden),
                "SM80 split-K {symbol} compiled unordered opcode {forbidden}"
            );
        }
        let fused = ptx_entry(&ptx, symbol);
        let minimum_blocks = if symbol.ends_with("_m32n32_bk32_s4") {
            ".minnctapersm 2"
        } else {
            ".minnctapersm 3"
        };
        assert_contains_all(
            fused,
            &[
                ".maxntid 128, 1, 1",
                minimum_blocks,
                "%ctaid.z",
                "cvt.rna.tf32.f32",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "membar.gl",
                "atom.global.inc.u32",
                "st.global.v2.f32",
                "add.rn.f32",
                "mul.rn.f32",
                "st.global.cg.f32",
                "st.global.cg.v2.f32",
                "ld.global.cg.f32",
                "ld.global.cg.v2.f32",
            ],
            "SM80 split-K fused compiled PTX",
        );
        if symbol.starts_with("nn_") {
            assert!(
                fused.contains("fma.rn.f32"),
                "SM80 split-K NN epilogue must retain beta FMA"
            );
        } else {
            assert!(
                !fused.contains("fma.rn.f32"),
                "SM80 split-K NT epilogue must not read or combine old output"
            );
        }
        assert!(
            !contains_float_mad_opcode(fused),
            "SM80 split-K fused compiled forbidden floating-point mad"
        );
        let atomic_limits = fused
            .lines()
            .filter_map(ptx_instruction)
            .filter(|(opcode, _)| *opcode == "atom.global.inc.u32")
            .map(|(_, operands)| ptx_operands(operands))
            .map(|operands| operands[2].to_owned())
            .collect::<Vec<_>>();
        let expected_limit = if symbol.contains("_splitk2_") {
            "1"
        } else if symbol.contains("_splitk4_") {
            "3"
        } else {
            "7"
        };
        assert_eq!(
            atomic_limits,
            [expected_limit],
            "SM80 split-K completion threshold drifted for {symbol}"
        );
        for forbidden in ["div.", "rem."] {
            assert!(
                !contains_opcode_prefix(fused, forbidden),
                "SM80 split-K fused kernel must not use runtime {forbidden}: {:?}",
                opcode_context(fused, forbidden)
            );
        }
        let partition_add_group = if symbol.contains("_splitk2_") {
            4
        } else if symbol.starts_with("nn_") {
            8
        } else if symbol.contains("_splitk4_") {
            6
        } else {
            14
        };
        assert_eq!(
            fused.matches("add.rn.f32").count() % partition_add_group,
            0,
            "SM80 split-K fused reducer lost its complete RN partition-add groups"
        );
    }
}

#[test]
fn portable_sm80_tf32_nt_splitk_candidates_freeze_cuda_contract() {
    let candidates = [
        (
            "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3",
            "tf32_splitk_fused_kernel<SgbTf32Nt, 16, 32, 3, 4>",
            "__launch_bounds__(128, 3)",
        ),
        (
            "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
            "tf32_splitk_fused_kernel<SgbTf32Nt, 16, 32, 4, 4>",
            "__launch_bounds__(128, 3)",
        ),
        (
            "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
            "tf32_splitk_fused_kernel<SgbTf32Nt, 32, 32, 3, 8>",
            "__launch_bounds__(128, 3)",
        ),
        (
            "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
            "tf32_splitk_fused_kernel<SgbTf32Nt, 32, 32, 4, 8>",
            "__launch_bounds__(128, 2)",
        ),
    ];
    for (symbol, specialization, launch_bounds) in candidates {
        assert!(
            SM80_SOURCE.contains(&format!("{launch_bounds}\nvoid {symbol}(")),
            "{symbol} launch bounds drifted"
        );
        assert!(
            SM80_SOURCE.contains(specialization),
            "{symbol} must bind its fixed Op/tile/stage/partition specialization"
        );
        assert!(
            SM80_SOURCE.contains(&format!("TF32_ASSERT_SPLITK_KERNEL_SIGNATURE({symbol})")),
            "{symbol} is missing its seven-parameter signature assertion"
        );
    }

    assert_contains_all(
        SM80_SOURCE,
        &[
            "sizeof(SgbTf32Storage<SgbTf32Nt, 16, 32, 3>) == 20736",
            "sizeof(SgbTf32Storage<SgbTf32Nt, 32, 32, 3>) == 27648",
            "sizeof(SgbTf32Storage<SgbTf32Nt, 32, 32, 4>) == 36864",
        ],
        "portable NT split-K shared-storage ABI",
    );

    let fused = compact_code(&source_mask(braced_scope_after(
        SM80_SOURCE,
        "tf32_splitk_fused_kernel",
    )));
    assert_contains_all(
        &fused,
        &[
            "SgbTf32Storage<Op,BM,BN,Stages>",
            "tf32_rows<Op>(params)",
            "tf32_columns<Op>(params)",
            "tf32_reduction<Op>(params)",
            "partial_stride=(longlong)rows*columns",
            "atomicInc(counters+tile,Partitions-1U)",
            "ifconstexpr(Partitions==8)",
            "sum0=__fadd_rn(p0.x,p1.x)",
            "sum0=__fadd_rn(sum0,p2.x)",
            "sum0=__fadd_rn(sum0,p3.x)",
            "sum0=__fadd_rn(sum0,p4.x)",
            "sum0=__fadd_rn(sum0,p5.x)",
            "sum0=__fadd_rn(sum0,p6.x)",
            "sum0=__fadd_rn(sum0,p7.x)",
        ],
        "portable NT split-K fused kernel",
    );
    for forbidden in ["atomicAdd", "atomicExch", "atomicCAS", "%"] {
        assert!(
            !fused.contains(forbidden),
            "portable NT split-K source contains forbidden {forbidden}"
        );
    }
}

#[test]
fn portable_sm80_tf32_tn_splitk_candidates_freeze_cuda_contract() {
    let candidates = [
        (
            "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s2",
            "tf32_tn_splitk_fused_kernel<SgbTf32Tn, 64, 64, 2, 8>",
            "__launch_bounds__(128, 2)",
        ),
        (
            "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3",
            "tf32_tn_splitk_fused_kernel<SgbTf32Tn, 64, 64, 3, 8>",
            "__launch_bounds__(128, 1)",
        ),
        (
            "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
            "tf32_tn_splitk_fused_kernel<SgbTf32Tn, 32, 32, 3, 8>",
            "__launch_bounds__(128, 3)",
        ),
        (
            "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
            "tf32_tn_splitk_fused_kernel<SgbTf32Tn, 32, 32, 4, 8>",
            "__launch_bounds__(128, 2)",
        ),
    ];
    for (symbol, specialization, launch_bounds) in candidates {
        assert!(
            SM80_TN_SPLITK_SOURCE.contains(&format!("{launch_bounds}\nvoid {symbol}(")),
            "{symbol} launch bounds drifted"
        );
        assert!(
            SM80_TN_SPLITK_SOURCE.contains(specialization),
            "{symbol} must bind its fixed tile/stage/partition specialization"
        );
        assert!(
            SM80_TN_SPLITK_SOURCE
                .contains(&format!("TF32_ASSERT_SPLITK_KERNEL_SIGNATURE({symbol})")),
            "{symbol} is missing its seven-parameter signature assertion"
        );
        assert!(
            !SM80_SOURCE.contains(symbol),
            "{symbol} must live in the extension fragment, not the shared source"
        );
    }

    assert_contains_all(
        SM80_SOURCE,
        &[
            "sizeof(SgbTf32Storage<SgbTf32Tn, 64, 64, 2>) == 36864",
            "sizeof(SgbTf32Storage<SgbTf32Tn, 64, 64, 3>) == 55296",
        ],
        "shared TN split-K storage ABI",
    );
    assert_contains_all(
        SM80_TN_SPLITK_SOURCE,
        &[
            "sizeof(SgbTf32Storage<SgbTf32Tn, 32, 32, 3>) == 30720",
            "sizeof(SgbTf32Storage<SgbTf32Tn, 32, 32, 4>) == 40960",
        ],
        "extension TN split-K storage ABI",
    );

    // The TN stage loads A k-major on the full-tile fast path; the fixup
    // folds beta into the fused output like NN does, without a bias term.
    let stage = compact_code(&source_mask(braced_scope_after(
        SM80_TN_SPLITK_SOURCE,
        "tf32_tn_splitk_stage_async",
    )));
    assert_contains_all(
        &stage,
        &[
            "static_assert(Op==SgbTf32Tn,",
            "constexprintRowChunks=BM/4;",
            "+(longlong)(reduction_base+reduction)*problem.params.lda+problem.tile_row+row;",
        ],
        "TN split-K k-major stage",
    );
    let fused = compact_code(&source_mask(braced_scope_after(
        SM80_TN_SPLITK_SOURCE,
        "tf32_tn_splitk_fused_kernel",
    )));
    assert_contains_all(
        &fused,
        &[
            "static_assert(Op==SgbTf32Tn,",
            "assert(bias==nullptr);",
            "tf32_tn_splitk_async_mainloop<",
            "if(params.beta!=0.0f){value0=__fmaf_rn(params.beta,destination[0],value0);}",
            "if(params.beta!=0.0f){value1=__fmaf_rn(params.beta,destination[1],value1);}",
            "atomicInc(counters+tile,Partitions-1U)",
        ],
        "TN split-K fused kernel",
    );
    for forbidden in ["atomicAdd", "atomicExch", "atomicCAS", "%"] {
        assert!(
            !fused.contains(forbidden),
            "TN split-K source contains forbidden {forbidden}"
        );
    }
}

#[test]
fn portable_sm80_tf32_uses_rna_m16n8k8_and_frozen_shared_bank_maps() {
    assert_contains_all(
        SM80_SOURCE,
        &["tf32_rna", "tf32_mma_m16n8k8", "bk32"],
        "portable SM80 TF32 mainloop",
    );
    let code = source_mask(SM80_SOURCE);
    for forbidden in [
        "cvt.rn.tf32.f32",
        "cvt.rz.tf32.f32",
        "cvt.rna.ftz.tf32.f32",
        "cvt.rna.satfinite.tf32.f32",
        "m16n8k4.row.col.f32.tf32",
    ] {
        assert!(
            !code.contains(forbidden),
            "portable TF32 source contains forbidden {forbidden}"
        );
    }

    let mut nn_a_banks = BTreeSet::new();
    let mut nn_b_banks = BTreeSet::new();
    let mut tn_wide_a_banks = BTreeSet::new();
    let mut tn_thin_a_banks = BTreeSet::new();
    let mut nt_b_banks = BTreeSet::new();
    for lane in 0..32 {
        let group = lane >> 2;
        let thread = lane & 3;
        assert!(nn_a_banks.insert((4 * group + thread) % 32));
        assert!(nn_b_banks.insert((8 * thread + group) % 32));
        assert!(tn_wide_a_banks.insert((8 * thread + group) % 32));
        assert!(tn_thin_a_banks.insert((24 * thread + group) % 32));
        assert!(nt_b_banks.insert((4 * group + thread) % 32));
    }
    assert_eq!(nn_a_banks.len(), 32);
    assert_eq!(nn_b_banks.len(), 32);
    assert_eq!(tn_wide_a_banks.len(), 32);
    assert_eq!(tn_thin_a_banks.len(), 32);
    assert_eq!(nt_b_banks.len(), 32);
}

#[test]
fn portable_tf32_splitk2_and_splitk4_have_fixed_partitions_and_distinct_ownership() {
    for symbol in [
        "nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4",
        "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
    ] {
        assert!(
            SM80_SOURCE.contains(&format!("TF32_ASSERT_SPLITK_KERNEL_SIGNATURE({symbol})")),
            "{symbol} is missing its C++ signature assertion"
        );
    }

    let fused = compact_code(&source_mask(braced_scope_after(
        SM80_SOURCE,
        "tf32_splitk_fused_kernel",
    )));
    assert!(SM80_SOURCE.contains("tf32_splitk_fused_kernel<SgbTf32Nn, 16, 32, 4, 2>"));
    assert!(SM80_SOURCE.contains("tf32_splitk_fused_kernel<SgbTf32Nn, 16, 32, 4, 4>"));
    assert!(SM80_SOURCE.contains("atomicInc(counters + tile, Partitions - 1U)"));
    assert!(SM80_SOURCE.contains("if constexpr (Partitions == 4)"));
    assert!(SM80_SOURCE.contains("nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4"));
    assert_contains_all(
        SM80_SOURCE,
        &[
            "tf32_partial_store_cg",
            "tf32_partial_store_cg_float2",
            "tf32_partial_load_cg",
            "tf32_partial_load_cg_float2",
        ],
        "portable TF32 split-K global visibility path",
    );
    for opcode in [
        "st.global.cg.f32",
        "st.global.cg.v2.f32",
        "ld.global.cg.f32",
        "ld.global.cg.v2.f32",
    ] {
        assert!(
            SM80_SOURCE.contains(opcode),
            "portable TF32 split-K global visibility path is missing {opcode}"
        );
    }
    let visibility_helpers = SM80_SOURCE
        .split_once("tf32_partial_store_cg")
        .and_then(|(_, tail)| {
            tail.split_once("template <SgbTf32Op Op, int BM, int BN, int Stages, int Partitions>")
        })
        .map(|(helpers, _)| helpers)
        .expect("portable TF32 split-K visibility helpers");
    assert_eq!(
        visibility_helpers.matches(": \"memory\"").count(),
        4,
        "portable TF32 split-K visibility helpers must freeze compiler memory ordering"
    );
    let prepared_gate = compact_code(braced_scope_after(
        LAUNCH_SOURCE,
        "fn validate_prepared_f32_triad",
    ));
    assert!(
        prepared_gate.contains("prepared.stream_token!=ctx.stream_token()"),
        "portable TF32 split-K shared workspace must remain bound to one ordered CUDA stream"
    );
    assert!(SM80_SOURCE.contains("nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4"));
    assert_contains_all(
        &fused,
        &[
            "blockIdx.z",
            "blockIdx.y",
            "full_tiles",
            "tiles_per_partition",
            "tile_begin",
            "tile_end",
            "tf32_compute_stage",
            "tf32_splitk_async_mainloop",
            "full_output_tile",
            "partial_stride",
            "packed_output",
            "__threadfence",
            "atomicInc",
            "last_partition",
            "__fadd_rn",
            "__fmul_rn",
            "__fmaf_rn",
            "reinterpret_cast<float2*>",
        ],
        "portable TF32 split-K4 fused kernel",
    );
    let mainloop = compact_code(&source_mask(braced_scope_after(
        SM80_SOURCE,
        "tf32_splitk_async_mainloop",
    )));
    assert_contains_all(
        &mainloop,
        &["tf32_splitk_stage_async"],
        "portable TF32 split-K async mainloop",
    );
    let mut cursor = 0;
    for operation in [
        "sum0=__fadd_rn(sum0,p0.x)",
        "sum0=__fadd_rn(sum0,p1.x)",
        "sum0=__fadd_rn(sum0,p2.x)",
        "sum0=__fadd_rn(sum0,p3.x)",
    ] {
        let offset = fused[cursor..]
            .find(operation)
            .unwrap_or_else(|| panic!("split-K4 fused reducer lost ordered operation {operation}"));
        cursor += offset + operation.len();
    }
    for forbidden in ["atomicAdd", "atomicExch", "atomicCAS"] {
        assert!(
            !fused.contains(forbidden),
            "portable TF32 split-K4 source contains forbidden {forbidden}"
        );
    }
    assert_contains_all(
        IDENTITY_SOURCE,
        &[
            "MmaTf32RnaSplitK4",
            "MmaTf32RnaSplitK2",
            "MmaTf32RnaSplitK8",
            "LastCtaPerOutputTileFixedSplitK2Reduce",
            "LastCtaPerOutputTileFixedSplitK4Reduce",
            "LastCtaPerOutputTileFixedSplitK8Reduce",
        ],
        "portable TF32 split-K4 numeric identity",
    );
    assert_contains_all(
        MODULE_SOURCE,
        &[
            "census_tf32_splitk_driver_abi",
            "validate_tf32_splitk_ptx",
            "tf32_splitk_functions",
            "tf32_splitk_function",
            "TF32_SPLITK_CANDIDATE_SPECS",
            "load_tf32_splitk_functions",
            "tf32_driver_abi(symbol)?",
        ],
        "portable TF32 split-K4 loader",
    );
}

#[test]
fn qualification_binds_every_boundary_output_corpus() {
    let boundary = braced_scope_after(QUALIFICATION_SOURCE, "fn qualify_shape_boundaries");
    for required in [
        "tf32-qualification-boundary-corpus.v1",
        "qualify_single_eager",
        "b\"reduction\"",
        "b\"row-tail\"",
        "b\"column-tail\"",
    ] {
        assert!(
            boundary.contains(required),
            "boundary corpus lost {required}"
        );
    }
    assert_eq!(boundary.matches("append_boundary_digest_case").count(), 3);
    assert!(boundary.contains("let mut observed_cases = 0_usize"));
    assert_eq!(boundary.matches("observed_cases += 1").count(), 3);
    assert!(boundary.contains("observed_cases != boundary_cases_per_route()"));
    let append = braced_scope_after(QUALIFICATION_SOURCE, "fn append_boundary_digest_case");
    assert!(append.contains("required(b\"output\""));
    let single = braced_scope_after(QUALIFICATION_SOURCE, "fn qualify_single_eager");
    for required in [
        "check_accuracy",
        "tf32-qualification-boundary-output.v1",
        "&actual",
    ] {
        assert!(single.contains(required), "boundary output lost {required}");
    }
    for required in [
        "MambaBiTf32QualificationArtifactV5",
        "MambaBiTf32QualificationV5",
        "boundary_output_digest",
        "boundary_cases_per_route",
        r#"\"boundary_cases\":"#,
        "tf32-qualification-all-boundary-output.v1",
        "route.boundary_digest",
        "driver_jit_resource_digest",
        "validate_driver_jit_resource_inventory",
        "validate_tf32_driver_jit_local_memory(",
        "local_admission.observed_bytes",
        "local_admission.approved_cap_bytes",
        "tf32-driver-jit-local-resources.v1",
        "route.eager_route_digest",
    ] {
        assert!(
            QUALIFICATION_SOURCE.contains(required),
            "qualification V5 lost {required}"
        );
    }
    assert!(!QUALIFICATION_SOURCE.contains("tf32_driver_jit_local_memory_cap"));
}

#[test]
fn m16n8k8_lane_map_covers_each_fragment_element_once() {
    let mut a = BTreeSet::new();
    let mut b = BTreeSet::new();
    let mut c = BTreeSet::new();
    for lane in 0..32 {
        let group = lane >> 2;
        let thread = lane & 3;
        for coordinate in [
            (group, thread),
            (group + 8, thread),
            (group, thread + 4),
            (group + 8, thread + 4),
        ] {
            assert!(a.insert(coordinate));
        }
        for coordinate in [(thread, group), (thread + 4, group)] {
            assert!(b.insert(coordinate));
        }
        for coordinate in [
            (group, 2 * thread),
            (group, 2 * thread + 1),
            (group + 8, 2 * thread),
            (group + 8, 2 * thread + 1),
        ] {
            assert!(c.insert(coordinate));
        }
    }
    assert_eq!(
        a,
        (0..16)
            .flat_map(|row| (0..8).map(move |col| (row, col)))
            .collect()
    );
    assert_eq!(
        b,
        (0..8)
            .flat_map(|row| (0..8).map(move |col| (row, col)))
            .collect()
    );
    assert_eq!(
        c,
        (0..16)
            .flat_map(|row| (0..8).map(move |col| (row, col)))
            .collect()
    );
}

#[test]
fn single_owner_oracle_covers_every_tf32_tile() {
    assert_single_owner(
        (128, 64),
        256,
        |thread| {
            let warp = thread >> 5;
            if warp >= 4 {
                return Vec::new();
            }
            let lane = thread & 31;
            let warp_m = (warp >> 1) * 64;
            let warp_n = (warp & 1) * 32;
            [0, 16, 32, 48]
                .into_iter()
                .flat_map(|row| {
                    [0, 8, 16, 24]
                        .into_iter()
                        .flat_map(move |column| m16n8_owner(lane, warp_m + row, warp_n + column))
                })
                .collect()
        },
        "portable M128N64",
    );
    assert_single_owner(
        (64, 64),
        128,
        |thread| {
            let warp = thread >> 5;
            let lane = thread & 31;
            let warp_m = (warp >> 1) * 32;
            let warp_n = (warp & 1) * 32;
            [0, 16]
                .into_iter()
                .flat_map(|row| {
                    [0, 8, 16, 24]
                        .into_iter()
                        .flat_map(move |column| m16n8_owner(lane, warp_m + row, warp_n + column))
                })
                .collect()
        },
        "portable M64N64",
    );
    assert_single_owner(
        (16, 32),
        128,
        |thread| m16n8_owner(thread & 31, 0, (thread >> 5) * 8),
        "portable M16N32",
    );
    assert_single_owner(
        (16, 16),
        64,
        |thread| m16n8_owner(thread & 31, 0, (thread >> 5) * 8),
        "portable M16N16",
    );
    assert_single_owner(
        (64, 128),
        256,
        |thread| {
            if thread >= 128 {
                return Vec::new();
            }
            (0..64)
                .map(|register| {
                    let q = thread & 3;
                    let row8 = (thread >> 2) & 7;
                    let warp = thread >> 5;
                    let pair = register & 1;
                    let row_half = (register >> 1) & 1;
                    let n_group = register >> 2;
                    (row8 + 16 * warp + 8 * row_half, 2 * q + pair + 8 * n_group)
                })
                .collect()
        },
        "SM90a M64N128",
    );
    for columns in [64, 128] {
        assert_single_owner(
            (128, columns),
            128,
            |thread| {
                let row = ((thread >> 5) & 3) * 32 + (thread & 31);
                (0..columns / 8)
                    .flat_map(|chunk| (0..8).map(move |column| (row, chunk * 8 + column)))
                    .collect()
            },
            &format!("SM100 M128N{columns} c4"),
        );
        assert_single_owner(
            (128, columns),
            256,
            |thread| {
                let row = ((thread >> 5) & 3) * 32 + (thread & 31);
                let group = thread >> 7;
                (group..columns / 8)
                    .step_by(2)
                    .flat_map(|chunk| (0..8).map(move |column| (row, chunk * 8 + column)))
                    .collect()
            },
            &format!("SM100 M128N{columns} p8"),
        );
    }
    for (rows, columns) in [(128, 64), (64, 128)] {
        assert_single_owner(
            (rows, columns),
            256,
            |thread| {
                let warp = thread >> 5;
                let lane = thread & 31;
                let (warp_m, warp_n) = if rows == 128 {
                    ((warp >> 1) * 32, (warp & 1) * 32)
                } else {
                    ((warp >> 2) * 32, (warp & 3) * 32)
                };
                [0, 16]
                    .into_iter()
                    .flat_map(|row| {
                        [0, 8, 16, 24].into_iter().flat_map(move |column| {
                            m16n8_owner(lane, warp_m + row, warp_n + column)
                        })
                    })
                    .collect()
            },
            &format!("SM120 M{rows}N{columns}"),
        );
    }
}

#[test]
fn bk32_always_issues_four_ordered_k8_atoms_and_zero_fills_the_tail() {
    const K_CASES: [usize; 16] = [0, 1, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 65, 97, 129, 257];
    const ISSUES: [usize; 4] = [0, 8, 16, 24];
    for k in K_CASES {
        let mut observed = Vec::new();
        let tiles = k.div_ceil(32);
        for tile in 0..tiles {
            for issue in ISSUES {
                for element in 0..8 {
                    let index = tile * 32 + issue + element;
                    observed.push((index < k).then_some(index));
                }
            }
        }
        let present: Vec<_> = observed.iter().flatten().copied().collect();
        assert_eq!(present, (0..k).collect::<Vec<_>>(), "K={k}");
        assert_eq!(observed.len(), tiles * 32, "K={k}");
        assert_eq!(
            observed.iter().filter(|value| value.is_none()).count(),
            tiles * 32 - k
        );
    }

    for (name, source, family) in [
        ("SM80", SM80_SOURCE, "_sm80_mma_tf32_"),
        ("SM90a", SM90A_SOURCE, "_sm90a_wgmma_tf32_"),
        ("SM100", SM100_SOURCE, "_sm100_tcgen_tf32_"),
        ("SM120", SM120_SOURCE, "_sm120_tma_mma_tf32_"),
    ] {
        assert!(
            source.contains(family),
            "{name} TF32 source family is missing before K-order validation"
        );
        assert!(
            source.contains("0, 8, 16, 24")
                || source.contains("{0, 8, 16, 24}")
                || source.contains("< 4") && source.contains("* 8"),
            "{name} TF32 source must structurally issue K offsets 0,8,16,24"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OracleOp {
    Nn,
    Tn,
    Nt,
}

fn normalized_problem(op: OracleOp, shape: (usize, usize, usize)) -> (usize, usize, usize) {
    let (m, k, n) = shape;
    match op {
        OracleOp::Nn => (m, n, k),
        OracleOp::Tn => (k, n, m),
        OracleOp::Nt => (m, k, n),
    }
}

fn zero_reduction_epilogue(
    op: OracleOp,
    alpha: f32,
    beta: f32,
    bias: Option<f32>,
    old_output: f32,
) -> f32 {
    match op {
        OracleOp::Nn => {
            let accumulator = bias.unwrap_or(0.0);
            let value = if alpha == 1.0 {
                accumulator
            } else {
                alpha * accumulator
            };
            if beta == 0.0 {
                value
            } else {
                beta.mul_add(old_output, value)
            }
        }
        OracleOp::Tn => alpha.mul_add(0.0, old_output),
        OracleOp::Nt => {
            if alpha == 1.0 {
                0.0
            } else {
                alpha * 0.0
            }
        }
    }
}

#[test]
fn zero_reduction_is_op_normalized_and_keeps_exact_epilogue_rounding() {
    for (op, shape, expected_output) in [
        (OracleOp::Nn, (3, 0, 5), (3, 5)),
        (OracleOp::Tn, (0, 3, 5), (3, 5)),
        (OracleOp::Nt, (3, 5, 0), (3, 5)),
    ] {
        let (rows, columns, reduction) = normalized_problem(op, shape);
        assert_eq!(reduction, 0, "{op:?}");
        assert_eq!((rows, columns), expected_output, "{op:?}");
    }

    assert_eq!(
        zero_reduction_epilogue(OracleOp::Nn, 1.0, 2.0, Some(3.0), 4.0),
        11.0
    );
    assert_eq!(
        zero_reduction_epilogue(OracleOp::Tn, -7.0, 1.0, None, 5.0),
        5.0
    );
    assert!(zero_reduction_epilogue(OracleOp::Tn, f32::INFINITY, 1.0, None, 5.0).is_nan());
    assert_eq!(
        zero_reduction_epilogue(OracleOp::Nt, -1.0, 0.0, None, 9.0).to_bits(),
        (-0.0_f32).to_bits()
    );
    assert!(zero_reduction_epilogue(OracleOp::Nt, f32::INFINITY, 0.0, None, 9.0).is_nan());
}

#[test]
fn zero_reduction_preparation_uses_a_revisioned_mapless_sentinel() {
    assert_code_contains_all(
        CONTRACT_SOURCE,
        &[
            "fn output_rows",
            "fn output_columns",
            "fn reduction",
            "ZeroReduction",
            "Encoded",
            "zeroed_tensor_map_sentinel",
            "ZERO_REDUCTION_MAP_REVISION",
            "ZERO_REDUCTION_DIGEST_DOMAIN",
        ],
        "zero-reduction prepared map contract",
    );
    let validation = braced_scope_after(CONTRACT_SOURCE, "impl F32TriadShape");
    assert_code_contains_all(
        validation,
        &[
            "ResolvedGemmOp::Nn",
            "ResolvedGemmOp::Tn",
            "ResolvedGemmOp::Nt",
        ],
        "op-normalized F32 shape validation",
    );

    let zero_sentinel = source_mask(braced_scope_after(
        CONTRACT_SOURCE,
        "pub fn zeroed_tensor_map_sentinel",
    ));
    assert_code_contains_all(
        &zero_sentinel,
        &[
            "MaybeUninit::<sys::CUtensorMap>::zeroed().assume_init()",
            "Tf32TensorMap(raw)",
        ],
        "production zeroed tensor-map sentinel",
    );

    let zero_constructor = source_mask(braced_scope_after(
        CONTRACT_SOURCE,
        "pub(super) fn zero_reduction",
    ));
    assert_code_contains_all(
        &zero_constructor,
        &[
            "Self::ZeroReduction",
            "a: zeroed_tensor_map_sentinel()",
            "b: zeroed_tensor_map_sentinel()",
            "revision: ZERO_REDUCTION_MAP_REVISION",
        ],
        "dedicated K=0 mapless tensor-map constructor",
    );

    let maps_identity = source_mask(braced_scope_after(
        CONTRACT_SOURCE,
        "pub fn identity_digest(&self) -> Sha256Digest",
    ));
    let zero_identity_marker = "Self::ZeroReduction { data } =>";
    let zero_identity_start = maps_identity
        .find(zero_identity_marker)
        .expect("ZeroReduction identity branch")
        + zero_identity_marker.len();
    let zero_identity = braced_scope_at(
        &maps_identity,
        zero_identity_start,
        "ZeroReduction identity branch",
    );
    assert_code_contains_all(
        zero_identity,
        &[
            "ZERO_REDUCTION_DIGEST_DOMAIN",
            "data.revision.to_le_bytes()",
            "data.format",
            "data.route",
            "std::mem::size_of::<sys::CUtensorMap>()",
            "std::mem::align_of::<sys::CUtensorMap>()",
        ],
        "revisioned K=0 map identity",
    );
    assert!(
        !zero_identity.contains("ZERO_REDUCTION_MAP_REVISION"),
        "K=0 identity must consume the stored sentinel revision"
    );

    let preparation = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad",
    ));
    let zero_marker = preparation
        .find("reduction() == 0")
        .or_else(|| preparation.find("reduction == 0"))
        .expect("prepare_f32_triad must branch on the normalized reduction");
    let zero_branch = braced_scope_after(&preparation[zero_marker..], "if ");
    let nonzero_tail = preparation
        .find("match resolve_f32_triad_auto")
        .expect("nonzero f32 dispatch tail");
    let zero_prelude = &preparation[..nonzero_tail];
    assert_code_contains_all(
        zero_branch,
        &[
            "F32PreparedTensorMaps::zero_reduction(",
            "prepare_scalar_zero_f32",
            "return Ok(",
        ],
        "K=0 mapless preparation branch",
    );
    validate_nonzero_input_pointer_guard(LAUNCH_SOURCE).unwrap_or_else(|error| panic!("{error}"));
    let operand_validation = source_mask(
        active_production_function_scope(LAUNCH_SOURCE, "validate_f32_triad_operands")
            .unwrap_or_else(|error| panic!("{error}")),
    );
    assert_code_excludes_all(
        &operand_validation,
        &[
            "with_inputs",
            "prepare_f32_maps_with",
            "tf32_tensor_map_plan",
            "prepare_tf32_tensor_maps",
            "Tf32TensorMap::encode",
            "cuTensorMapEncodeTiled",
        ],
        "operand validation",
    );
    let capture_guard = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "fn require_f32_preparation_outside_capture",
    ));
    let output_query = source_mask(braced_scope_after(
        CONTRACT_SOURCE,
        "pub(super) fn query_output",
    ));
    assert_code_contains_all(
        &output_query,
        &["operands.output", "operands.bias", "inputs: None"],
        "K=0 output-only allocation query",
    );
    let scalar_zero = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "fn prepare_scalar_zero_f32",
    ));
    assert_code_contains_all(
        &scalar_zero,
        &[
            "maps.identity_digest()",
            "resources.digest(request, operands, maps_digest)",
        ],
        "K=0 scalar preparation digest chain",
    );
    let resource_digest = source_mask(braced_scope_after(CONTRACT_SOURCE, "pub(super) fn digest"));
    assert_code_contains_all(
        &resource_digest,
        &[
            "tensor_maps_digest",
            "operands.output",
            "operands.bias",
            "self.inputs",
        ],
        "K=0 launch resource digest",
    );
    assert!(
        !token_present(&resource_digest, "Sm90aAllocationIdentity::query"),
        "K=0 resource digest may consume captured identities but may not query allocations"
    );
    for forbidden in [
        "with_inputs",
        "prepare_f32_maps_with",
        "tf32_tensor_map_plan",
        "prepare_tf32_tensor_maps",
        "Tf32TensorMap::encode",
        "cuTensorMapEncodeTiled",
        "operands.a",
        "operands.b",
    ] {
        for (scope, label) in [
            (zero_prelude, "public preparation prelude"),
            (zero_branch, "preparation branch"),
            (capture_guard.as_str(), "capture guard"),
            (output_query.as_str(), "output resource query"),
            (zero_sentinel.as_str(), "sentinel constructor"),
            (zero_constructor.as_str(), "constructor"),
            (zero_identity, "identity digest"),
            (scalar_zero.as_str(), "scalar zero preparation"),
            (resource_digest.as_str(), "resource digest"),
        ] {
            assert!(
                !token_present(scope, forbidden),
                "K=0 mapless {label} may not perform {forbidden}"
            );
        }
    }

    let contract_tests = braced_scope_after(CONTRACT_SOURCE, "mod tests");
    let normalized = braced_scope_after(
        contract_tests,
        "fn f32_zero_reduction_validation_is_op_normalized",
    );
    assert_code_contains_all(
        normalized,
        &[
            "ResolvedGemmOp::Nn",
            "ResolvedGemmOp::Tn",
            "ResolvedGemmOp::Nt",
            "reduction()",
            "output_rows()",
            "output_columns()",
            "is_err()",
        ],
        "behavioral zero-reduction host validation",
    );

    let identity_tests = active_test_module_scope(IDENTITY_SOURCE, "cache_and_header_tests")
        .unwrap_or_else(|error| panic!("{error}"));
    let domain = source_mask(
        direct_test_function_scope(
            identity_tests,
            "zero_reduction_identity_is_domain_separated_and_pointer_free",
            "#[test]",
        )
        .unwrap_or_else(|error| panic!("{error}")),
    );
    assert_code_contains_all(
        &domain,
        &[
            "ZeroReduction",
            "Encoded",
            "ZERO_REDUCTION_MAP_REVISION",
            "ZERO_REDUCTION_DIGEST_DOMAIN",
            "assert_ne!",
            "build_resolved_gemm_launch_set",
        ],
        "zero-reduction map and graph-route digest domains",
    );
    let route_identity = source_mask(braced_scope_after(
        IDENTITY_SOURCE,
        "fn build_zero_reduction_route_identity",
    ));
    for forbidden in [
        "operands.a",
        "operands.b",
        "a_map",
        "b_map",
        "AllocationIdentity",
    ] {
        assert!(
            !route_identity.contains(forbidden),
            "K=0 graph identity must not depend on {forbidden}"
        );
    }
}

#[test]
fn zero_reduction_device_branch_dominates_every_descriptor_use() {
    let specialized: BTreeSet<_> = expected_sm90a_symbols()
        .into_iter()
        .chain(expected_sm100_symbols())
        .chain(expected_sm120_symbols())
        .collect();
    assert_eq!(specialized.len(), 60, "canonical specialized K=0 census");
    for (label, source) in [
        ("SM80", SM80_SOURCE),
        ("SM90a", SM90A_SOURCE),
        ("SM100", SM100_SOURCE),
        ("SM120", SM120_SOURCE),
    ] {
        let source = source_mask(source);
        assert!(
            source.matches("zero_reduction_epilogue").count() >= 2,
            "{label} must define and call one fixed K=0 epilogue"
        );
        let marker = source
            .find("reduction == 0")
            .unwrap_or_else(|| panic!("{label} must branch on normalized reduction == 0"));
        let branch = braced_scope_after(&source[marker..], "if ");
        assert_code_contains_all(
            branch,
            &["zero_reduction_epilogue", "return"],
            &format!("{label} uniform K=0 branch"),
        );
        for forbidden in [
            "a_map",
            "b_map",
            "CUtensorMap",
            "cp.async",
            "mbarrier",
            "wgmma.",
            "tcgen05.",
            "mma.sync",
            "__syncthreads",
        ] {
            assert!(
                !branch.contains(forbidden),
                "{label} K=0 branch may not touch {forbidden}"
            );
        }
    }
}

#[test]
fn tf32_conversion_goldens_freeze_rna_ties_without_global_nan_promises() {
    let cases = [
        (0x3f80_0fff, 0x3f80_0000),
        (0x3f80_1000, 0x3f80_2000),
        (0x3f80_1001, 0x3f80_2000),
        (0xbf80_1000, 0xbf80_2000),
    ];
    for (input, expected) in cases {
        assert_eq!(tf32_rna_finite_normal(input), expected);
    }

    let artifact_scoped_edges = [
        0x0000_0000_u32,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7f80_0001,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
    ];
    assert_eq!(artifact_scoped_edges.len(), 10);
    assert_contains_all(
        MODULE_SOURCE,
        &[
            "TF32_EXCEPTIONAL_PROBE_BITS",
            "0x00000000",
            "0x80000000",
            "0x7f800000",
            "0xff800000",
            "0x7fc00001",
            "0x7f800001",
            "0x00000001",
            "0x007fffff",
            "0x00800000",
            "0x7f7fffff",
        ],
        "artifact-scoped exceptional conversion vectors",
    );
    let artifact_probe = braced_scope_after(MODULE_SOURCE, "fn qualify_tf32_conversion_artifact");
    assert_contains_all(
        artifact_probe,
        &[
            "TF32_EXCEPTIONAL_PROBE_BITS",
            "artifact",
            "launch",
            "download",
            "digest",
        ],
        "production exceptional-value artifact probe",
    );
}

#[test]
fn tma_tf32_descriptor_contract_freezes_k8_offsets_and_plane_geometry() {
    let descriptor = |start: u64, leading: u64, stride: u64| {
        (start & 0x3fff)
            | ((leading & 0x3fff) << 16)
            | ((stride & 0x3fff) << 32)
            | (1 << 46)
            | (2 << 61)
    };
    let decode = |value: u64| {
        (
            value & 0x3fff,
            (value >> 16) & 0x3fff,
            (value >> 32) & 0x3fff,
            (value >> 61) & 7,
        )
    };
    assert_eq!(decode(descriptor(0, 1, 64)), (0, 1, 64, 2));
    assert_eq!(decode(descriptor(0, 0, 64)), (0, 0, 64, 2));
    assert_eq!(decode(descriptor(0, 256, 64)), (0, 256, 64, 2));

    let k_offsets = [0_u64, 8, 16, 24];
    assert_eq!(k_offsets.map(|offset| offset / 4), [0, 2, 4, 6]);
    assert_eq!(k_offsets.map(|offset| offset * 8), [0, 64, 128, 192]);
    let boxes = BTreeSet::from([
        ("nn-a", 32, 64),
        ("nn-b-plane", 32, 32),
        ("tn-x-plane", 32, 32),
        ("tn-dy-plane", 32, 32),
        ("nt-dy", 32, 64),
        ("nt-w", 32, 128),
    ]);
    assert_eq!(boxes.len(), 6);
    let sm90a_decode = braced_scope_after(SM90A_SOURCE, "sm90a_tf32_desc");
    assert_contains_all(
        sm90a_decode,
        &["0x3fff", "<< 16", "<< 32", "1ULL << 46", "2ULL << 61"],
        "SM90a production TF32 descriptor encoder",
    );
    let sm100_decode = braced_scope_after(SM100_SOURCE, "sm100_tf32_instruction_descriptor");
    assert_contains_all(
        sm100_decode,
        &[
            "0x08100910",
            "0x08110910",
            "0x08118910",
            "0x08200910",
            "0x08210910",
            "0x08218910",
            "static_assert",
        ],
        "SM100 production TF32 instruction descriptor decode",
    );
    let production = braced_scope_after(
        CONTRACT_SOURCE,
        "fn tf32_descriptor_oracles_match_production_builders",
    );
    assert_code_contains_all(
        production,
        &[
            "build_sm90a_tf32_descriptor",
            "decode_sm90a_tf32_descriptor",
            "sm100_tf32_instruction_descriptor",
            "assert_eq!",
            "0x4000404000010000",
            "0x40007fff00010000",
            "0x08100910",
            "0x08218910",
        ],
        "direct production descriptor oracle",
    );
    assert_code_contains_all(
        braced_scope_after(LAUNCH_SOURCE, "fn prepare_specialized_tf32_maps"),
        &[
            "build_sm90a_tf32_descriptor",
            "sm100_tf32_instruction_descriptor",
        ],
        "production descriptor launch wiring",
    );
}

#[test]
fn sm120_sw128_production_decode_is_a_bijection_for_every_phase() {
    let decode = compact_code(braced_scope_after(SM120_SOURCE, "sm120_tf32_sw128_offset"));
    assert_eq!(
        decode,
        "sm120_tf32_sw128_offset(unsignedplane_base,unsignedlogical_row,unsignedelement){\
         unsignedchunk=element/4;unsignedelement_in_vector=element&3;\
         unsignedoffset=(plane_base/128)%8;\
         unsignedphysical_chunk=chunk^((logical_row+offset)%8);\
         returnplane_base+logical_row*128+physical_chunk*16+element_in_vector*4;}"
    );

    for phase in 0_usize..8 {
        let plane_base = phase * 128;
        for logical_row in 0_usize..32 {
            let mut physical = BTreeSet::new();
            for element in 0_usize..32 {
                let chunk = element / 4;
                let element_in_vector = element & 3;
                let offset = (plane_base / 128) % 8;
                let physical_chunk = chunk ^ ((logical_row + offset) % 8);
                let address = logical_row * 128 + physical_chunk * 16 + element_in_vector * 4;
                assert!(physical.insert(address), "phase={phase} row={logical_row}");
            }
            let expected: BTreeSet<_> = (0..32)
                .map(|element| logical_row * 128 + element * 4)
                .collect();
            assert_eq!(physical, expected, "phase={phase} row={logical_row}");
        }
    }
    let load_a = compact_code(braced_scope_after(SM120_SOURCE, "sm120_tf32_load_a"));
    assert_eq!(
        load_a,
        "sm120_tf32_load_a(unsignedchar*storage,intstage,introw,intreduction){\
         constexprintstage_bytes=Sm120Tf32Storage<M,N,Stages>::stage_bytes;\
         unsignedplane=(unsigned)(row/32)*4096U;\
         unsignedlogical_row=(unsigned)(row&31);unsignedelement=(unsigned)reduction;\
         ifconstexpr(Op==Sm120Tn){logical_row=(unsigned)reduction;element=(unsigned)(row&31);}\
         unsignedoffset=sm120_tf32_sw128_offset(plane,logical_row,element);\
         return*reinterpret_cast<float*>(storage+stage*stage_bytes+offset);}"
    );
    let load_b = compact_code(braced_scope_after(SM120_SOURCE, "sm120_tf32_load_b"));
    assert_eq!(
        load_b,
        "sm120_tf32_load_b(unsignedchar*storage,intstage,intreduction,intcolumn){\
         constexprintstage_bytes=Sm120Tf32Storage<M,N,Stages>::stage_bytes;\
         unsignedbase=M*32*4;unsignedplane=base+(unsigned)(column/32)*4096U;\
         unsignedlogical_row=(unsigned)reduction;unsignedelement=(unsigned)(column&31);\
         ifconstexpr(Op==Sm120Nt){logical_row=(unsigned)(column&31);element=(unsigned)reduction;}\
         unsignedoffset=sm120_tf32_sw128_offset(plane,logical_row,element);\
         return*reinterpret_cast<float*>(storage+stage*stage_bytes+offset);}"
    );
    assert!(
        compact_code(SM120_SOURCE).contains(
            "template<intOp,intM,intN,intStages>static__device__\
             __forceinline__voidsm120_tf32_issue_stage"
        ),
        "the TF32 issue stage must carry the operation at compile time"
    );
    assert_contains_all(
        SM120_SOURCE,
        &["sm120_tf32_issue_stage<Op, M, N, Stages>"],
        "operation-aware TF32 issue-stage call",
    );
    assert_code_contains_all(
        braced_scope_after(
            CONTRACT_SOURCE,
            "fn sm120_sw128_oracle_matches_production_decode",
        ),
        &[
            "sm120_tf32_sw128_offset",
            "ResolvedGemmOp::Nn",
            "ResolvedGemmOp::Tn",
            "ResolvedGemmOp::Nt",
            "assert_eq!",
        ],
        "direct SM120 production decode oracle",
    );
}

#[test]
fn wgmma_tf32_uses_tfloat32_tma_and_exact_sm90a_k8_contract() {
    assert_contains_all(
        SM90A_SOURCE,
        &[
            "sm90a_tf32_wgmma_k8",
            "SM90A_STAGE_BYTES 24576",
            "SM90A_DYNAMIC_SHARED_BYTES 73984",
            "sm90a_tf32_wgmma_fence",
            "sm90a_tf32_wgmma_commit",
            "sm90a_tf32_wgmma_wait",
        ],
        "SM90a TF32 WGMMA source",
    );
    assert!(
        !source_mask(SM90A_SOURCE).contains("cvt.rna.tf32.f32"),
        "WGMMA TFLOAT32 tensor maps own conversion; register cvt is forbidden"
    );
    assert!(
        source_mask(CONTRACT_SOURCE).contains("CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"),
        "SM90a/SM100 TF32 tensor maps must request TFLOAT32 conversion"
    );
    assert!(
        source_mask(SM90A_SOURCE).contains("__CUDA_ARCH__ == 900"),
        "WGMMA route must remain exact sm_90a"
    );
}

#[test]
fn tcgen_tf32_uses_k8_descriptor_constants_and_never_reaches_sm120() {
    assert_contains_all(
        SM100_SOURCE,
        &[
            "sm100_tf32_tcgen05_k8",
            "0x08100910",
            "0x08110910",
            "0x08118910",
            "0x08200910",
            "0x08210910",
            "0x08218910",
            "__CUDA_ARCH__ == 1000",
            "__CUDA_ARCH__ == 1030",
        ],
        "SM100 TF32 TCGEN05 source",
    );
    assert!(
        !source_mask(SM120_SOURCE).contains("tcgen05")
            && !source_mask(SM120_SOURCE).contains("kind::tf32"),
        "SM120 must use warp MMA rather than TCGEN05"
    );
}

#[test]
fn sm120_tf32_tma_preserves_f32_bits_before_rna_warp_mma() {
    assert_contains_all(
        SM120_SOURCE,
        &[
            "sm120_tf32_rna",
            "sm120_tf32_mma_m16n8k8",
            "__CUDA_ARCH__ == 1200",
            "__CUDA_ARCH__ == 1210",
        ],
        "SM120 TF32 TMA+MMA source",
    );
    assert!(
        source_mask(CONTRACT_SOURCE).contains("CU_TENSOR_MAP_DATA_TYPE_UINT32"),
        "SM120 tensor maps must move raw f32 bits as UINT32"
    );
    assert!(
        !source_mask(SM120_SOURCE).contains("CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"),
        "SM120 must not perform TFLOAT32 conversion in TMA"
    );
}

#[test]
fn tf32_epilogues_keep_f32_rounding_placement_and_single_owner_reduction() {
    for (name, source, helper, store, params, nn, tn, rows, columns, portable) in [
        (
            "SM80",
            SM80_SOURCE,
            "tf32_epilogue",
            "tf32_store",
            "Sm80Tf32KernelParams",
            "SgbTf32Nn",
            "SgbTf32Tn",
            "tf32_rows",
            "tf32_columns",
            true,
        ),
        (
            "SM90a",
            SM90A_SOURCE,
            "sm90a_tf32_epilogue",
            "sm90a_tf32_store",
            "Sm90aTf32KernelParams",
            "Sm90aNn",
            "Sm90aTn",
            "sm90a_tf32_rows",
            "sm90a_tf32_columns",
            false,
        ),
        (
            "SM100",
            SM100_SOURCE,
            "sm100_tf32_epilogue",
            "sm100_tf32_store",
            "Sm100KernelParams",
            "Sm100Nn",
            "Sm100Tn",
            "sm100_tf32_rows",
            "sm100_tf32_columns",
            false,
        ),
        (
            "SM120",
            SM120_SOURCE,
            "sm120_tf32_epilogue",
            "sm120_tf32_store",
            "Sm120KernelParams",
            "Sm120Nn",
            "Sm120Tn",
            "sm120_tf32_rows",
            "sm120_tf32_columns",
            false,
        ),
    ] {
        let exact_branches = format!(
            "ifconstexpr(Op=={nn}){{(void)bias;(void)column;\
             floatvalue=params.alpha==1.0f?accumulator:__fmul_rn(params.alpha,accumulator);\
             if(params.beta==0.0f)returnvalue;\
             return__fmaf_rn(params.beta,old_output,value);}}\
             elseifconstexpr(Op=={tn}){{(void)bias;(void)column;\
             return__fmaf_rn(params.alpha,accumulator,old_output);}}\
             else{{(void)old_output;(void)bias;(void)column;\
             returnparams.alpha==1.0f?accumulator:__fmul_rn(params.alpha,accumulator);}}"
        );
        let expected_epilogue = format!(
            "{helper}(floataccumulator,floatold_output,constfloat*bias,intcolumn,\
             const{params}&params){{{exact_branches}}}"
        );
        assert_eq!(
            compact_cuda_executable_scope(source, helper),
            expected_epilogue,
            "{name} TF32 epilogue executable scope changed"
        );

        let store_prefix = if portable {
            format!(
                "{store}(float*output,introw,intcolumn,floataccumulator,constfloat*bias,\
                 const{params}&params){{introws={rows}<Op>(params);\
                 intcolumns={columns}<Op>(params);\
                 if(row>=rows||column>=columns)return;\
                 float*destination=output+(longlong)row*params.ldc+column;"
            )
        } else {
            format!(
                "{store}(void*output,introw,intcolumn,floataccumulator,constfloat*bias,\
                 const{params}&params){{if(row>={rows}<Op>(params)||\
                 column>={columns}<Op>(params))return;\
                 float*destination=static_cast<float*>(output)+\
                 static_cast<longlong>(row)*params.ldc+column;"
            )
        };
        let expected_store = format!(
            "{store_prefix}floatold_output=0.0f;\
             ifconstexpr(Op=={tn}){{old_output=*destination;}}\
             elseifconstexpr(Op=={nn}){{if(params.beta!=0.0f)old_output=*destination;}}\
             floatvalue={helper}<Op>(accumulator,old_output,bias,column,params);\
             *destination=value;}}"
        );
        assert_eq!(
            compact_cuda_executable_scope(source, store),
            expected_store,
            "{name} TF32 store executable scope changed"
        );
        assert!(
            !contains_opcode_prefix(source, "atom.") && !contains_opcode_prefix(source, "red."),
            "{name} triad source must not contain numeric atomics or reductions"
        );
        assert!(
            source_mask(source).matches(helper).count() >= 3,
            "{name} zero and nonzero paths must share the named {helper}"
        );
    }
    assert!(
        source_mask(SM80_SOURCE).contains("_bytes == 0 ? 0"),
        "zero-byte cp.async lanes must not form out-of-allocation pointers"
    );
    assert_contains_all(
        IDENTITY_SOURCE,
        &["OneCtaPerOutputTile", "ownership"],
        "single-CTA resolved route identity",
    );
    let operands = braced_scope_after(LAUNCH_SOURCE, "fn validate_f32_triad_operands");
    assert_contains_all(
        operands,
        &[
            "output",
            "== 0",
            "bias",
            "align_of",
            "ResolvedGemmOp::Nn",
            "alpha",
            "1.0",
            "ResolvedGemmOp::Tn",
            "ResolvedGemmOp::Nt",
        ],
        "host f32 epilogue and pointer validation",
    );
    assert!(
        !operands.contains("is_null"),
        "CUDA device-pointer validation must not create host pointers"
    );
    let portable_stage = braced_scope_after(SM80_SOURCE, "tf32_stage_async");
    assert!(
        portable_stage.contains("_bytes == 0 ? 0"),
        "portable TF32 staging itself must suppress zero-byte pointer formation"
    );
    assert_code_contains_all(
        braced_scope_after(
            CONTRACT_SOURCE,
            "fn tf32_epilogue_matches_scalar_reference_bits",
        ),
        &[
            "ResolvedGemmOp::Nn",
            "ResolvedGemmOp::Tn",
            "ResolvedGemmOp::Nt",
            "to_bits",
            "INFINITY",
            "NAN",
            "-0.0",
        ],
        "bit-exact production epilogue unit oracle",
    );
}

#[test]
fn sm110_feature_targets_nvrtc_ptxas_pipeline() {
    let nvrtc = loaded_nvrtc_version();
    // SM110 is a CUDA 13.2 target: an older toolkit cannot name it at all, so
    // the gate reports the gap in the stack instead of reporting a defect.
    if nvrtc < (13, 2) {
        eprintln!("skipping SM110 compile gate: loaded NVRTC is {nvrtc:?}, needs 13.2");
        return;
    }
    let version = checked_output(
        {
            let mut command = Command::new(cuda_tool("ptxas"));
            command.arg("--version");
            command
        },
        "ptxas --version",
    );
    let version_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&version.stdout),
        String::from_utf8_lossy(&version.stderr)
    );
    let release = version_text
        .split_once("release ")
        .and_then(|(_, tail)| tail.split_once(','))
        .map(|(release, _)| release)
        .unwrap_or_else(|| panic!("unrecognized ptxas version: {version_text}"));
    let mut components = release.split('.');
    let ptxas = (
        components
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        components
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
    );
    assert_eq!(nvrtc, ptxas, "NVRTC and ptxas major-minor must match");

    let expected = expected_sm100_symbols();
    for (nvrtc_target, ptx_target, sass_target) in [
        ("compute_110f", "sm_110f", "sm_110f"),
        ("compute_110a", "sm_110a", "sm_110a"),
    ] {
        let ptx = compile_tf32_ptx(tf32_cuda_blob(SM100_SOURCE, false), nvrtc_target);
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {ptx_target}")),
            "{nvrtc_target} must emit exact PTX target {ptx_target}"
        );
        assert_eq!(
            ptx_entry_symbols(&ptx, "_sm100_tcgen_tf32_"),
            expected,
            "{nvrtc_target} must export the complete SM100 TF32 inventory"
        );
        for symbol in &expected {
            assert_eq!(
                ptx.matches(&format!(".entry {symbol}(")).count(),
                1,
                "{nvrtc_target}/{symbol} must occur exactly once"
            );
            let entry = ptx_entry(&ptx, symbol);
            assert_eq!(
                parse_ptx_parameters(ptx_parameters(entry, symbol)),
                expected_ptx_parameters(symbol, 40, 128),
                "{nvrtc_target}/{symbol} ABI"
            );
            assert_k0_cfg_dominates_entry(entry, symbol, 40);
        }
        assemble_tf32_checker(&ptx, sass_target, nvrtc_target);
        let (resources, object_resources, sass, sass_cfg) =
            assemble_and_disassemble_tf32(&ptx, sass_target, nvrtc_target);
        assert_per_entry_zero_resources(&resources, &expected, nvrtc_target);
        assert_cuobjdump_zero_resources(&object_resources, &expected, nvrtc_target);
        for symbol in &expected {
            assert_sass_entry_contract(&sass, symbol, "SM100");
            assert_sass_cfg_corroboration(&sass_cfg, &sass, SM100_SOURCE, symbol, "SM100");
        }
    }
}

#[test]
fn release_target_entry_matrix_nvrtc_ptxas_pipeline() {
    // The release matrix spans targets that only a 13.2 toolkit can compile.
    let nvrtc = loaded_nvrtc_version();
    if nvrtc < (13, 2) {
        eprintln!("skipping release feature-target matrix: loaded NVRTC is {nvrtc:?}, needs 13.2");
        return;
    }
    let mut checked_entries = 0;
    for (nvrtc_target, ptx_target, family) in RELEASE_TARGET_MATRIX {
        if !release_target_compiles(nvrtc_target, nvrtc) {
            eprintln!("skipping {nvrtc_target}: loaded NVRTC is {nvrtc:?}, needs 13.4");
            continue;
        }
        let (label, source, needs_mma16, expected) = specialized_family_contract(*family);
        let ptx = compile_tf32_ptx(tf32_cuda_blob(source, needs_mma16), nvrtc_target);
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {ptx_target}")),
            "{nvrtc_target} must emit exact PTX target {ptx_target}"
        );
        let family_marker = match family {
            SpecializedTf32Family::Sm90a => "_sm90a_wgmma_tf32_",
            SpecializedTf32Family::Sm100 => "_sm100_tcgen_tf32_",
            SpecializedTf32Family::Sm120 => "_sm120_tma_mma_tf32_",
        };
        assert_eq!(ptx_entry_symbols(&ptx, family_marker), expected);
        for symbol in &expected {
            assert_eq!(
                ptx.matches(&format!(".entry {symbol}(")).count(),
                1,
                "{nvrtc_target}/{symbol} must occur exactly once"
            );
            let entry = ptx_entry(&ptx, symbol);
            assert_eq!(
                parse_ptx_parameters(ptx_parameters(entry, symbol)),
                expected_ptx_parameters(symbol, 40, 128),
                "{nvrtc_target}/{symbol} ABI"
            );
            assert_k0_cfg_dominates_entry(entry, symbol, 40);
        }
        let (resources, object_resources, sass, sass_cfg) =
            assemble_and_disassemble_tf32(&ptx, ptx_target, nvrtc_target);
        assert_per_entry_zero_resources(&resources, &expected, nvrtc_target);
        assert_cuobjdump_zero_resources(&object_resources, &expected, nvrtc_target);
        for symbol in &expected {
            assert_sass_entry_contract(&sass, symbol, label);
            assert_sass_cfg_corroboration(&sass_cfg, &sass, source, symbol, label);
        }
        checked_entries += expected.len();
    }
    assert_eq!(release_entry_target_count((13, 2)), 258);
    assert_eq!(
        release_entry_target_count((13, 4)),
        258 + 2 * expected_sm100_symbols().len()
    );
    assert_eq!(checked_entries, release_entry_target_count(nvrtc));
}

#[test]
fn tf32_target_and_toolchain_admission_is_fail_closed() {
    assert_contains_all(
        DEVICE_SOURCE,
        &["fn resolve_nvrtc_target", "(11, 0)"],
        "SM80+ device target ladder",
    );
    assert_contains_all(
        MODULE_SOURCE,
        &[
            "CudaTarget::Compute110f",
            "CudaTarget::Compute110a",
            "CudaTarget::Sm110f",
            "CudaTarget::Sm110a",
        ],
        "TF32 optional-module admission",
    );
    assert_contains_all(
        ARCH_GATE_SOURCE,
        &[
            "compiles_for_sm80",
            "compiles_for_sm89",
            "compiles_for_sm90a",
            "compiles_for_sm100",
            "compiles_for_sm120",
        ],
        "TF32 PTX/SASS target census",
    );
    assert_contains_all(
        DISPATCH_SOURCE,
        &["ExactScalarFma", "AllowDeterministicTf32", "ScalarFma"],
        "exact/allow TF32 dispatch",
    );

    assert_eq!(
        mamba_rs::mamba_ssm::gpu::device::GpuDevice::resolve_nvrtc_target((11, 0)),
        Ok("sm_110a"),
        "SM110 runs its arch-specific target so the Fixed tcgen05 rung is in its PTX"
    );
    assert_contains_all(
        SM100_SOURCE,
        &["__CUDA_ARCH__ == 1100", "sm100_tf32_tcgen05_k8"],
        "SM110 TCGEN05 source guard",
    );
    let sm110_probe = braced_scope_after(MODULE_SOURCE, "fn probe_sm100_target");
    assert_contains_all(
        sm110_probe,
        &["compile_ptx_with_opts", "load_module", "load_function"],
        "SM110 compile/load/opcode target probe",
    );
    assert_code_contains_all(
        braced_scope_after(
            MODULE_SOURCE,
            "fn sm110_feature_candidates_exclude_ordinary_sm110",
        ),
        &[
            "CudaTarget::Compute110f",
            "CudaTarget::Sm110f",
            "CudaTarget::Compute110a",
            "CudaTarget::Sm110a",
            "assert_eq!",
        ],
        "behavioral SM110 feature-candidate test",
    );
}

#[test]
fn tf32_forced_routes_fail_instead_of_falling_back() {
    assert_contains_all(
        CONTRACT_SOURCE,
        &[
            "MmaTf32Rna",
            "Sm90aWgmmaTf32Tma",
            "Sm100Tcgen05Tf32Tma",
            "Sm120TmaMmaTf32Rna",
        ],
        "forced TF32 route contracts",
    );
    let forced = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_tf32_forced",
    ));
    assert_contains_all(
        &forced,
        &[
            "request: F32TriadRequest",
            "availability: F32TriadAvailability",
            "route: Tf32PhysicalRoute",
            "Result<Tf32PhysicalRoute, String>",
        ],
        "forced TF32 resolver",
    );
    assert!(
        !forced.contains("Option<")
            && !forced.contains("ScalarFma")
            && !forced.contains("resolve_f32_triad_auto"),
        "forced TF32 admission must return Err rather than fallback"
    );
}

#[test]
fn task14_rust_api_never_exceeds_seven_arguments() {
    for (label, source) in [
        ("context", CONTEXT_SOURCE),
        ("identity", IDENTITY_SOURCE),
        ("triad contract", CONTRACT_SOURCE),
        ("triad dispatch", DISPATCH_SOURCE),
        ("triad launch", LAUNCH_SOURCE),
        ("triad modules", MODULE_SOURCE),
        ("triad qualification", QUALIFICATION_SOURCE),
        ("graph capture", GRAPH_CAPTURE_SOURCE),
        ("M1 training graph", TRAINING_GRAPH_SOURCE),
        ("M1 trainer", TRAINER_SOURCE),
        ("M1 inference", INFERENCE_SOURCE),
        ("M1 prefill", PREFILL_SOURCE),
        ("M3 training graph", MAMBA3_TRAINING_GRAPH_SOURCE),
        ("M3 trainer", MAMBA3_TRAINER_SOURCE),
        ("M3 prefill", MAMBA3_PREFILL_SOURCE),
    ] {
        assert_no_rust_function_exceeds_seven(source, label);
        assert!(
            !source.contains("allow(clippy::too_many_arguments)"),
            "{label} may not suppress the seven-argument contract"
        );
    }
}

#[test]
fn rust_argument_scanner_handles_generics_raw_strings_and_lifetimes() {
    let fixture = r#####"
        // fn comment(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8, g: u8, h: u8) {}
        const NORMAL: &str = "fn normal(a,b,c,d,e,f,g,h) { }";
        const RAW: &str = r###"fn raw(a,b,c,d,e,f,g,h) { }"###;
        const BYTE_RAW: &[u8] = br##"fn byte_raw(a,b,c,d,e,f,g,h) { }"##;
        const C_RAW: &CStr = cr#"fn c_raw(a,b,c,d,e,f,g,h) { }"#;
        const fnord: usize = 0;
        fn generic<'a, T: Fn(&'a str, fn(u8, u8)) -> Result<(), E>, E>(
            first: T,
            second: &'a str,
            tuple: (u8, u8),
            array: [u8; { 1 + 2 }],
            callback: fn(u8, u8),
            character: char,
            trailing: u8,
        ) {
            'retry: loop { break 'retry; }
            let _quote = '\'';
        }
        fn eight(a:u8,b:u8,c:u8,d:u8,e:u8,f:u8,g:u8,h:u8) {}
    "#####;
    let mask = source_mask(fixture);
    assert!(!mask.contains("fn comment") && !mask.contains("fn raw"));
    assert!(mask.contains("'a") && mask.contains("'retry"));
    assert_eq!(function_parameter_count(&mask, "fn"), 7);
    let generic = mask.find("fn generic").expect("generic fixture");
    assert_eq!(function_parameter_count(&mask[generic..], "fn"), 7);
    let eight = mask.find("fn eight").expect("eight fixture");
    assert_eq!(function_parameter_count(&mask[eight..], "fn"), 8);
    assert_no_rust_function_exceeds_seven(&mask[..eight], "lexer fixture");
}

#[test]
fn method_scanner_matches_the_exact_name_token() {
    let fixture = r#"
        struct Holder;
        impl Holder {
            fn step_gpu_only(&self) { let _ = GPU_ONLY; }
            fn step(&self) { let _ = STEP_EXACT; }
        }
    "#;
    let scope = source_mask(method_scope_for_type(fixture, "Holder", "step"));
    assert!(scope.contains("STEP_EXACT"));
    assert!(!scope.contains("GPU_ONLY"));
}

#[test]
fn exact_item_scanner_rejects_nested_suffix_and_duplicate_spoofs() {
    let fixture = r#"
        mod cache_and_header_tests_suffix { fn target() { let _ = SUFFIX; } }
        mod outer { mod cache_and_header_tests { fn target() { let _ = NESTED; } } }
        discard!(mod cache_and_header_tests { fn target() { let _ = PAREN_MACRO; } });
        discard![mod cache_and_header_tests { fn target() { let _ = BRACKET_MACRO; } }];
        mod cache_and_header_tests { fn target() { let _ = EXACT; } }
    "#;
    let exact =
        unique_named_item_scope_at_depth(fixture, "mod", "cache_and_header_tests", 0).unwrap();
    assert!(source_mask(exact).contains("EXACT"));
    assert!(!source_mask(exact).contains("SUFFIX"));
    assert!(!source_mask(exact).contains("NESTED"));
    assert!(!source_mask(exact).contains("PAREN_MACRO"));
    assert!(!source_mask(exact).contains("BRACKET_MACRO"));

    let duplicate = format!("{fixture}\nmod cache_and_header_tests {{ fn target() {{}} }}");
    assert!(
        unique_named_item_scope_at_depth(&duplicate, "mod", "cache_and_header_tests", 0).is_err()
    );
    assert!(
        unique_named_item_scope_at_depth(
            "mod outer { mod cache_and_header_tests {} }",
            "mod",
            "cache_and_header_tests",
            0,
        )
        .is_err()
    );
    assert!(
        unique_named_item_scope_at_depth(
            "mod cache_and_header_tests; fn unrelated() {}",
            "mod",
            "cache_and_header_tests",
            0,
        )
        .is_err()
    );

    let function_fixture = r#"
        mod cache_and_header_tests {
            discard!(fn target() { let _ = PAREN_FUNCTION; });
            discard![fn target() { let _ = BRACKET_FUNCTION; }];
            fn target() { let _ = LIVE_FUNCTION; }
        }
    "#;
    let module =
        unique_named_item_scope_at_depth(function_fixture, "mod", "cache_and_header_tests", 0)
            .unwrap();
    let function = unique_named_item_scope_at_depth(module, "fn", "target", 1).unwrap();
    assert!(source_mask(function).contains("LIVE_FUNCTION"));
    assert!(!source_mask(function).contains("PAREN_FUNCTION"));
    assert!(!source_mask(function).contains("BRACKET_FUNCTION"));
}

#[test]
fn exact_item_scanner_finds_the_body_after_a_macro_return_type() {
    let fixture = r#"
        macro_rules! proof_type {
            ($($proof:tt)*) => { () };
        }

        fn target() -> proof_type! { DISCARDED_PROOF_TOKENS } {
            let _ = LIVE_FUNCTION_BODY;
        }
    "#;
    let function = unique_named_item_scope_at_depth(fixture, "fn", "target", 0).unwrap();
    let function = source_mask(function);
    assert!(function.contains("LIVE_FUNCTION_BODY"));
}

#[test]
fn active_test_item_scanner_rejects_disabled_cfg_spoofs() {
    let active = r#"
        #[cfg(test)]
        mod cache_and_header_tests {
            #[test]
            fn target() {}
        }
    "#;
    let module = active_test_module_scope(active, "cache_and_header_tests").unwrap();
    direct_test_function_scope(module, "target", "#[test]").unwrap();

    let disabled_module = active.replace("#[cfg(test)]", "#[cfg(any())]");
    assert!(active_test_module_scope(&disabled_module, "cache_and_header_tests").is_err());
    let stacked_module = active.replace("#[cfg(test)]", "#[cfg(any())]\n#[cfg(test)]");
    assert!(active_test_module_scope(&stacked_module, "cache_and_header_tests").is_err());

    let disabled_function = active.replace(
        "#[test]\n            fn",
        "#[test]\n            #[cfg(any())]\n            fn",
    );
    let module = active_test_module_scope(&disabled_function, "cache_and_header_tests").unwrap();
    assert!(direct_test_function_scope(module, "target", "#[test]").is_err());

    let ignored_function = active.replace(
        "#[test]\n            fn",
        "#[test]\n            #[ignore]\n            fn",
    );
    let module = active_test_module_scope(&ignored_function, "cache_and_header_tests").unwrap();
    assert!(direct_test_function_scope(module, "target", "#[test]").is_err());

    let panicking_function = active.replace(
        "#[test]\n            fn",
        "#[test]\n            #[should_panic]\n            fn",
    );
    let module = active_test_module_scope(&panicking_function, "cache_and_header_tests").unwrap();
    assert!(direct_test_function_scope(module, "target", "#[test]").is_err());

    active_production_function_scope("fn target() {}", "target").unwrap();
    assert!(active_production_function_scope("#[cfg(any())] fn target() {}", "target").is_err());
    assert!(
        active_production_function_scope(
            "#[cfg(any())] fn target() {} fn r#target() {}",
            "target",
        )
        .is_err()
    );
}

#[test]
fn active_item_scanner_rejects_spaced_attributes_visibility_and_inner_cfg() {
    let disabled_public = r#"
        macro_rules! live_wrong {
            () => {
                fn target() { let _ = LIVE_WRONG_FUNCTION; }
            };
        }
        live_wrong!();

        # [cfg(any())]
        pub(crate) fn target() { let _ = DISABLED_CORRECT_FUNCTION; }
    "#;
    assert!(active_production_function_scope(disabled_public, "target").is_err());

    let disabled_inner_module = r#"
        macro_rules! live_wrong {
            () => {
                #[cfg(test)]
                mod cache_and_header_tests {
                    #[test]
                    fn target() { let _ = LIVE_WRONG_TEST; }
                }
            };
        }
        live_wrong!();

        #[cfg(test)]
        mod cache_and_header_tests {
            #![cfg(any())]

            #[test]
            fn target() { let _ = DISABLED_CORRECT_TEST; }
        }
    "#;
    assert!(active_test_module_scope(disabled_inner_module, "cache_and_header_tests").is_err());
}

#[test]
fn nonzero_input_guard_oracle_rejects_nested_and_after_guard_reads() {
    let canonical = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
        }
    "#;
    assert!(validate_nonzero_input_pointer_guard(canonical).is_ok());

    let after_guard = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
            let _ = operands.a;
        }
    "#;
    assert!(validate_nonzero_input_pointer_guard(after_guard).is_err());

    let nested = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if ready {
                if request.shape.reduction(request.op) != 0 {
                    for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                        if pointer == 0 || !pointer.is_multiple_of(alignment) {
                            return Err(format!("input pointer {name}"));
                        }
                    }
                }
            }
            let _ = operands.a;
        }
    "#;
    assert!(validate_nonzero_input_pointer_guard(nested).is_err());

    let alias = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            let copy = operands;
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
            let _ = (copy.a, copy.b);
        }
    "#;
    assert!(validate_nonzero_input_pointer_guard(alias).is_err());

    let destructured = canonical.replace(
        "if request.shape.reduction(request.op) != 0",
        "let Operands { a, b, .. } = operands;\n            if request.shape.reduction(request.op) != 0",
    );
    assert!(validate_nonzero_input_pointer_guard(&destructured).is_err());

    let raw_identifier = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
            let _ = r#operands.a;
        }
    "#;
    assert!(validate_nonzero_input_pointer_guard(raw_identifier).is_err());

    let raw_item_spoof = format!(
        "#[cfg(any())]\n{canonical}\nfn r#validate_f32_triad_operands() {{ let _ = BAD; }}"
    );
    assert!(validate_nonzero_input_pointer_guard(&raw_item_spoof).is_err());
}

#[test]
fn nonzero_input_guard_oracle_rejects_disabled_and_token_separated_reads() {
    let canonical = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
        }
    "#;
    let disabled_guard = canonical.replace(
        "if request.shape.reduction(request.op) != 0",
        "#[cfg(any())]\n            if request.shape.reduction(request.op) != 0",
    );
    let separated_read = r#"
        fn validate_f32_triad_operands(request: Request, operands: Operands) {
            if request.shape.reduction(request.op) != 0 {
                for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
                    if pointer == 0 || !pointer.is_multiple_of(alignment) {
                        return Err(format!("input pointer {name}"));
                    }
                }
            }
            let _ = operands /* live gap */ .a;
        }
    "#;
    let disabled_rejected = validate_nonzero_input_pointer_guard(&disabled_guard).is_err();
    let separated_rejected = validate_nonzero_input_pointer_guard(separated_read).is_err();
    assert!(
        disabled_rejected && separated_rejected,
        "disabled guard rejected={disabled_rejected}, token-separated read rejected={separated_rejected}"
    );
}

#[test]
fn driver_abi_lookup_oracle_rejects_dead_correct_and_live_wrong_calls() {
    let canonical = MODULE_SOURCE;
    assert!(validate_cuda12_driver_abi_lookup_contract(canonical).is_ok());

    let dead_correct = canonical.replace(
        "unsafe { std::mem::transmute(driver_proc_address(\"cuFuncGetParamInfo\", 12_040)?) }",
        "if false { unsafe { std::mem::transmute(driver_proc_address(\"cuFuncGetParamInfo\", 12_040)?) } } else { unsafe { std::mem::transmute(driver_proc_address(\"wrong\", 13_000)?) } }",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&dead_correct).is_err());

    let live_wrong = canonical.replace(
        "driver_proc_address(\"cuFuncGetParamInfo\", 12_040)",
        "driver_proc_address(\"wrong\", 13_000)",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&live_wrong).is_err());

    let direct_downstream = r#"let abi = query_tf32_driver_parameter_abi(
            &label,
            tf32_driver_parameter_count(module_kind, symbol),
            |index, offset, size| unsafe { get_parameter_info(function, index, offset, size) },
        )?;"#;
    let dead_downstream = canonical.replace(
        direct_downstream,
        &format!(
            "if false {{ {direct_downstream} }}\n                let abi = synthesize_driver_abi();"
        ),
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&dead_downstream).is_err());

    let shadowed_abi = canonical.replace(
        "if census.insert(symbol, abi).is_some()",
        "black_box(&abi);\n                let abi = synthesize_driver_abi();\n                if census.insert(symbol, abi).is_some()",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&shadowed_abi).is_err());

    let prefix_header =
        canonical.replace("for symbol in symbols {", "for symbol in symbols_live {");
    assert!(validate_cuda12_driver_abi_lookup_contract(&prefix_header).is_err());

    let wrong_terminal = canonical.replacen(
        "if extra != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE",
        "if extra != cudarc::driver::sys::CUresult::CUDA_SUCCESS",
        1,
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&wrong_terminal).is_err());

    let synthetic_tail = canonical.replace(
        "Tf32DriverAbi::checked(parameters.len(), parameters)\n        .map_err(|error| format!(\"{label}: {error}\"))",
        "let _ = Tf32DriverAbi::checked(parameters.len(), parameters);\n    Ok(synthesize_driver_abi())",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&synthetic_tail).is_err());

    let empty_symbols = canonical.replace(
        "    let symbols: Vec<&'static str> = super::contract::tf32_route_specs_for(module_kind, extensions)\n        .map(|spec| spec.symbol)\n        .collect();",
        "let symbols: Vec<&'static str> = Vec::new();",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&empty_symbols).is_err());

    let cleared_census = canonical.replace(
        "module.unload()?;",
        "census.clear();\n    module.unload()?;",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&cleared_census).is_err());

    let early_terminal = canonical.replace(
        "let mut extra_offset = 0;",
        "return Ok(synthesize_driver_abi());\n    let mut extra_offset = 0;",
    );
    assert!(validate_cuda12_driver_abi_lookup_contract(&early_terminal).is_err());

    let raw_item_spoof = canonical.replacen(
        "fn census_tf32_driver_abi(",
        "#[cfg(any())]\nfn census_tf32_driver_abi(",
        1,
    ) + "\nfn r#census_tf32_driver_abi() { synthesize_driver_abi() }";
    assert!(validate_cuda12_driver_abi_lookup_contract(&raw_item_spoof).is_err());
}

#[test]
fn driver_abi_lookup_oracle_rejects_mutated_live_helper_bodies() {
    let proc_address_ignores_symbol = MODULE_SOURCE.replace(
        "let symbol = CString::new(symbol).expect(\"static CUDA Driver symbol\");",
        "let symbol = CString::new(\"cuFuncGetParamInfo\").expect(\"static CUDA Driver symbol\");",
    );
    let driver_call_accepts_invalid_value = MODULE_SOURCE.replace(
        "if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {",
        "if result == cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE {",
    );
    let checked_synthesizes_parameters = MODULE_SOURCE.replace(
        "parameters: checked.into_boxed_slice(),",
        "parameters: Vec::new().into_boxed_slice(),",
    );
    let proc_address_rejected =
        validate_cuda12_driver_abi_lookup_contract(&proc_address_ignores_symbol).is_err();
    let driver_call_rejected =
        validate_cuda12_driver_abi_lookup_contract(&driver_call_accepts_invalid_value).is_err();
    let checked_rejected =
        validate_cuda12_driver_abi_lookup_contract(&checked_synthesizes_parameters).is_err();
    assert!(
        proc_address_rejected && driver_call_rejected && checked_rejected,
        "driver_proc_address rejected={proc_address_rejected}, driver_call rejected={driver_call_rejected}, Tf32DriverAbi::checked rejected={checked_rejected}"
    );
}

#[test]
fn exact_cuda_scope_matcher_rejects_inserted_executable_prefixes() {
    let expected = "epilogue(floataccumulator){returnaccumulator;}";
    let canonical = "float epilogue(float accumulator) { return accumulator; }";
    assert_eq!(
        compact_cuda_executable_scope(canonical, "epilogue"),
        expected
    );

    for mutated in [
        "float epilogue(float accumulator) { float early = accumulator * 2.0f; return accumulator; }",
        "float epilogue(float accumulator) { if (accumulator == 0.0f) return 1.0f; return accumulator; }",
    ] {
        assert_ne!(compact_cuda_executable_scope(mutated, "epilogue"), expected);
    }
}

#[test]
fn graph_launch_scanner_rejects_a_bypass_beside_a_guarded_call() {
    let guarded = r#"
        fn replay(&self, ctx: &GpuCtx) -> Result<(), String> {
            self.plan.with_validated_launch(ctx, "replay", || {
                self.graph.launch()
            })?;
            Ok(())
        }
    "#;
    let bypass = r#"
        fn replay(&self, ctx: &GpuCtx) -> Result<(), String> {
            self.plan.with_validated_launch(ctx, "replay", || Ok(()))?;
            self.graph.launch()?;
            Ok(())
        }
    "#;
    assert!(graph_launches_are_guarded(guarded));
    assert!(!graph_launches_are_guarded(bypass));
    let model_guarded = guarded.replace(
        "self.plan.with_validated_launch",
        "with_validated_gemm_graph_launch",
    );
    let model_bypass = bypass.replace(
        "self.plan.with_validated_launch",
        "with_validated_gemm_graph_launch",
    );
    assert!(graph_launches_are_guarded(&model_guarded));
    assert!(!graph_launches_are_guarded(&model_bypass));
    let impostor = model_guarded.replace(
        "with_validated_gemm_graph_launch",
        "unchecked_with_validated_gemm_graph_launch",
    );
    assert!(!graph_launches_are_guarded(&impostor));
    for receiver in ["graph", "g"] {
        let guarded_local =
            model_guarded.replace("self.graph.launch()", &format!("{receiver}\n .launch()"));
        let bypass_local =
            model_bypass.replace("self.graph.launch()", &format!("{receiver}\n .launch()"));
        assert!(graph_launches_are_guarded(&guarded_local));
        assert!(!graph_launches_are_guarded(&bypass_local));
    }
}

#[test]
fn k0_cfg_checker_propagates_values_and_rejects_bad_zero_subgraphs() {
    assert!(!contains_opcode_prefix("ld.shared.f32 %f1, [%r1];", "red."));
    assert!(contains_opcode_prefix(
        "red.global.add.u32 [%r1], %r2;",
        "red."
    ));
    assert!(!contains_float_mad_opcode("mad.lo.s32 %r1, %r2, %r3, %r4;"));
    assert!(contains_float_mad_opcode("mad.rn.f32 %f1, %f2, %f3, %f4;"));
    let symbol = "nn_sm100_tcgen_tf32_m128n64_bk32_s2_c4";
    let valid = format!(
        r#"
.visible .entry {symbol}(
    .param .align 4 .b8 {symbol}_param_4[40]
)
{{
    ld.param.u32 %r1, [{symbol}_param_4+28];
    mov.u32 %r2, %r1;
    cvt.u32.u32 %r3, %r2;
    setp.eq.u32 %p1, %r3, 0;
    @%p1 bra K0;
MAIN:
    cp.async.bulk.tensor.2d.shared::cta.global.tile [%r4], [%r5];
    tcgen05.mma.cta_group::1.kind::tf32 [%r6], %r7, %r8, %r9;
    bra.uni DONE;
K0:
    mul.rn.f32 %f1, %f2, %f3;
    fma.rn.f32 %f4, %f1, %f5, %f6;
    st.global.f32 [%rd1], %f4;
    ret;
DONE:
    ret;
DEAD:
    bar.sync 0;
    bra MAIN;
}}
"#
    );
    assert_k0_cfg_dominates_entry(&valid, symbol, 40);
    let unreachable_overwrite = valid.replacen(
        "    setp.eq.u32 %p1, %r3, 0;",
        "    bra GUARD;\n    add.u32 %r3, %r4, 1;\nGUARD:\n    setp.eq.u32 %p1, %r3, 0;",
        1,
    );
    assert_k0_cfg_dominates_entry(&unreachable_overwrite, symbol, 40);
    let trap = valid.replacen("    ret;\nDONE:", "    trap;\nDONE:", 1);
    assert!(std::panic::catch_unwind(|| assert_k0_cfg_dominates_entry(&trap, symbol, 40)).is_err());
    let bad_fp = valid.replacen(
        "    mul.rn.f32",
        "    add.rn.f32 %f7, %f2, %f3;\n    mul.rn.f32",
        1,
    );
    assert!(
        std::panic::catch_unwind(|| assert_k0_cfg_dominates_entry(&bad_fp, symbol, 40)).is_err()
    );
    let duplicate_guard = valid.replacen(
        "    @%p1 bra K0;",
        "    @%p1 bra K0;\n    setp.ne.u32 %p2, %r3, 0;\n    @!%p2 bra K0;",
        1,
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_k0_cfg_dominates_entry(&duplicate_guard, symbol, 40)
        })
        .is_err()
    );
    let predicated_setp = valid.replacen(
        "    setp.eq.u32 %p1, %r3, 0;",
        "    @%p9 setp.eq.u32 %p1, %r3, 0;",
        1,
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_k0_cfg_dominates_entry(&predicated_setp, symbol, 40)
        })
        .is_err()
    );
    let overwritten_predicate = valid.replacen(
        "    @%p1 bra K0;",
        "    setp.eq.u32 %p1, %r4, 0;\n    @%p1 bra K0;",
        1,
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_k0_cfg_dominates_entry(&overwritten_predicate, symbol, 40)
        })
        .is_err()
    );
}

#[test]
fn sass_line_parser_skips_the_unmapped_compiler_prologue() {
    let entry = r#"
        /*0000*/ MOV R1, c[0x0][0x28] ;
        //## File "mamba_tf32_k0_guard", line 1001
        /*0010*/ ISETP.EQ.AND P0, PT, R2, RZ, PT ;
    "#;
    let instructions = sass_line_instructions(entry, "self-oracle");
    assert_eq!(instructions.len(), 1);
    assert_eq!(instructions[0].offset, 0x10);
    assert_eq!(instructions[0].file, "mamba_tf32_k0_guard");
    assert_eq!(instructions[0].line, 1001);
}

#[test]
fn sass_guard_parser_accepts_uniform_predicate_branches() {
    let entry = r#"
        //## File "mamba_tf32_k0_guard", line 1001
        /*0000*/ UISETP.NE.U32.AND UP0, UPT, UR5, URZ, UPT ;
        /*0010*/ BRA.U UP0, `(.L_main) ;
        /*0020*/ UISETP.NE.U32.AND UP1, UPT, UR6, URZ, UPT ;
        /*0030*/ BRA.U !UP1, `(.L_retry) ;
    "#;
    let instructions = sass_line_instructions(entry, "self-oracle");
    assert_eq!(
        sass_guard_candidates(&instructions),
        [(0x0, 0x10), (0x20, 0x30)]
    );
    assert_eq!(
        sass_branch_predicate(&instructions[1]),
        Some(("UP0", false))
    );
    assert_eq!(sass_branch_predicate(&instructions[3]), Some(("UP1", true)));
}

#[test]
fn sass_cfg_removes_only_tcgen_guardrail_trap_fallthroughs() {
    let nodes = BTreeMap::from([
        (
            "trap".to_owned(),
            "CALL.REL.NOINC $__cuda_sm10x_tcgen05_guardrail_trap_phase_invalid_during_alloc"
                .to_owned(),
        ),
        ("ordinary".to_owned(), "CALL.REL.NOINC helper".to_owned()),
        ("next".to_owned(), "EXIT".to_owned()),
    ]);
    let successors = BTreeMap::from([
        ("trap".to_owned(), vec!["next".to_owned()]),
        ("ordinary".to_owned(), vec!["next".to_owned()]),
        ("next".to_owned(), Vec::new()),
    ]);
    let normal = sass_normal_successors(&nodes, &successors, "self-oracle");
    assert!(normal["trap"].is_empty());
    assert_eq!(normal["ordinary"], ["next"]);
}

#[test]
fn dot_parser_keeps_nodes_adjacent_to_unterminated_edge_records() {
    let graph = r#"
subgraph "cluster_self-oracle" {
"entry"
[label="{<entry>0000: ISETP ;|<exit0>0010: BRA ;}"]
"entry":exit0:e -> "done":entry:n [style=solid];
"done"
[label="{<entry>|<exit0>0020: EXIT ;}"]
}
"#;
    let (nodes, successors) = parse_dot_cfg(graph, "self-oracle");
    assert_eq!(nodes.len(), 2);
    assert_eq!(successors["entry"], ["done"]);
    assert!(successors["done"].is_empty());
}

#[test]
fn dot_instruction_offsets_ignore_hexadecimal_basic_block_labels() {
    let body = r#"[label="{<entry>.L_x_1150:\l3610:\ \ \ MOV\ R2,\ 0x3630\ ;\l|<exit0>3620:\ \ \ EXIT\ ;\l}"]"#;
    assert_eq!(
        dot_instruction_offsets(body),
        BTreeSet::from([0x3610, 0x3620])
    );
}

#[test]
fn k0_provenance_kills_overwritten_registers() {
    let symbol = "nn_sm100_tcgen_tf32_m128n64_bk32_s2_c4";
    let overwritten = format!(
        r#"
.visible .entry {symbol}(
    .param .align 4 .b8 {symbol}_param_4[40]
)
{{
    ld.param.u32 %r1, [{symbol}_param_4+28];
    mov.u32 %r2, %r1;
    add.u32 %r2, %r4, 1;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra K0;
MAIN:
    cp.async.bulk.tensor.2d.shared::cta.global.tile [%r4], [%r5];
    tcgen05.mma.cta_group::1.kind::tf32 [%r6], %r7, %r8, %r9;
    bra DONE;
K0:
    mul.rn.f32 %f1, %f2, %f3;
    fma.rn.f32 %f4, %f1, %f5, %f6;
    st.global.f32 [%rd1], %f4;
    ret;
DONE:
    ret;
}}
"#
    );
    assert!(
        std::panic::catch_unwind(|| { assert_k0_cfg_dominates_entry(&overwritten, symbol, 40) })
            .is_err()
    );

    let branch_overwrite = format!(
        r#"
.visible .entry {symbol}(
    .param .align 4 .b8 {symbol}_param_4[40]
)
{{
    ld.param.u32 %r1, [{symbol}_param_4+28];
    @%p9 bra PRESERVE;
    add.u32 %r2, %r4, 1;
    bra JOIN;
PRESERVE:
    mov.u32 %r2, %r1;
JOIN:
    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra K0;
MAIN:
    cp.async.bulk.tensor.2d.shared::cta.global.tile [%r4], [%r5];
    tcgen05.mma.cta_group::1.kind::tf32 [%r6], %r7, %r8, %r9;
    bra DONE;
K0:
    mul.rn.f32 %f1, %f2, %f3;
    fma.rn.f32 %f4, %f1, %f5, %f6;
    st.global.f32 [%rd1], %f4;
    ret;
DONE:
    ret;
}}
"#
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_k0_cfg_dominates_entry(&branch_overwrite, symbol, 40)
        })
        .is_err()
    );
    let predicated_copy = overwritten.replace(
        "    mov.u32 %r2, %r1;\n    add.u32 %r2, %r4, 1;",
        "    @%p9 mov.u32 %r2, %r1;",
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_k0_cfg_dominates_entry(&predicated_copy, symbol, 40)
        })
        .is_err()
    );
}

#[test]
fn sass_cfg_checker_requires_the_guarded_zero_partition() {
    // The strict tcgen management contract holds from CUDA 12.9; below it the
    // checker skips that family, so this self-test has nothing to reject.
    if loaded_nvrtc_version() < (12, 9) {
        eprintln!("sass_cfg_checker_requires_the_guarded_zero_partition: skipped below CUDA 12.9");
        return;
    }
    let symbol = "nn_sm100_tcgen_tf32_m128n64_bk32_s2_c4";
    let source = "#line 1001 \"mamba_tf32_k0_guard\"\n#line 1002 \"mamba_tf32_k0_branch\"\n#line 2001 \"mamba_tf32_k0_zero_store\"\n";
    let valid = format!(
        r#"digraph "{symbol}" {{
"entry" [label="0000: ISETP; 0010: @P0 BRA;"];
"entry" -> "zero";
"entry" -> "main";
"zero" [label="0020: STG; 0030: EXIT;"];
"main" [label="0040: UTMALDG; 0050: UTCATOMSWS.FIND_AND_SET.ALIGN; 0060: ATOMS.OR; 0070: ATOMS.OR; 0078: BAR.SYNC; 0080: UTCHMMA; 0090: STTM; 00a0: LDTM;"];
"main" -> "release";
"release" [label="00b0: BAR.SYNC; 00c0: UVIRTCOUNT.DEALLOC.SMPOOL; 00d0: UTCATOMSWS.AND; 00e0: EXIT;"];
}}"#
    );
    let line_sass = format!(
        "Function : {symbol}\n\
         //## File \"/root/mamba_tf32_k0_guard\", line 1001\n\
         /*0000*/ ISETP.NE.AND P0, PT, R1, RZ, PT ;\n\
         //## File \"/root/mamba_tf32_k0_branch\", line 1002\n\
         /*0010*/ @P0 BRA `(.L_zero) ;\n\
         //## File \"/root/mamba_tf32_k0_zero_store\", line 2001\n\
         /*0020*/ STG.E [R2.64], R3 ;\n\
         //## File \"/root/tf32.cu\", line 10\n\
         /*0030*/ EXIT ;\n\
         /*0040*/ UTMALDG.2D ;\n\
         /*0050*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;\n\
         /*0060*/ ATOMS.OR RZ, [R7+0x14], R8 ;\n\
         /*0070*/ ATOMS.OR RZ, [R7+0x18], R9 ;\n\
         /*0078*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;\n\
         /*0080*/ UTCHMMA.16816 ;\n\
         /*0090*/ STTM.x8 tmem[UR5], R8 ;\n\
         /*00a0*/ LDTM.x8 R8, tmem[UR5] ;\n\
         /*00b0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;\n\
         /*00c0*/ UVIRTCOUNT.DEALLOC.SMPOOL 0x80 ;\n\
         /*00d0*/ UTCATOMSWS.AND URZ, UR4 ;\n\
         /*00e0*/ EXIT ;\n"
    );
    assert_sass_cfg_corroboration(&valid, &line_sass, source, symbol, "SM100");
    let bypass = valid.replace("0020: STG;", "0020: UTCHMMA; 0028: STG;");
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&bypass, &line_sass, source, symbol, "SM100")
        })
        .is_err()
    );
    let overwritten_predicate_sass = line_sass.replace(
        "/*0010*/ @P0 BRA",
        "/*0008*/ PLOP3.LUT P0, PT, P1, P2, PT, 0x80, 0x0 ;\n\
         /*0010*/ @P0 BRA",
    );
    let overwritten_predicate_dot = valid.replace(
        "0000: ISETP; 0010: @P0 BRA;",
        "0000: ISETP; 0008: PLOP3.LUT; 0010: @P0 BRA;",
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(
                &overwritten_predicate_dot,
                &overwritten_predicate_sass,
                source,
                symbol,
                "SM100",
            )
        })
        .is_err()
    );
    let not_dominated = format!(
        r#"digraph "{symbol}" {{
"entry" [label="0000: ISETP; 0010: @P0 BRA;"];
"entry" -> "zero";
"entry" -> "dispatch";
"zero" [label="0020: STG; 0030: EXIT;"];
"dispatch" [label="0040: UTMALDG; 0048: BRA;"];
"dispatch" -> "find";
"dispatch" -> "alloc";
"find" [label="0050: UTCATOMSWS.FIND_AND_SET.ALIGN;"];
"find" -> "alloc";
"alloc" [label="0060: ATOMS.OR; 0070: ATOMS.OR; 0080: UTCHMMA;"];
"alloc" -> "dealloc";
"dealloc" [label="0090: UTCATOMSWS.AND; 00a0: EXIT;"];
}}"#
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&not_dominated, &line_sass, source, symbol, "SM100")
        })
        .is_err()
    );
    let bad_deallocation = valid.replace("00c0: UVIRTCOUNT.DEALLOC.SMPOOL;", "00c0: NOP;");
    let bad_deallocation_sass = line_sass.replace(
        "/*00c0*/ UVIRTCOUNT.DEALLOC.SMPOOL 0x80 ;",
        "/*00c0*/ NOP ;",
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(
                &bad_deallocation,
                &bad_deallocation_sass,
                source,
                symbol,
                "SM100",
            )
        })
        .is_err()
    );
    let matrix_bypasses_allocation = format!(
        r#"digraph "{symbol}" {{
"entry" [label="0000: ISETP; 0010: @P0 BRA;"];
"entry" -> "zero";
"entry" -> "dispatch";
"zero" [label="0020: STG; 0030: EXIT;"];
"dispatch" [label="0040: UTMALDG; 0050: UTCATOMSWS.FIND_AND_SET.ALIGN; 0058: BRA;"];
"dispatch" -> "alloc";
"dispatch" -> "bypass";
"alloc" [label="0060: ATOMS.OR; 0070: ATOMS.OR;"];
"alloc" -> "matrix";
"bypass" [label="0078: UTCHMMA;"];
"bypass" -> "matrix";
"matrix" [label="0080: UTCHMMA;"];
"matrix" -> "dealloc";
"dealloc" [label="0090: UTCATOMSWS.AND; 00a0: EXIT;"];
}}"#
    );
    let bypass_sass = line_sass
        .replace(
            "/*0050*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;",
            "/*0050*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;\n         /*0058*/ BRA `(.L_alloc) ;",
        )
        .replace(
            "/*0080*/ UTCHMMA.16816 ;",
            "/*0078*/ UTCHMMA.16816 ;\n         /*0080*/ UTCHMMA.16816 ;",
        );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(
                &matrix_bypasses_allocation,
                &bypass_sass,
                source,
                symbol,
                "SM100",
            )
        })
        .is_err()
    );
    let exit_skips_deallocation = format!(
        r#"digraph "{symbol}" {{
"entry" [label="0000: ISETP; 0010: @P0 BRA;"];
"entry" -> "zero";
"entry" -> "main";
"zero" [label="0020: STG; 0030: EXIT;"];
"main" [label="0040: UTMALDG; 0050: UTCATOMSWS.FIND_AND_SET.ALIGN; 0060: ATOMS.OR; 0070: ATOMS.OR; 0080: UTCHMMA; 0088: BRA;"];
"main" -> "dealloc";
"main" -> "escaped";
"dealloc" [label="0090: UTCATOMSWS.AND;"];
"dealloc" -> "done";
"escaped" [label="00a0: EXIT;"];
"done" [label="00b0: EXIT;"];
}}"#
    );
    let escaped_sass = line_sass
        .replace(
            "/*0080*/ UTCHMMA.16816 ;",
            "/*0080*/ UTCHMMA.16816 ;\n         /*0088*/ BRA `(.L_done) ;",
        )
        .replace(
            "/*00a0*/ EXIT ;",
            "/*00a0*/ EXIT ;\n         /*00b0*/ EXIT ;",
        );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(
                &exit_skips_deallocation,
                &escaped_sass,
                source,
                symbol,
                "SM100",
            )
        })
        .is_err()
    );
}

#[test]
fn sass_atomic_parser_rejects_management_lookalikes() {
    // The strict tcgen management contract holds from CUDA 12.9; below it the
    // checker skips that family, so this self-test has nothing to reject.
    if loaded_nvrtc_version() < (12, 9) {
        eprintln!("sass_atomic_parser_rejects_management_lookalikes: skipped below CUDA 12.9");
        return;
    }
    let symbol = "nn_sm100_tcgen_tf32_m128n64_bk32_s2_c4";
    let valid = format!(
        "Function : {symbol}\n\
         //## File \"/root/tf32.cu\", line 1\n\
         /*0000*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;\n\
         /*0010*/ ATOMS.OR RZ, [R7+0x14], R8 ;\n\
         /*0020*/ ATOMS.OR RZ, [R7+0x18], R9 ;\n\
         /*0028*/ REDUX UR8, R0 ;\n\
         /*0030*/ UTCATOMSWS.AND URZ, UR4 ;\n\
         /*0034*/ @P0 ATOMS.AND RZ, [UR6+0x14], R2 ;\n\
         /*0038*/ @P0 ATOMS.AND RZ, [UR6+0x18], R3 ;\n\
         /*0040*/ FFMA R0, R1, R2, R3 ;\n\
         /*0050*/ FMUL R0, R1, R2 ;\n"
    );
    assert_sass_entry_contract(&valid, symbol, "SM100");
    let lookalike = valid.replace("ATOMS.OR RZ, [R7+0x14], R8", "ATOMS.OR R1, [R7+0x10], R8");
    assert!(
        std::panic::catch_unwind(|| { assert_sass_entry_contract(&lookalike, symbol, "SM100") })
            .is_err()
    );
    for forbidden in [
        "ATOMG.ADD R0, [R1], R2",
        "ATOMS.AND RZ, [R1], R2",
        "SUATOM.ADD R0, [R1], R2",
        "RED.ADD [R1], R2",
        "REDG.ADD [R1], R2",
        "URED.ADD [UR1], UR2",
        "SURED.ADD [R1], R2",
        "REDUX R0, R1",
        "UTCATOMSWS.XOR URZ, UR4",
    ] {
        let bypass = valid.replace(
            "/*0040*/ FFMA",
            &format!("/*0038*/ {forbidden} ;\n/*0040*/ FFMA"),
        );
        assert!(
            std::panic::catch_unwind(|| { assert_sass_entry_contract(&bypass, symbol, "SM100") })
                .is_err(),
            "accepted {forbidden}"
        );
    }
}

#[test]
fn resource_parser_rejects_duplicate_records_and_cap_overruns() {
    let symbol = "nn_sm80_mma_tf32_m16n32_bk32_s4";
    let symbols = BTreeSet::from([symbol.to_owned()]);
    let ptxas = format!(
        "ptxas info : Compiling entry function '{symbol}' for 'sm_80'\n\
         ptxas info : Function properties for {symbol}\n\
         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n\
         ptxas info : Used 64 registers, 1024 bytes smem\n"
    );
    assert_per_entry_zero_resources(&ptxas, &symbols, "self-oracle");
    let dynamic_shared = ptxas.replace(
        "Used 64 registers, 1024 bytes smem",
        "Used 64 registers, used 1 barriers, 416 bytes cmem[0]",
    );
    assert_per_entry_zero_resources(&dynamic_shared, &symbols, "self-oracle");
    let duplicate = format!("{ptxas}ptxas info : Function properties for {symbol}\n");
    assert!(
        std::panic::catch_unwind(|| {
            assert_per_entry_zero_resources(&duplicate, &symbols, "self-oracle")
        })
        .is_err()
    );
    let cuobjdump =
        format!("Function {symbol}:\nREG:64 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:392\n");
    assert_cuobjdump_zero_resources(&cuobjdump, &symbols, "self-oracle");
    let over_cap = cuobjdump.replace("REG:64", "REG:97");
    assert!(
        std::panic::catch_unwind(|| {
            assert_cuobjdump_zero_resources(&over_cap, &symbols, "self-oracle")
        })
        .is_err()
    );
}

#[test]
fn resource_caps_match_the_frozen_cuda_map() {
    for (symbol, expected) in [
        ("nn_sm80_mma_tf32_m128n64_bk32_s2", (192, 55_296)),
        ("nn_sm80_mma_tf32_m128n64_bk32_s3", (192, 82_944)),
        ("nn_sm80_mma_tf32_m64n64_bk32_s2", (128, 36_864)),
        ("nn_sm80_mma_tf32_m64n64_bk32_s3", (128, 55_296)),
        ("nn_sm80_mma_tf32_m16n32_bk32_s4", (96, 29_696)),
        ("nn_sm80_mma_tf32_m16n16_bk32_s4", (96, 21_504)),
        ("tn_sm80_mma_tf32_m128n64_bk32_s2", (192, 53_248)),
        ("tn_sm80_mma_tf32_m128n64_bk32_s3", (192, 79_872)),
        ("tn_sm80_mma_tf32_m64n64_bk32_s2", (128, 36_864)),
        ("tn_sm80_mma_tf32_m64n64_bk32_s3", (128, 55_296)),
        ("tn_sm80_mma_tf32_m16n32_bk32_s4", (96, 32_768)),
        ("tn_sm80_mma_tf32_m16n16_bk32_s4", (96, 24_576)),
        ("nt_sm80_mma_tf32_m128n64_bk32_s2", (192, 55_296)),
        ("nt_sm80_mma_tf32_m128n64_bk32_s3", (192, 82_944)),
        ("nt_sm80_mma_tf32_m64n64_bk32_s2", (128, 36_864)),
        ("nt_sm80_mma_tf32_m64n64_bk32_s3", (128, 55_296)),
        ("nt_sm80_mma_tf32_m16n32_bk32_s4", (96, 27_648)),
        ("nt_sm80_mma_tf32_m16n16_bk32_s4", (96, 18_432)),
        ("nn_sm90a_wgmma_tf32_m64n128_bk32_s3_wg1", (168, 73_984)),
        ("nn_sm90a_wgmma_tf32_m64n128_bk32_s3_wg2", (128, 73_984)),
        ("nn_sm100_tcgen_tf32_m128n64_bk32_s2_c4", (128, 49_408)),
        ("nn_sm100_tcgen_tf32_m128n64_bk32_s4_p8", (128, 98_560)),
        ("nn_sm100_tcgen_tf32_m128n128_bk32_s3_c4", (128, 98_560)),
        ("nn_sm120_tma_mma_tf32_m128n64_bk32_s2", (128, 49_280)),
        ("nn_sm120_tma_mma_tf32_m64n128_bk32_s3", (128, 73_856)),
        ("tn_sm120_tma_mma_tf32_m64n128_bk32_s4_pair", (128, 98_432)),
        (
            "tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk",
            (255, 73_856),
        ),
    ] {
        assert_eq!(tf32_resource_caps(symbol), expected, "{symbol}");
    }
}

#[test]
fn strict_json_parser_rejects_duplicate_malformed_and_trailing_data() {
    let valid = r#"{"schema":"v1","count":54,"nested":[true,false,null,"\u03bb"]}"#;
    assert!(parse_strict_json(valid).is_ok());
    for invalid in [
        r#"{"schema":"v1","schema":"v2"}"#,
        r#"{"schema":"v1","\u0073chema":"v2"}"#,
        r#"{"count":01}"#,
        r#"{"bad":"\uD800"}"#,
        r#"{"schema":"v1"} trailing"#,
        r#"["not","an","object"]"#,
    ] {
        assert!(
            parse_strict_json(invalid).is_err(),
            "strict JSON parser accepted {invalid}"
        );
    }
}

#[test]
fn qualification_digest_oracles_recompute_artifact_and_report_bytes() {
    let report = concat!(
        r#"{"artifact_digest":"d25252040204953b4a9926344bf5de38d5bbd36d01e71eb25b4c68a535f99248","driver_abi_digest":"bdcb82b2fd4a41614442c79003347ce4a5acb032aab9555a028967f89eff92ee","report_digest":""#,
        "4f2c9370b1133a64243d8b1259620d31eaf772ff41f87e11504e26f8a72caf50",
        r#""}"#,
    );
    assert!(verify_qualification_digests(report, b"artifact-v1", b"driver-abi-v1").is_ok());
    assert!(verify_qualification_digests(report, b"artifact-v2", b"driver-abi-v1").is_err());
    assert!(verify_qualification_digests(report, b"artifact-v1", b"driver-abi-v2").is_err());
    let tampered = report.replacen('{', r#"{"extra":true,"#, 1);
    assert!(verify_qualification_digests(&tampered, b"artifact-v1", b"driver-abi-v1").is_err());
}

fn qualification_digest_hex(digest: &[u8; 32]) -> String {
    let mut value = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

fn replace_json_string_value(report: &str, field: &str, value: &str) -> String {
    let marker = format!("\"{field}\":\"");
    let start = report
        .find(&marker)
        .unwrap_or_else(|| panic!("missing JSON string field {field}"))
        + marker.len();
    let end = start
        + report[start..]
            .find('"')
            .unwrap_or_else(|| panic!("unterminated JSON string field {field}"));
    let mut replaced = report.to_owned();
    replaced.replace_range(start..end, value);
    replaced
}

fn replace_json_number_value(report: &str, field: &str, value: u64) -> String {
    let marker = format!("\"{field}\":");
    let start = report
        .find(&marker)
        .unwrap_or_else(|| panic!("missing JSON number field {field}"))
        + marker.len();
    let end = start
        + report[start..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
    assert!(end > start, "JSON number field {field} has no digits");
    let mut replaced = report.to_owned();
    replaced.replace_range(start..end, &value.to_string());
    replaced
}

fn remove_json_string_field(report: &str, field: &str) -> String {
    let marker = format!("\"{field}\":\"");
    let start = report
        .find(&marker)
        .unwrap_or_else(|| panic!("missing JSON string field {field}"));
    let value_start = start + marker.len();
    let value_end = value_start
        + report[value_start..]
            .find('"')
            .unwrap_or_else(|| panic!("unterminated JSON string field {field}"));
    let mut end = value_end + 1;
    if report.as_bytes().get(end) == Some(&b',') {
        end += 1;
    }
    let mut removed = report.to_owned();
    removed.replace_range(start..end, "");
    removed
}

fn reseal_qualification_report(report: &str) -> String {
    let zeroed = replace_json_string_value(report, "report_digest", &"0".repeat(64));
    let digest = sha256_hex(zeroed.as_bytes());
    replace_json_string_value(&zeroed, "report_digest", &digest)
}

fn bind_qualification_artifact(report: &str, artifact: &[u8]) -> String {
    let report = replace_json_string_value(report, "artifact_digest", &sha256_hex(artifact));
    reseal_qualification_report(&report)
}

fn replace_first_qualification_route_field(artifact: &[u8], field: usize, value: &str) -> Vec<u8> {
    let mut artifact = String::from_utf8(artifact.to_vec()).expect("UTF-8 V5 fixture");
    let route = artifact
        .lines()
        .nth(7)
        .expect("first V5 fixture route")
        .to_owned();
    let mut fields = route.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), 16);
    fields[field] = value;
    artifact = artifact.replacen(&route, &fields.join("\t"), 1);
    artifact.into_bytes()
}

fn qualification_fixture() -> (String, Vec<u8>, Vec<u8>) {
    let driver_abi_proof = b"MambaBiTf32DriverAbiV2\nfixture\n".to_vec();
    let driver_abi_digest = sha256_hex(&driver_abi_proof);
    let mut route_lines = Vec::new();
    for (route_index, symbol) in ["route-a", "route-b"].into_iter().enumerate() {
        let mut fields = vec![symbol.to_owned()];
        for field in 1..=10 {
            fields.push(sha256_hex(
                format!("qualification-v5-fixture:{route_index}:{field}").as_bytes(),
            ));
        }
        fields[2] = fields[1].clone();
        fields[5] = fields[6].clone();
        fields.extend(["1.250000000".to_owned(), "1.000000000".to_owned()]);
        fields.push("64".to_owned());
        fields.extend(["0".to_owned(), "0".to_owned()]);
        route_lines.push(fields.join("\t"));
    }
    let mut artifact = format!(
        concat!(
            "MambaBiTf32QualificationArtifactV5\n",
            "cc\t8.9\n",
            "suite\tsanitizer\n",
            "repeat\t1\n",
            "boundary_cases_per_route\t47\n",
            "artifact_set\t{}\n",
            "driver_abi\t{}\n"
        ),
        sha256_hex(b"qualification-v5-fixture-artifact-set"),
        driver_abi_digest,
    );
    for route in &route_lines {
        artifact.push_str(route);
        artifact.push('\n');
    }
    let artifact = artifact.into_bytes();
    let route_rows = route_lines
        .iter()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let semantic_fields = [
        (1, b"tf32-qualification-all-output.v1".as_slice()),
        (3, b"tf32-qualification-all-zero-reduction.v1".as_slice()),
        (4, b"tf32-qualification-all-boundary-output.v1".as_slice()),
        (7, b"tf32-qualification-all-encoded-maps.v1".as_slice()),
        (6, b"tf32-qualification-all-graph-routes.v1".as_slice()),
        (8, b"tf32-qualification-all-staged-guarded.v1".as_slice()),
        (
            9,
            b"tf32-qualification-all-exceptional-values.v1".as_slice(),
        ),
        (10, b"tf32-qualification-all-cross-m.v1".as_slice()),
    ];
    let semantic_digests = semantic_fields.map(|(field, domain)| {
        qualification_digest_hex(
            &qualification_semantic_digest(&route_rows, field, domain)
                .expect("valid V5 fixture route digest"),
        )
    });
    let (max_local_bytes, exception_count, driver_jit_resource_digest) =
        qualification_driver_jit_resource_summary(&route_rows)
            .expect("valid V5 fixture resource evidence");
    let report = format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32QualificationV5\",",
            "\"exact_cc\":\"8.9\",\"suite\":\"sanitizer\",\"repeat\":1,",
            "\"driver_jit_local_memory\":\"pass\",",
            "\"routes_qualified\":2,\"boundary_cases\":94,",
            "\"max_driver_jit_local_memory_bytes\":{},",
            "\"driver_jit_local_memory_exception_count\":{},",
            "\"artifact_digest\":\"{}\",",
            "\"output_digest\":\"{}\",",
            "\"zero_reduction_digest\":\"{}\",",
            "\"boundary_output_digest\":\"{}\",",
            "\"tensor_map_identity_digest\":\"{}\",",
            "\"ordered_graph_route_digest\":\"{}\",",
            "\"staged_guarded_digest\":\"{}\",",
            "\"exceptional_values_digest\":\"{}\",",
            "\"cross_m_invariance_digest\":\"{}\",",
            "\"driver_abi_digest\":\"{}\",",
            "\"driver_jit_resource_digest\":\"{}\",",
            "\"report_digest\":\"{}\"}}"
        ),
        max_local_bytes,
        exception_count,
        sha256_hex(&artifact),
        semantic_digests[0],
        semantic_digests[1],
        semantic_digests[2],
        semantic_digests[3],
        semantic_digests[4],
        semantic_digests[5],
        semantic_digests[6],
        semantic_digests[7],
        driver_abi_digest,
        qualification_digest_hex(&driver_jit_resource_digest),
        "0".repeat(64),
    );
    (
        reseal_qualification_report(&report),
        artifact,
        driver_abi_proof,
    )
}

#[test]
fn qualification_fixture_rejects_every_boundary_and_schema_mutation() {
    let (report, artifact, driver_abi_proof) = qualification_fixture();
    assert!(verify_qualification_digests(&report, &artifact, &driver_abi_proof).is_ok());

    let wrong_schema = replace_json_string_value(&report, "schema", "MambaBiTf32QualificationV3");
    let wrong_schema = reseal_qualification_report(&wrong_schema);
    assert!(
        verify_qualification_digests(&wrong_schema, &artifact, &driver_abi_proof).is_err(),
        "a V5 artifact must require the V5 report schema"
    );

    let missing_boundary = remove_json_string_field(&report, "boundary_output_digest");
    let missing_boundary = reseal_qualification_report(&missing_boundary);
    assert!(
        verify_qualification_digests(&missing_boundary, &artifact, &driver_abi_proof).is_err(),
        "a V5 artifact must not bypass semantic verification by omitting its boundary digest"
    );

    let wrong_case_count = String::from_utf8(artifact.clone())
        .expect("UTF-8 V5 fixture")
        .replacen(
            "boundary_cases_per_route\t47",
            "boundary_cases_per_route\t46",
            1,
        )
        .into_bytes();
    let wrong_case_report = replace_json_number_value(&report, "boundary_cases", 92);
    let wrong_case_report = bind_qualification_artifact(&wrong_case_report, &wrong_case_count);
    assert!(
        verify_qualification_digests(&wrong_case_report, &wrong_case_count, &driver_abi_proof,)
            .is_err(),
        "V5 must freeze all 47 boundary cases per route"
    );

    let mut wrong_boundary = String::from_utf8(artifact.clone()).expect("UTF-8 V5 fixture");
    let boundary = wrong_boundary
        .lines()
        .nth(7)
        .expect("first V5 fixture route")
        .split('\t')
        .nth(4)
        .expect("boundary route digest")
        .to_owned();
    wrong_boundary = wrong_boundary.replacen(&boundary, &sha256_hex(b"changed-boundary"), 1);
    let wrong_boundary = wrong_boundary.into_bytes();
    let wrong_boundary_report = bind_qualification_artifact(&report, &wrong_boundary);
    assert!(
        verify_qualification_digests(&wrong_boundary_report, &wrong_boundary, &driver_abi_proof,)
            .is_err(),
        "the all-route boundary digest must bind every per-route boundary digest"
    );

    for (field, label) in [(2, "graph output"), (5, "eager route")] {
        let changed = replace_first_qualification_route_field(
            &artifact,
            field,
            &sha256_hex(format!("changed-{label}").as_bytes()),
        );
        let changed_report = bind_qualification_artifact(&report, &changed);
        assert!(
            verify_qualification_digests(&changed_report, &changed, &driver_abi_proof).is_err(),
            "V5 must semantically bind its {label} field"
        );
    }

    let mut wrong_driver = String::from_utf8(artifact.clone()).expect("UTF-8 V5 fixture");
    wrong_driver = wrong_driver.replacen(
        &format!("driver_abi\t{}", sha256_hex(&driver_abi_proof)),
        &format!("driver_abi\t{}", sha256_hex(b"different-driver-abi")),
        1,
    );
    let wrong_driver = wrong_driver.into_bytes();
    let wrong_driver_report = bind_qualification_artifact(&report, &wrong_driver);
    assert!(
        verify_qualification_digests(&wrong_driver_report, &wrong_driver, &driver_abi_proof)
            .is_err(),
        "the artifact Driver ABI must match both the report and proof"
    );

    let mut short_row = String::from_utf8(artifact.clone()).expect("UTF-8 V5 fixture");
    let route = short_row
        .lines()
        .nth(7)
        .expect("first V5 fixture route")
        .to_owned();
    let shortened = route.rsplit_once('\t').expect("route register field").0;
    short_row = short_row.replacen(&route, shortened, 1);
    let short_row = short_row.into_bytes();
    let short_row_report = bind_qualification_artifact(&report, &short_row);
    assert!(
        verify_qualification_digests(&short_row_report, &short_row, &driver_abi_proof).is_err(),
        "V5 route rows must contain exactly 16 fields"
    );

    let zero_registers = replace_first_qualification_route_field(&artifact, 13, "0");
    let zero_registers_report = bind_qualification_artifact(&report, &zero_registers);
    assert!(
        verify_qualification_digests(&zero_registers_report, &zero_registers, &driver_abi_proof,)
            .is_err(),
        "V5 resource evidence must report a nonzero register count"
    );

    let excessive_local = replace_first_qualification_route_field(&artifact, 14, "16");
    let excessive_local_report = bind_qualification_artifact(&report, &excessive_local);
    assert!(
        verify_qualification_digests(&excessive_local_report, &excessive_local, &driver_abi_proof,)
            .is_err(),
        "V5 must reject observed Driver JIT local bytes above the approved cap"
    );

    let admitted_local = replace_first_qualification_route_field(&artifact, 14, "16");
    let admitted_local = replace_first_qualification_route_field(&admitted_local, 15, "16");
    let admitted_local_report = bind_qualification_artifact(&report, &admitted_local);
    assert!(
        verify_qualification_digests(&admitted_local_report, &admitted_local, &driver_abi_proof,)
            .is_err(),
        "V5 resource digest must bind each route's observed bytes and approved cap"
    );

    let wrong_resource_digest = replace_json_string_value(
        &report,
        "driver_jit_resource_digest",
        &sha256_hex(b"different-driver-jit-resources"),
    );
    let wrong_resource_digest = reseal_qualification_report(&wrong_resource_digest);
    assert!(
        verify_qualification_digests(&wrong_resource_digest, &artifact, &driver_abi_proof).is_err(),
        "V5 must reject a report-only Driver JIT resource digest mutation"
    );

    assert!(parse_sha256_hex(&"A".repeat(64), "uppercase fixture digest").is_err());
}

#[test]
fn hardware_sanitizer_invocations_isolate_outputs_and_expected_driver_errors() {
    let directory = Path::new("/tmp/gemm-bi-tf32-sanitizer-contract");
    let binary = directory.join("gemm-bi-tf32-qualification");
    let mut output_paths = BTreeSet::new();
    for tool in ["memcheck", "racecheck", "initcheck", "synccheck"] {
        let arguments = sanitizer_qualification_arguments((8, 9), 15, directory, tool);
        for flag in ["--artifact-output", "--driver-abi-proof-output"] {
            let index = arguments
                .iter()
                .position(|argument| argument == flag)
                .expect("sanitizer output flag");
            assert!(
                output_paths.insert(arguments[index + 1].clone()),
                "sanitizer output paths must be unique"
            );
        }
        let command = sanitizer_command(&binary, &arguments, tool);
        let command_arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            &command_arguments[..7],
            [
                "--error-exitcode",
                "99",
                "--report-api-errors",
                "no",
                "--tool",
                tool,
                binary.to_str().expect("ASCII test path"),
            ]
        );
        assert_eq!(
            &command_arguments[command_arguments.len() - 2..],
            ["--suite", "sanitizer"]
        );
    }
    assert_eq!(output_paths.len(), 8);
}

#[test]
fn driver_abi_lookup_uses_the_cuda12_compatible_symbol() {
    assert_cuda12_driver_abi_lookup_contract();
}

#[test]
fn wide_qualification_op_census_includes_the_composed_nn_extension() {
    for (cc, total, expected) in [
        ((8, 0), 19, (7, 6, 6)),
        ((8, 6), 19, (7, 6, 6)),
        ((8, 7), 19, (7, 6, 6)),
        ((8, 9), 19, (7, 6, 6)),
        ((9, 0), 25, (9, 8, 8)),
        ((10, 0), 55, (19, 18, 18)),
        ((10, 3), 55, (19, 18, 18)),
        ((11, 0), 55, (19, 18, 18)),
    ] {
        let census = qualified_op_census(cc, total);
        assert_eq!((census.nn, census.tn, census.nt), expected, "CC {cc:?}");
    }
}

#[test]
fn wide_qualification_op_census_preserves_cc12_without_the_extension() {
    for cc in [(12, 0), (12, 1)] {
        let census = qualified_op_census(cc, 36);
        assert_eq!((census.nn, census.tn, census.nt), (12, 13, 11), "CC {cc:?}");
    }
}
