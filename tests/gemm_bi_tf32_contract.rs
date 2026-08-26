#![cfg(feature = "cuda")]

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SM80_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");
const SM90A_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm90a.cu");
const SM100_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm100.cu");
const SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120.cu");
const CONTEXT_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/context.rs");
const DEVICE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/device.rs");
const IDENTITY_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/kernel_identity.rs");
const CONTRACT_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs");
const DISPATCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
const LAUNCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");
const MODULE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
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

fn assert_contains_all(source: &str, required: &[&str], contract: &str) {
    let mask = source_mask(source);
    for item in required {
        assert!(mask.contains(item), "{contract} is missing {item}");
    }
}

fn assert_code_contains_all(source: &str, required: &[&str], contract: &str) {
    assert_contains_all(source, required, contract);
}

fn braced_scope_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let mask = source_mask(source);
    let marker_offset = mask
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"));
    braced_scope_at(source, marker_offset, marker)
}

fn braced_scope_at<'a>(source: &'a str, start: usize, label: &str) -> &'a str {
    let mask = source_mask(source);
    let open_offset = mask[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing opening brace after {label}"));
    let mut depth = 0_u32;
    for (offset, byte) in mask[open_offset..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open_offset + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated scope after {label}")
}

fn token_present(source: &str, token: &str) -> bool {
    let bytes = source.as_bytes();
    let token_bytes = token.as_bytes();
    bytes
        .windows(token_bytes.len())
        .enumerate()
        .any(|(offset, window)| {
            window == token_bytes
                && offset
                    .checked_sub(1)
                    .and_then(|index| bytes.get(index))
                    .is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
                && bytes
                    .get(offset + token_bytes.len())
                    .is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
        })
}

fn impl_scopes_for_type<'a>(source: &'a str, type_name: &str) -> Vec<&'a str> {
    let mask = source_mask(source);
    let mut scopes = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = mask[cursor..].find("impl") {
        let start = cursor + relative;
        let before_is_identifier = start
            .checked_sub(1)
            .and_then(|index| mask.as_bytes().get(index))
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_');
        let after_is_identifier = mask
            .as_bytes()
            .get(start + 4)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_');
        if before_is_identifier || after_is_identifier {
            cursor = start + 4;
            continue;
        }
        let Some(open_relative) = mask[start..].find('{') else {
            break;
        };
        let open = start + open_relative;
        if !token_present(&mask[start..open], type_name) {
            cursor = open + 1;
            continue;
        }
        let mut depth = 0_u32;
        let mut end = None;
        for (offset, byte) in mask[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + offset + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.unwrap_or_else(|| panic!("unterminated impl for {type_name}"));
        scopes.push(&source[start..end]);
        cursor = end;
    }
    assert!(!scopes.is_empty(), "missing impl block for {type_name}");
    scopes
}

fn method_scope_for_type<'a>(source: &'a str, type_name: &str, method: &str) -> &'a str {
    let matching: Vec<_> = impl_scopes_for_type(source, type_name)
        .into_iter()
        .filter_map(|implementation| {
            let mask = source_mask(implementation);
            let offsets = named_function_offsets(&mask, method);
            match offsets.as_slice() {
                [] => None,
                [offset] => Some((implementation, *offset)),
                _ => panic!("duplicate {type_name}::{method} methods in one impl"),
            }
        })
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "expected exactly one {type_name}::{method} implementation"
    );
    braced_scope_at(matching[0].0, matching[0].1, method)
}

fn graph_launches_are_guarded(source: &str) -> bool {
    let mask = source_mask(source);
    let mut closures = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = mask[cursor..].find("with_validated_launch") {
        let call = cursor + relative;
        cursor = call + "with_validated_launch".len();
        if !token_at(&mask, call, "with_validated_launch") {
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

    let mut launch_cursor = 0;
    while let Some(relative) = mask[launch_cursor..].find("self.graph.launch()") {
        let launch = launch_cursor + relative;
        if !closures
            .iter()
            .any(|(start, end)| *start <= launch && launch < *end)
        {
            return false;
        }
        launch_cursor = launch + "self.graph.launch()".len();
    }
    true
}

fn matching_delimiter(
    source: &str,
    open: usize,
    open_delimiter: u8,
    close_delimiter: u8,
) -> Option<usize> {
    let mut depth = 0_u32;
    for (relative, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        if byte == open_delimiter {
            depth += 1;
        } else if byte == close_delimiter {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(open + relative);
            }
        }
    }
    None
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

fn identifiers_with_prefix(source: &str, prefix: &str) -> BTreeSet<String> {
    source
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .filter(|token| token.starts_with(prefix))
        .map(str::to_owned)
        .collect()
}

fn source_mask(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        Line,
        Block(u32),
        Quoted { quote: u8, escaped: bool },
        Raw { hashes: usize },
    }

    let bytes = source.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut state = State::Code;
    let mut cursor = 0;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        let next = bytes.get(cursor + 1).copied();
        match state {
            State::Code if byte == b'/' && next == Some(b'/') => {
                result.extend_from_slice(b"  ");
                cursor += 2;
                state = State::Line;
            }
            State::Code if byte == b'/' && next == Some(b'*') => {
                result.extend_from_slice(b"  ");
                cursor += 2;
                state = State::Block(1);
            }
            State::Code => {
                let raw_prefix = match byte {
                    b'r' => Some(1),
                    b'b' | b'c' if next == Some(b'r') => Some(2),
                    _ => None,
                };
                if let Some(prefix) = raw_prefix {
                    let mut quote = cursor + prefix;
                    while bytes.get(quote) == Some(&b'#') {
                        quote += 1;
                    }
                    if bytes.get(quote) == Some(&b'"') {
                        let hashes = quote - cursor - prefix;
                        for raw_byte in &bytes[cursor..=quote] {
                            result.push(if *raw_byte == b'\n' { b'\n' } else { b' ' });
                        }
                        cursor = quote + 1;
                        state = State::Raw { hashes };
                        continue;
                    }
                }

                let is_character = byte == b'\'' && looks_like_character_literal(bytes, cursor);
                if byte == b'"' || is_character {
                    result.push(b' ');
                    cursor += 1;
                    state = State::Quoted {
                        quote: byte,
                        escaped: false,
                    };
                } else {
                    result.push(byte);
                    cursor += 1;
                }
            }
            State::Line => {
                result.push(if byte == b'\n' { b'\n' } else { b' ' });
                cursor += 1;
                if byte == b'\n' {
                    state = State::Code;
                }
            }
            State::Block(depth) if byte == b'/' && next == Some(b'*') => {
                result.extend_from_slice(b"  ");
                cursor += 2;
                state = State::Block(depth + 1);
            }
            State::Block(depth) if byte == b'*' && next == Some(b'/') => {
                result.extend_from_slice(b"  ");
                cursor += 2;
                state = if depth == 1 {
                    State::Code
                } else {
                    State::Block(depth - 1)
                };
            }
            State::Block(depth) => {
                result.push(if byte == b'\n' { b'\n' } else { b' ' });
                cursor += 1;
                state = State::Block(depth);
            }
            State::Quoted { quote, escaped } => {
                result.push(if byte == b'\n' { b'\n' } else { b' ' });
                cursor += 1;
                if escaped {
                    state = State::Quoted {
                        quote,
                        escaped: false,
                    };
                } else if byte == b'\\' {
                    state = State::Quoted {
                        quote,
                        escaped: true,
                    };
                } else if byte == quote {
                    state = State::Code;
                }
            }
            State::Raw { hashes } => {
                result.push(if byte == b'\n' { b'\n' } else { b' ' });
                cursor += 1;
                if byte == b'"'
                    && bytes
                        .get(cursor..cursor + hashes)
                        .is_some_and(|tail| tail.iter().all(|candidate| *candidate == b'#'))
                {
                    result.extend(std::iter::repeat_n(b' ', hashes));
                    cursor += hashes;
                    state = State::Code;
                }
            }
        }
    }
    String::from_utf8(result).expect("source mask is ASCII plus preserved code")
}

fn looks_like_character_literal(source: &[u8], quote: usize) -> bool {
    let mut cursor = quote + 1;
    let mut escaped = false;
    while let Some(byte) = source.get(cursor).copied() {
        if byte == b'\n' || (!escaped && byte.is_ascii_whitespace()) {
            return false;
        }
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'\'' {
            return true;
        } else if matches!(byte, b':' | b',' | b'>' | b'=' | b'+' | b'-') {
            return false;
        }
        cursor += 1;
    }
    false
}

fn skip_ascii_whitespace(source: &str, mut cursor: usize) -> usize {
    while source
        .as_bytes()
        .get(cursor)
        .is_some_and(u8::is_ascii_whitespace)
    {
        cursor += 1;
    }
    cursor
}

fn identifier_end(source: &str, mut cursor: usize) -> usize {
    while source
        .as_bytes()
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'#')
    {
        cursor += 1;
    }
    cursor
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'#'
}

fn token_at(source: &str, offset: usize, token: &str) -> bool {
    source.as_bytes().get(offset..offset + token.len()) == Some(token.as_bytes())
        && offset
            .checked_sub(1)
            .and_then(|index| source.as_bytes().get(index))
            .is_none_or(|byte| !is_identifier_byte(*byte))
        && source
            .as_bytes()
            .get(offset + token.len())
            .is_none_or(|byte| !is_identifier_byte(*byte))
}

fn named_function_offsets(source: &str, expected_name: &str) -> Vec<usize> {
    let mut matches = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find("fn") {
        let start = cursor + relative;
        cursor = start + 2;
        if !token_at(source, start, "fn") {
            continue;
        }
        let name_start = skip_ascii_whitespace(source, cursor);
        let name_end = identifier_end(source, name_start);
        if name_start != name_end && &source[name_start..name_end] == expected_name {
            matches.push(start);
        }
    }
    matches
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
    if bundle_bytes == 32 {
        vec![
            parameter(0, "u64", 8, None),
            parameter(1, "u64", 8, None),
            parameter(2, "u64", 8, None),
            parameter(3, "u64", 8, None),
            parameter(4, "b8", 32, Some(4)),
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

fn loaded_nvrtc_version() -> (i32, i32) {
    let mut major = 0;
    let mut minor = 0;
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(result, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    (major, minor)
}

fn normalized_reduction_bundle_offset(symbol: &str, bundle_bytes: usize) -> usize {
    match (
        bundle_bytes,
        symbol.contains("_nn_"),
        symbol.contains("_tn_"),
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
                || !(compact.contains(".u32") || compact.contains(".s32"))
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
        .unwrap_or_else(|| panic!("{symbol} must load normalized reduction at bundle +{offset}"));
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
    let all_blocks: BTreeSet<_> = (0..ranges.len()).collect();
    let mut dominators = vec![all_blocks.clone(); ranges.len()];
    dominators[0] = BTreeSet::from([0]);
    loop {
        let mut changed = false;
        for block in 1..ranges.len() {
            let mut next = if let Some(first) = predecessors[block].first() {
                dominators[*first].clone()
            } else {
                BTreeSet::new()
            };
            for predecessor in predecessors[block].iter().skip(1) {
                next = next
                    .intersection(&dominators[*predecessor])
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
        if protected.iter().any(|opcode| body.contains(opcode)) {
            assert!(
                dominators[block].contains(&guard_block),
                "{symbol} zero guard does not dominate protected block {block}"
            );
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
    let allowed_fp = if symbol.contains("_nn_") {
        BTreeSet::from(["mul.rn.f32", "fma.rn.f32"])
    } else if symbol.contains("_tn_") {
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
            (192, 27_648 * stage)
        } else if symbol.contains("_m64n64_") {
            assert!(matches!(stage, 2 | 3), "invalid SM80 M64N64 stage");
            (128, 18_432 * stage)
        } else if symbol.contains("_m16n32_") {
            assert_eq!(stage, 4, "invalid SM80 M16N32 stage");
            (96, 29_696)
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
            symbol.contains("_m128n64_") || symbol.contains("_m64n128_"),
            "unknown SM120 TF32 tile for {symbol}"
        );
        let stage = if symbol.contains("_s2") {
            2
        } else if symbol.contains("_s3") {
            3
        } else {
            panic!("unknown SM120 TF32 stage for {symbol}")
        };
        (128, 128 + 24_576 * stage)
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
    for line in report.lines() {
        for marker in [
            " bytes stack frame",
            " bytes spill stores",
            " bytes spill loads",
        ] {
            if let Some(value) = metric_before(line, marker) {
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
        assert!(
            record.contains("STACK:0") && record.contains("LOCAL:0"),
            "{label}/{symbol} resource record must explicitly report STACK:0 LOCAL:0: {record}"
        );
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
        assert_eq!(
            properties.get(symbol).map(Vec::as_slice),
            Some(&[(0, 0, 0)][..]),
            "{label}/{symbol} requires an explicit zero stack/spill ptxas record"
        );
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
    let text_marker = format!(".text.{symbol}:");
    let start = sass
        .find(&text_marker)
        .unwrap_or_else(|| panic!("SASS is missing function {symbol}"));
    let tail = &sass[start..];
    let end = tail[text_marker.len()..]
        .find("\n.text.")
        .map(|offset| text_marker.len() + offset)
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
            if !identifier.is_empty() {
                assert!(
                    nodes
                        .insert(identifier.clone(), attributes.to_owned())
                        .is_none(),
                    "{symbol} duplicate DOT node {identifier}"
                );
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
            "{symbol} DOT edge from unknown {from}"
        );
        targets.sort();
        targets.dedup();
        for target in targets.iter() {
            assert!(
                nodes.contains_key(target),
                "{symbol} DOT edge to unknown {target}"
            );
        }
    }
    for node in nodes.keys() {
        successors.entry(node.clone()).or_default();
    }
    (nodes, successors)
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
        tail.trim().parse().expect("nvdisasm source line number"),
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
        let (file, source_line) = location
            .as_ref()
            .unwrap_or_else(|| panic!("{symbol} SASS offset {offset:x} has no line information"));
        assert!(
            offsets.insert(offset),
            "{symbol} duplicate SASS instruction offset {offset:x}"
        );
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

fn anchored_sass_node(
    nodes: &BTreeMap<String, String>,
    instructions: &[SassInstruction<'_>],
    anchor: (&str, u64),
    accepts: impl Fn(&str) -> bool,
    symbol: &str,
    role: &str,
) -> String {
    let anchored: Vec<_> = instructions
        .iter()
        .filter(|instruction| {
            Path::new(&instruction.file)
                .file_name()
                .and_then(|file| file.to_str())
                == Some(anchor.0)
                && instruction.line == anchor.1
                && accepts(instruction.mnemonic)
        })
        .collect();
    assert!(
        !anchored.is_empty(),
        "{symbol} has no {role} SASS instruction at {}:{}",
        anchor.0,
        anchor.1
    );
    let mut containing_nodes = BTreeSet::new();
    for instruction in anchored {
        containing_nodes.insert(sass_offset_node(nodes, instruction.offset, symbol, role));
    }
    assert_eq!(
        containing_nodes.len(),
        1,
        "{symbol} duplicate {role} anchor nodes"
    );
    containing_nodes.pop_first().expect("one anchored DOT node")
}

fn anchored_sass_offset(
    instructions: &[SassInstruction<'_>],
    anchor: (&str, u64),
    accepts: impl Fn(&SassInstruction<'_>) -> bool,
    symbol: &str,
    role: &str,
) -> u64 {
    let anchored: Vec<_> = instructions
        .iter()
        .filter(|instruction| {
            Path::new(&instruction.file)
                .file_name()
                .and_then(|file| file.to_str())
                == Some(anchor.0)
                && instruction.line == anchor.1
                && accepts(instruction)
        })
        .map(|instruction| instruction.offset)
        .collect();
    assert_eq!(
        anchored.len(),
        1,
        "{symbol} requires exactly one {role} SASS instruction at {}:{}",
        anchor.0,
        anchor.1
    );
    anchored[0]
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

fn assert_sass_cfg_corroboration(
    dot: &str,
    line_sass: &str,
    source: &str,
    symbol: &str,
    label: &str,
) {
    let graph = dot_function_graph(dot, symbol);
    let (nodes, successors) = parse_dot_cfg(graph, symbol);
    let line_entry = sass_entry(line_sass, symbol);
    source_anchor(source, K0_GUARD_ANCHOR, label);
    source_anchor(source, K0_BRANCH_ANCHOR, label);
    source_anchor(source, K0_ZERO_STORE_ANCHOR, label);
    let instructions = sass_line_instructions(line_entry, symbol);
    let compare_offset = anchored_sass_offset(
        &instructions,
        K0_GUARD_ANCHOR,
        |instruction| {
            instruction.predicate.is_none()
                && (instruction.mnemonic.starts_with("ISETP")
                    || instruction.mnemonic.starts_with("UISETP"))
        },
        symbol,
        "K=0 compare",
    );
    let branch_offset = anchored_sass_offset(
        &instructions,
        K0_BRANCH_ANCHOR,
        |instruction| instruction.mnemonic.starts_with("BRA") && instruction.predicate.is_some(),
        symbol,
        "K=0 conditional branch",
    );
    let guard = sass_offset_node(&nodes, branch_offset, symbol, "K=0 conditional branch");
    assert_eq!(
        sass_offset_node(&nodes, compare_offset, symbol, "K=0 compare"),
        guard,
        "{label}/{symbol} K=0 compare and branch must share one DOT node"
    );
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
        compare_operands
            .first()
            .is_some_and(|predicate| sass_register(predicate, "P", "PT") && *predicate != "PT"),
        "{label}/{symbol} K=0 compare has no concrete predicate destination"
    );
    let compare_predicate = compare_operands[0];
    let (branch_predicate, _) = branch.predicate.expect("anchored conditional branch");
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
    let zero_store = anchored_sass_node(
        &nodes,
        &instructions,
        K0_ZERO_STORE_ANCHOR,
        |mnemonic| mnemonic.starts_with("STG"),
        symbol,
        "K=0 store",
    );
    assert_ne!(guard, zero_store, "{label}/{symbol} guard/store node alias");

    let mut predecessors = BTreeMap::<String, Vec<String>>::new();
    for node in nodes.keys() {
        predecessors.insert(node.clone(), Vec::new());
    }
    for (from, targets) in &successors {
        for target in targets {
            predecessors
                .get_mut(target)
                .expect("validated DOT target")
                .push(from.clone());
        }
    }
    let entries: Vec<_> = predecessors
        .iter()
        .filter_map(|(node, incoming)| incoming.is_empty().then_some(node.clone()))
        .collect();
    assert_eq!(entries.len(), 1, "{label}/{symbol} DOT entry block");
    let entry = &entries[0];
    let all: BTreeSet<_> = nodes.keys().cloned().collect();
    let mut dominators: BTreeMap<String, BTreeSet<String>> = nodes
        .keys()
        .map(|node| {
            let initial = if node == entry {
                BTreeSet::from([node.clone()])
            } else {
                all.clone()
            };
            (node.clone(), initial)
        })
        .collect();
    loop {
        let mut changed = false;
        for node in nodes.keys().filter(|node| *node != entry) {
            let incoming = &predecessors[node];
            let mut next = incoming
                .first()
                .map_or_else(BTreeSet::new, |first| dominators[first].clone());
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
    let (transfer, matrix): (&[&str], &[&str]) = match label {
        "SM80" => (&["LDGSTS"], &["HMMA"]),
        "SM90a" => (&["UTMALDG"], &["HGMMA", "WGMMA"]),
        "SM100" => (&["UTMALDG"], &["UTCHMMA", "TCGEN"]),
        "SM120" => (&["UTMALDG"], &["HMMA"]),
        _ => panic!("unknown SASS CFG family {label}"),
    };
    let protected = [
        "UTMALDG", "HGMMA", "WGMMA", "UTCHMMA", "TCGEN", "TMEM", "HMMA", "LDGSTS", "BAR.",
    ];
    assert!(
        nodes[&guard].contains("BRA") && successors[&guard].len() == 2,
        "{label}/{symbol} anchored guard must be a conditional branch node"
    );
    let regions: Vec<_> = successors[&guard]
        .iter()
        .map(|successor| reachable(successor))
        .collect();
    let zero_indices: Vec<_> = (0..2)
        .filter(|index| regions[*index].contains(&zero_store))
        .collect();
    assert_eq!(
        zero_indices.len(),
        1,
        "{label}/{symbol} anchored store must select exactly one zero successor"
    );
    let zero_index = zero_indices[0];
    let zero_body = regions[zero_index]
        .iter()
        .map(|block| nodes[block].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let nonzero_body = regions[1 - zero_index]
        .iter()
        .map(|block| nodes[block].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !protected.iter().any(|opcode| zero_body.contains(opcode)) && zero_body.contains("EXIT"),
        "{label}/{symbol} anchored zero SASS region is unsafe or unterminated"
    );
    assert!(
        transfer.iter().any(|opcode| nonzero_body.contains(opcode))
            && matrix.iter().any(|opcode| nonzero_body.contains(opcode)),
        "{label}/{symbol} anchored nonzero SASS region is vacuous"
    );
    if label == "SM100" && symbol.contains("_sm100_tcgen_tf32_v1_") {
        assert_tcgen_management_cfg(
            &nodes,
            &successors,
            &dominators,
            &instructions,
            &regions[zero_index],
            &regions[1 - zero_index],
            symbol,
        );
    }
    for (node, body) in &nodes {
        if protected.iter().any(|opcode| body.contains(opcode)) {
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

    for instruction in instructions {
        let mnemonic = instruction.mnemonic;
        let allowed_management = matches!(
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
            "{symbol} numeric atomic/reduction mnemonic {mnemonic}"
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
        })
        .map(|instruction| {
            (
                instruction,
                sass_offset_node(nodes, instruction.offset, symbol, "TCGEN matrix"),
            )
        })
        .collect();
    assert!(!matrix.is_empty(), "{symbol} has no TCGEN matrix work");
    for (instruction, matrix_node) in &matrix {
        assert!(
            nonzero_region.contains(matrix_node)
                && dominators[matrix_node].contains(&allocation_node)
                && (matrix_node != &allocation_node || last_or < instruction.offset),
            "{symbol} TCGEN allocation does not dominate matrix offset {:x}",
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
    for node in &normal_exit_nodes {
        assert!(
            successors[node].is_empty(),
            "{symbol} normal exit node {node} has successors"
        );
    }
    let mut reaches_normal_exit = normal_exit_nodes.clone();
    loop {
        let mut changed = false;
        for node in nonzero_region {
            if successors[node]
                .iter()
                .any(|successor| reaches_normal_exit.contains(successor))
            {
                changed |= reaches_normal_exit.insert(node.clone());
            }
        }
        if !changed {
            break;
        }
    }
    assert_eq!(
        &reaches_normal_exit, nonzero_region,
        "{symbol} nonzero CFG contains a path with no normal exit"
    );

    let synthetic_exit = "$normal_exit".to_owned();
    assert!(!nodes.contains_key(&synthetic_exit));
    let mut universe = nonzero_region.clone();
    universe.insert(synthetic_exit.clone());
    let mut postdominators: BTreeMap<_, _> = nonzero_region
        .iter()
        .map(|node| (node.clone(), universe.clone()))
        .collect();
    postdominators.insert(
        synthetic_exit.clone(),
        BTreeSet::from([synthetic_exit.clone()]),
    );
    loop {
        let mut changed = false;
        for node in nonzero_region {
            let outgoing: Vec<_> = if normal_exit_nodes.contains(node) {
                vec![synthetic_exit.clone()]
            } else {
                successors[node].clone()
            };
            assert!(!outgoing.is_empty(), "{symbol} non-exit sink {node}");
            let mut next = postdominators[&outgoing[0]].clone();
            for successor in outgoing.iter().skip(1) {
                next = next
                    .intersection(&postdominators[successor])
                    .cloned()
                    .collect();
            }
            next.insert(node.clone());
            if next != postdominators[node] {
                postdominators.insert(node.clone(), next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    assert!(
        postdominators[&allocation_node].contains(&deallocation_node),
        "{symbol} TCGEN deallocation does not postdominate allocation"
    );
    for (instruction, matrix_node) in &matrix {
        assert!(
            postdominators[matrix_node].contains(&deallocation_node)
                && (matrix_node != &deallocation_node || instruction.offset < deallocation.offset),
            "{symbol} TCGEN deallocation does not postdominate matrix offset {:x}",
            instruction.offset
        );
    }
    for (instruction, exit_node) in normal_exits {
        assert!(
            exit_node != deallocation_node || deallocation.offset < instruction.offset,
            "{symbol} normal exit precedes same-block TCGEN deallocation"
        );
    }
}

fn assert_sass_entry_contract(sass: &str, symbol: &str, label: &str) {
    let entry = sass_entry(sass, symbol);
    assert!(
        !entry.contains("LDL") && !entry.contains("STL"),
        "{label}/{symbol} must not use local memory"
    );
    let instructions = sass_line_instructions(entry, symbol);
    if label == "SM100" && symbol.contains("_sm100_tcgen_tf32_v1_") {
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
    if symbol.contains("_nn_") {
        assert!(entry.contains("FFMA") && entry.contains("FMUL"));
    } else if symbol.contains("_tn_") {
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

fn compact_code(source: &str) -> String {
    source_mask(source)
        .chars()
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
    for (index, (_, field)) in fields.iter().enumerate() {
        assert!(
            code.contains(&format!(
                "static_assert(offsetof({name},{field})=={}",
                index * 4
            )),
            "{name} must freeze offset {} for {field}",
            index * 4
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

fn expected_sm80_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for (tile, stages) in [
            ("m128n64", &[2, 3][..]),
            ("m64n64", &[2, 3][..]),
            ("m16n32", &[4][..]),
        ] {
            for stage in stages {
                symbols.insert(format!(
                    "sgemm_bi_{op}_sm80_mma_tf32_v1_{tile}_bk32_s{stage}"
                ));
            }
        }
    }
    symbols
}

fn expected_sm90a_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for warpgroups in [1, 2] {
            symbols.insert(format!(
                "sgemm_bi_{op}_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg{warpgroups}"
            ));
        }
    }
    symbols
}

fn expected_sm100_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for columns in [64, 128] {
            for stages in [2, 3, 4] {
                for schedule in ["c4", "p8"] {
                    symbols.insert(format!(
                        "sgemm_bi_{op}_sm100_tcgen_tf32_v1_m128n{columns}_bk32_s{stages}_{schedule}"
                    ));
                }
            }
        }
    }
    symbols
}

fn expected_sm120_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["m128n64", "m64n128"] {
            for stages in [2, 3] {
                symbols.insert(format!(
                    "sgemm_bi_{op}_sm120_tma_mma_tf32_v1_{tile}_bk32_s{stages}"
                ));
            }
        }
    }
    symbols
}

fn expected_hardware_symbols(cc: (u32, u32)) -> BTreeSet<String> {
    let mut symbols = expected_sm80_symbols();
    let specialized = match cc.0 {
        8 => BTreeSet::new(),
        9 => expected_sm90a_symbols(),
        10 | 11 => expected_sm100_symbols(),
        12 => expected_sm120_symbols(),
        _ => panic!("unsupported TF32 qualification CC {}.{}", cc.0, cc.1),
    };
    symbols.extend(specialized);
    symbols
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

fn release_entry_target_count() -> usize {
    RELEASE_TARGET_MATRIX
        .iter()
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
fn exact_f32_policy_is_the_public_default_and_env_is_strict() {
    let declaration = braced_scope_after(CONTEXT_SOURCE, "pub enum F32TriadPolicy");
    assert_contains_all(
        declaration,
        &[
            "#[default]",
            "ExactScalarFmaV1 = 0",
            "AllowDeterministicTf32V1 = 1",
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

    let parser = source_mask(braced_scope_after(CONTEXT_SOURCE, "pub fn parse_env_value"));
    assert_contains_all(
        &parser,
        &[
            "pub fn parse_env_value(value: &str) -> Result<Self, String>",
            "ExactScalarFmaV1",
            "AllowDeterministicTf32V1",
        ],
        "MAMBA_RS_BI_F32_POLICY parser",
    );
    for forbidden in [
        "to_ascii_lowercase",
        "\"\" =>",
        "\"1\" =>",
        "\"on\" =>",
        "\"true\" =>",
        "\"yes\" =>",
    ] {
        assert!(
            !parser.contains(forbidden),
            "F32 triad policy must reject bool-like value {forbidden}"
        );
    }
    let env = braced_scope_after(CONTEXT_SOURCE, "fn f32_triad_policy_from_env");
    assert_contains_all(
        env,
        &["parse_env_value", "ExactScalarFmaV1", "NotUnicode"],
        "F32 triad environment initialization",
    );
}

#[test]
fn deterministic_tf32_policy_is_separate_from_cublas_tf32_state() {
    assert_contains_all(
        CONTEXT_SOURCE,
        &[
            "cublas_tf32:",
            "f32_triad_policy:",
            "pub fn set_f32_triad_policy(&self, policy: F32TriadPolicy)",
            "pub fn f32_triad_policy(&self) -> F32TriadPolicy",
            "pub fn disable_tf32(&self)",
            "pub fn tf32(&self) -> bool",
        ],
        "cuBLAS/triad TF32 state separation",
    );
    assert!(
        !source_mask(CONTEXT_SOURCE).contains("\n    tf32: std::cell::Cell<bool>"),
        "the internal legacy field must be named cublas_tf32"
    );

    let disable_cublas = source_mask(braced_scope_after(
        CONTEXT_SOURCE,
        "pub fn disable_tf32(&self)",
    ));
    assert_contains_all(
        &disable_cublas,
        &["cublasSetMathMode", "cublas_tf32.set(false)"],
        "legacy cuBLAS TF32 setter",
    );
    assert!(
        !disable_cublas.contains("f32_triad_policy.set"),
        "legacy cuBLAS state must not mutate deterministic triad policy"
    );

    let set_policy = source_mask(braced_scope_after(
        CONTEXT_SOURCE,
        "pub fn set_f32_triad_policy",
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
            "MmaTf32RnaV1",
            "Sm90aWgmmaTf32TmaV1",
            "Sm100Tcgen05Tf32TmaV1",
            "Sm120TmaMmaTf32RnaV1",
            "NUMERIC_CONTRACT_DOMAIN",
            "ARTIFACT_DIGEST_DOMAIN",
            "COMPILER_TARGET_DOMAIN",
            "DRIVER_BUILD_DIGEST_DOMAIN",
            "TUNING_TABLE_REVISION",
            "SCHEDULE_REVISION",
            "ResolvedInstructionFamily",
            "ResolvedInstructionShape",
            "ResolvedOperandConversion",
            "RegisterCvtRnaTf32F32V1",
            "TensorMapTfloat32V1",
            "TensorMapUint32ThenCvtRnaTf32F32V1",
        ],
        "resolved TF32 route identity",
    );
    for variant in [
        "MmaTf32RnaV1",
        "Sm90aWgmmaTf32TmaV1",
        "Sm100Tcgen05Tf32TmaV1",
        "Sm120TmaMmaTf32RnaV1",
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
fn exact_policy_never_selects_tf32_and_allow_policy_falls_back_to_scalar() {
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
            "MmaTf32RnaV1(Tf32PortableRoute)",
            "Sm90aWgmmaTf32TmaV1(Tf32Sm90aRoute)",
            "Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route)",
            "Sm120TmaMmaTf32RnaV1(Tf32Sm120Route)",
            "pub enum F32TriadSelection",
            "ScalarFmaV1",
            "Tf32(Tf32PhysicalRoute)",
            "pub struct Tf32QualifiedModule",
            "module_kind: ModuleKind",
            "target: CudaTarget",
            "artifact: ArtifactIdentity",
            "compiler: CompilerIdentity",
            "device: DeviceIdentity",
            "device_caps: DeviceCaps",
            "pub struct F32TriadAvailability",
            "portable: Option<Tf32QualifiedModule>",
            "specialized: Option<Tf32QualifiedModule>",
        ],
        "public TF32 selection contract",
    );

    let resolver = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_f32_triad_auto",
    ));
    assert_contains_all(
        &resolver,
        &[
            "policy: F32TriadPolicy",
            "request: F32TriadRequest",
            "availability: F32TriadAvailability",
            "Result<F32TriadSelection, String>",
            "F32TriadPolicy::ExactScalarFmaV1",
            "F32TriadPolicy::AllowDeterministicTf32V1",
            "F32TriadSelection::ScalarFmaV1",
            "request",
            "availability",
            "validate",
        ],
        "automatic TF32 resolver",
    );
    let exact = resolver
        .find("F32TriadPolicy::ExactScalarFmaV1")
        .expect("exact policy branch");
    let allow = resolver
        .find("F32TriadPolicy::AllowDeterministicTf32V1")
        .expect("allow policy branch");
    let exact_branch = if exact < allow {
        &resolver[exact..allow]
    } else {
        &resolver[exact..]
    };
    assert!(
        exact_branch.contains("F32TriadSelection::ScalarFmaV1")
            && !exact_branch.contains("F32TriadSelection::Tf32"),
        "exact policy must resolve directly to ScalarFmaV1"
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
            &["tile: Tf32Sm120Tile", "stages: Sm120Stages"][..],
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
    let automatic = source_mask(braced_scope_after(
        DISPATCH_SOURCE,
        "pub fn resolve_f32_triad_auto",
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
            "Tf32PhysicalRoute::MmaTf32RnaV1",
            "Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1",
            "Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1",
            "Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1",
        ],
        "forced resolver physical-family admission",
    );
}

#[test]
fn graph_identity_rejects_policy_and_physical_route_drift_before_launch() {
    let route_snapshot = braced_scope_after(CONTEXT_SOURCE, "pub fn gemm_route(&self)");
    assert_contains_all(
        route_snapshot,
        &["cublas_tf32", "f32_triad_policy", "GemmRouteIdentity"],
        "context graph route snapshot",
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
            "pub struct CapturedGemmGraphPlan",
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
        &["CapturedGemmGraphPlan", "with_validated_launch"],
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
            "Result<(CudaGraph, Option<CapturedGemmGraphPlan>), String>",
            "begin_gemm_route_recording",
            "capture_into_graph(&ctx.stream",
            ".finish()",
        ],
        "common graph capture/route-plan wrapper",
    );
    assert!(
        capture
            .find("begin_gemm_route_recording")
            .expect("recorder")
            < capture.find("capture_into_graph").expect("capture"),
        "route recorder allocation must precede cuStreamBeginCapture"
    );

    let launch = source_mask(braced_scope_after(
        IDENTITY_SOURCE,
        "pub(crate) fn with_validated_launch",
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

    let identity_tests = braced_scope_after(IDENTITY_SOURCE, "#[cfg(test)]");
    let mutation = braced_scope_after(
        identity_tests,
        "fn validated_launch_rejects_policy_mutation_before_closure",
    );
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
fn exact_graph_inventory_wires_only_direct_and_conditional_triad_holders() {
    let direct_holders = [
        (
            "M1 f32 training",
            TRAINING_GRAPH_SOURCE,
            "GpuMambaF32TrainingStepGraph",
            "capture",
            &["replay"][..],
        ),
        (
            "M1 f32 inference",
            INFERENCE_SOURCE,
            "GpuMambaInference",
            "capture_graph",
            &["step", "step_gpu_only"][..],
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
    ];
    for (label, source, holder, capture, replays) in direct_holders {
        let structure = source_mask(braced_scope_after(source, &format!("struct {holder}")));
        assert!(
            structure.contains("Option<CapturedGemmGraphPlan>") && structure.contains("GemmRoute"),
            "{label} must store a physical plan beside its existing route snapshot"
        );
        let capture = source_mask(method_scope_for_type(source, holder, capture));
        assert_code_contains_all(
            &capture,
            &[
                "capture_into_graph_with_gemm_plan",
                "require_f32_triad_graph_plan",
            ],
            &format!("{label} scoped capture"),
        );
        for replay in replays {
            let replay = source_mask(method_scope_for_type(source, holder, replay));
            assert!(
                replay.contains("with_validated_launch")
                    || replay.contains("launch_captured_graph")
                    || replay.contains("launch_mixed_native_graph"),
                "{label}::{replay} must reach the scoped validated launch"
            );
            assert!(
                graph_launches_are_guarded(&replay),
                "{label}::{replay} may not launch outside the validated closure"
            );
        }
    }

    let conditional_holders = [
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
        (
            "M1 mixed-native inference",
            INFERENCE_SOURCE,
            "GpuMambaInferenceMixed",
            "capture_graph_mixed_native",
            &["step_mixed_native", "step_gpu_only_mixed_native"][..],
        ),
    ];
    for (label, source, holder, capture, replays) in conditional_holders {
        let structure = source_mask(braced_scope_after(source, &format!("struct {holder}")));
        assert!(
            structure.contains("Option<CapturedGemmGraphPlan>") && structure.contains("GemmRoute"),
            "{label} must retain route and optional physical plan storage"
        );
        let capture = source_mask(method_scope_for_type(source, holder, capture));
        assert_code_contains_all(
            &capture,
            &["capture_into_graph_with_gemm_plan"],
            &format!("{label} scoped capture"),
        );
        assert!(!capture.contains("require_f32_triad_graph_plan"));
        for replay in replays {
            let replay = source_mask(method_scope_for_type(source, holder, replay));
            assert!(
                replay.contains("with_validated_launch")
                    || replay.contains("launch_mixed_native_graph"),
                "{label}::{replay} must reach scoped validation"
            );
            assert!(
                graph_launches_are_guarded(&replay),
                "{label}::{replay} may not launch outside the validated closure"
            );
        }
    }

    let legacy_capture = source_mask(method_scope_for_type(
        INFERENCE_SOURCE,
        "GpuMambaInferenceMixed",
        "capture_graph",
    ));
    assert!(
        legacy_capture.contains("capture_into_graph")
            && !legacy_capture.contains("capture_into_graph_with_gemm_plan"),
        "M1 legacy mixed inference is not a Triad graph-plan capture"
    );

    assert!(
        !source_mask(MAMBA3_INFERENCE_SOURCE).contains("CapturedGemmGraphPlan"),
        "M3 inference uses context-free cuBLAS and must not receive a fake Triad plan"
    );
    assert_code_contains_all(
        MAMBA3_INFERENCE_SOURCE,
        &["sgemm_no_bias", "gpu_gemm_typed_raw_no_bias"],
        "M3 inference exclusion",
    );

    for (holder, helper) in [
        ("GpuMambaInference", "launch_captured_graph"),
        ("GpuMambaInferenceMixed", "launch_mixed_native_graph"),
    ] {
        let scope = method_scope_for_type(INFERENCE_SOURCE, holder, helper);
        assert_code_contains_all(
            scope,
            &["with_validated_launch"],
            &format!("{holder} shared replay helper"),
        );
        assert!(
            graph_launches_are_guarded(scope),
            "{holder}::{helper} may not launch outside the validated closure"
        );
    }
}

#[test]
fn typed_fallback_records_scalar_routes_without_reading_f32_policy() {
    for marker in [
        "fn bi_sgemm_forward_typed",
        "fn bi_sgemm_backward_dw_typed",
        "fn bi_sgemm_backward_dx_typed",
    ] {
        let scope = source_mask(braced_scope_after(BLAS_SOURCE, marker));
        assert!(
            scope.contains("record_resolved_gemm_route"),
            "{marker} must record every scalar f32 fallback launch"
        );
        assert!(
            !scope.contains("f32_triad_policy") && !scope.contains("AllowDeterministicTf32V1"),
            "{marker} typed numeric contract must ignore f32 TF32 policy"
        );
    }
}

#[test]
fn f32_triad_capture_path_is_prepared_and_records_scalar_fallback_nodes() {
    assert_contains_all(
        LAUNCH_SOURCE,
        &[
            "pub fn prepare_f32_triad(",
            "request: F32TriadRequest",
            "operands: F32TriadOperands",
            "Result<PreparedF32TriadLaunch, String>",
            "pub unsafe fn launch_prepared_f32_triad(",
            "record_resolved_gemm_route",
        ],
        "prepared f32 triad launch path",
    );
    let launch = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub unsafe fn launch_prepared_f32_triad",
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
fn tf32_sources_export_the_exact_planned_symbol_inventories() {
    let inventories = [
        (
            SM80_SOURCE,
            "sgemm_bi_",
            "_sm80_mma_tf32_v1_",
            expected_sm80_symbols(),
            15,
        ),
        (
            SM90A_SOURCE,
            "sgemm_bi_",
            "_sm90a_wgmma_tf32_v1_",
            expected_sm90a_symbols(),
            6,
        ),
        (
            SM100_SOURCE,
            "sgemm_bi_",
            "_sm100_tcgen_tf32_v1_",
            expected_sm100_symbols(),
            36,
        ),
        (
            SM120_SOURCE,
            "sgemm_bi_",
            "_sm120_tma_mma_tf32_v1_",
            expected_sm120_symbols(),
            12,
        ),
    ];

    for (source, prefix, family, expected, count) in inventories {
        let source = source_mask(source);
        let actual: BTreeSet<_> = identifiers_with_prefix(&source, prefix)
            .into_iter()
            .filter(|symbol| symbol.contains(family))
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
            "15",
            "6",
            "36",
            "12",
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
        module_code.contains("validate_tf32_ptx_inventory(module_kind, ptx")
            || module_code.contains("validate_tf32_ptx_inventory(request.module_kind, ptx"),
        "module admission must call the compiled-PTX inventory validator"
    );
    let module_tests = braced_scope_after(MODULE_SOURCE, "#[cfg(test)]");
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
fn cuda_and_rust_tf32_kernel_parameter_layouts_match() {
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
        (SM80_SOURCE, "_sm80_mma_tf32_v1_"),
        (SM90A_SOURCE, "_sm90a_wgmma_tf32_v1_"),
        (SM100_SOURCE, "_sm100_tcgen_tf32_v1_"),
        (SM120_SOURCE, "_sm120_tma_mma_tf32_v1_"),
    ] {
        let code = source_mask(source);
        assert!(
            code.contains("TF32_ASSERT_KERNEL_SIGNATURE"),
            "{family} must compile decltype(&symbol) signature assertions"
        );
        let symbols = identifiers_with_prefix(&code, "sgemm_bi_")
            .into_iter()
            .filter(|symbol| symbol.contains(family));
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
            compile_tf32_ptx(tf32_cuda_blob(SM120_SOURCE, true), "compute_120"),
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
        let actual = ptx_entry_symbols(&ptx, "_tf32_v1_");
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
            if symbol.contains("_nn_") {
                assert!(
                    entry.contains("mul.rn.f32") && entry.contains("fma.rn.f32"),
                    "{label} {symbol} NN epilogue must keep alpha multiply and beta FMA"
                );
            } else if symbol.contains("_tn_") {
                assert!(
                    entry.contains("fma.rn.f32") && !entry.contains("mul.rn.f32"),
                    "{label} {symbol} TN epilogue must use only its RN alpha FMA"
                );
            } else if symbol.contains("_nt_") {
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
fn portable_sm80_tf32_uses_rna_m16n8k8_and_frozen_shared_bank_maps() {
    assert_contains_all(
        SM80_SOURCE,
        &["sgb_tf32_rna", "sgb_tf32_mma_m16n8k8", "bk32"],
        "portable SM80 TF32 mainloop",
    );
    for forbidden in [
        "cvt.rn.tf32.f32",
        "cvt.rz.tf32.f32",
        "cvt.rna.ftz.tf32.f32",
        "cvt.rna.satfinite.tf32.f32",
        "m16n8k4.row.col.f32.tf32",
    ] {
        assert!(
            !source_mask(SM80_SOURCE).contains(forbidden),
            "portable TF32 source contains forbidden {forbidden}"
        );
    }

    let mut a_banks = BTreeSet::new();
    let mut b_banks = BTreeSet::new();
    for lane in 0..32 {
        let group = lane >> 2;
        let thread = lane & 3;
        assert!(a_banks.insert((4 * group + thread) % 32));
        assert!(b_banks.insert((8 * thread + group) % 32));
    }
    assert_eq!(a_banks.len(), 32);
    assert_eq!(b_banks.len(), 32);
    assert_contains_all(
        SM80_SOURCE,
        &["36", "72", "40", "55296", "82944"],
        "portable TF32 padded shared layouts",
    );
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
    const K_CASES: [usize; 12] = [0, 1, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33];
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
        ("SM80", SM80_SOURCE, "_sm80_mma_tf32_v1_"),
        ("SM90a", SM90A_SOURCE, "_sm90a_wgmma_tf32_v1_"),
        ("SM100", SM100_SOURCE, "_sm100_tcgen_tf32_v1_"),
        ("SM120", SM120_SOURCE, "_sm120_tma_mma_tf32_v1_"),
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
            "ZeroReductionV1",
            "EncodedV1",
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

    let preparation = source_mask(braced_scope_after(
        LAUNCH_SOURCE,
        "pub fn prepare_f32_triad",
    ));
    let zero_marker = preparation
        .find("reduction() == 0")
        .or_else(|| preparation.find("reduction == 0"))
        .expect("prepare_f32_triad must branch on the normalized reduction");
    let zero_branch = braced_scope_after(&preparation[zero_marker..], "if ");
    assert_code_contains_all(
        zero_branch,
        &[
            "ZeroReductionV1",
            "zeroed_tensor_map_sentinel",
            "return Ok(",
        ],
        "K=0 mapless preparation branch",
    );
    for forbidden in [
        "tensor_map_plan",
        "cuTensorMapEncodeTiled",
        "AllocationIdentity::query",
        "operands.a",
        "operands.b",
    ] {
        assert!(
            !zero_branch.contains(forbidden),
            "K=0 mapless preparation may not perform {forbidden}"
        );
    }

    let sentinel = unsafe {
        std::mem::MaybeUninit::<cudarc::driver::sys::CUtensorMap>::zeroed().assume_init()
    };
    assert_eq!(std::mem::size_of_val(&sentinel), 128);
    assert!(matches!(std::mem::align_of_val(&sentinel), 64 | 128));
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&sentinel as *const cudarc::driver::sys::CUtensorMap).cast::<u8>(),
            std::mem::size_of_val(&sentinel),
        )
    };
    assert!(bytes.iter().all(|&byte| byte == 0));

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
    let launch_tests = braced_scope_after(LAUNCH_SOURCE, "#[cfg(test)]");
    let no_query = source_mask(braced_scope_after(
        launch_tests,
        "fn zero_reduction_preparation_never_queries_input_allocations",
    ));
    assert_code_contains_all(
        &no_query,
        &[
            "a_input_queries",
            "b_input_queries",
            "tensor_map_plan_queries",
            "tensor_map_encodes",
            "ZeroReductionV1",
            "assert_eq!(a_input_queries.get(), 0)",
            "assert_eq!(b_input_queries.get(), 0)",
            "assert_eq!(tensor_map_plan_queries.get(), 0)",
            "assert_eq!(tensor_map_encodes.get(), 0)",
        ],
        "behavioral zero-reduction triple-counter proof",
    );

    let identity_tests = braced_scope_after(IDENTITY_SOURCE, "#[cfg(test)]");
    let domain = source_mask(braced_scope_after(
        identity_tests,
        "fn zero_reduction_identity_is_domain_separated_and_pointer_free",
    ));
    assert_code_contains_all(
        &domain,
        &[
            "ZeroReductionV1",
            "EncodedV1",
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
    assert_eq!(specialized.len(), 54, "canonical specialized K=0 census");
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
    let decode = braced_scope_after(SM120_SOURCE, "sm120_tf32_sw128_offset");
    assert_contains_all(
        decode,
        &[
            "element / 4",
            "element & 3",
            "plane_base / 128",
            "% 8",
            "logical_row",
            "^",
            "* 16",
            "* 4",
        ],
        "SM120 production SW128 decode",
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
    assert_contains_all(
        SM120_SOURCE,
        &["Sm120Nn", "Sm120Tn", "Sm120Nt", "sm120_tf32_sw128_offset"],
        "SM120 NN/TN/NT register loads use the production SW128 decode",
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
    for (name, source, helper) in [
        ("SM80", SM80_SOURCE, "sgb_tf32_epilogue"),
        ("SM90a", SM90A_SOURCE, "sm90a_tf32_epilogue"),
        ("SM100", SM100_SOURCE, "sm100_tf32_epilogue"),
        ("SM120", SM120_SOURCE, "sm120_tf32_epilogue"),
    ] {
        assert_contains_all(
            source,
            &["alpha == 1.0f", "bias == nullptr", "__fmul_rn", "__fmaf_rn"],
            &format!("{name} TF32 epilogue"),
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
        &["OneCtaPerOutputTileV1", "ownership"],
        "single-CTA resolved route identity",
    );
    let operands = braced_scope_after(LAUNCH_SOURCE, "fn validate_f32_triad_operands");
    assert_contains_all(
        operands,
        &[
            "output",
            "is_null",
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
    let portable_stage = braced_scope_after(SM80_SOURCE, "sgb_tf32_stage_async");
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
    assert!(nvrtc >= (13, 2), "SM110 compile gate requires NVRTC 13.2+");
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
            ptx_entry_symbols(&ptx, "_sm100_tcgen_tf32_v1_"),
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
    assert!(
        loaded_nvrtc_version() >= (13, 2),
        "release feature-target matrix requires CUDA 13.2+"
    );
    let mut checked_entries = 0;
    for (nvrtc_target, ptx_target, family) in RELEASE_TARGET_MATRIX {
        let (label, source, needs_mma16, expected) = specialized_family_contract(*family);
        let ptx = compile_tf32_ptx(tf32_cuda_blob(source, needs_mma16), nvrtc_target);
        assert!(
            ptx.lines()
                .any(|line| line.trim() == format!(".target {ptx_target}")),
            "{nvrtc_target} must emit exact PTX target {ptx_target}"
        );
        let family_marker = match family {
            SpecializedTf32Family::Sm90a => "_sm90a_wgmma_tf32_v1_",
            SpecializedTf32Family::Sm100 => "_sm100_tcgen_tf32_v1_",
            SpecializedTf32Family::Sm120 => "_sm120_tma_mma_tf32_v1_",
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
    assert_eq!(checked_entries, 246);
    assert_eq!(checked_entries, release_entry_target_count());
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
        &[
            "ExactScalarFmaV1",
            "AllowDeterministicTf32V1",
            "ScalarFmaV1",
        ],
        "exact/allow TF32 dispatch",
    );

    assert_eq!(
        mamba_rs::mamba_ssm::gpu::device::GpuDevice::resolve_nvrtc_target((11, 0)),
        Ok("sm_110"),
        "SM110 portable target support is mandatory"
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
            "MmaTf32RnaV1",
            "Sm90aWgmmaTf32TmaV1",
            "Sm100Tcgen05Tf32TmaV1",
            "Sm120TmaMmaTf32RnaV1",
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
            && !forced.contains("ScalarFmaV1")
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
    let symbol = "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s2_c4";
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
fn k0_provenance_kills_overwritten_registers() {
    let symbol = "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s2_c4";
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
    let symbol = "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s2_c4";
    let source = "#line 1001 \"mamba_tf32_k0_guard\"\n#line 1002 \"mamba_tf32_k0_branch\"\n#line 2001 \"mamba_tf32_k0_zero_store\"\n";
    let valid = format!(
        r#"digraph "{symbol}" {{
"entry" [label="0000: ISETP; 0010: @P0 BRA;"];
"entry" -> "zero";
"entry" -> "main";
"zero" [label="0020: STG; 0030: EXIT;"];
"main" [label="0040: UTMALDG; 0050: UTCATOMSWS.FIND_AND_SET.ALIGN; 0060: ATOMS.OR; 0070: ATOMS.OR; 0080: UTCHMMA;"];
"main" -> "dealloc";
"dealloc" [label="0090: UTCATOMSWS.AND; 00a0: EXIT;"];
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
         /*0080*/ UTCHMMA.16816 ;\n\
         /*0090*/ UTCATOMSWS.AND URZ, UR4 ;\n\
         /*00a0*/ EXIT ;\n"
    );
    assert_sass_cfg_corroboration(&valid, &line_sass, source, symbol, "SM100");
    let bypass = valid.replace("0020: STG;", "0020: UTCHMMA; 0028: STG;");
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&bypass, &line_sass, source, symbol, "SM100")
        })
        .is_err()
    );
    let unbound = line_sass.replace("mamba_tf32_k0_zero_store", "foreign_safe_store");
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&valid, &unbound, source, symbol, "SM100")
        })
        .is_err()
    );
    let duplicate_anchor = line_sass.replace(
        "/*0040*/ UTMALDG.2D ;",
        "//## File \"/root/mamba_tf32_k0_zero_store\", line 2001\n\
         /*0040*/ STG.E [R8.64], R9 ;",
    );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&valid, &duplicate_anchor, source, symbol, "SM100")
        })
        .is_err()
    );
    let foreign_branch = line_sass.replace("mamba_tf32_k0_branch", "foreign_k0_branch");
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&valid, &foreign_branch, source, symbol, "SM100")
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
    let bad_deallocation = valid
        .replace("0080: UTCHMMA;", "0080: UTCHMMA; 0090: UTCATOMSWS.AND;")
        .replace(
            "\"dealloc\" [label=\"0090: UTCATOMSWS.AND; 00a0: EXIT;\"]",
            "\"dealloc\" [label=\"00a0: EXIT;\"]",
        );
    assert!(
        std::panic::catch_unwind(|| {
            assert_sass_cfg_corroboration(&bad_deallocation, &line_sass, source, symbol, "SM100")
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
    let symbol = "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s2_c4";
    let valid = format!(
        "Function : {symbol}\n\
         //## File \"/root/tf32.cu\", line 1\n\
         /*0000*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;\n\
         /*0010*/ ATOMS.OR RZ, [R7+0x14], R8 ;\n\
         /*0020*/ ATOMS.OR RZ, [R7+0x18], R9 ;\n\
         /*0030*/ UTCATOMSWS.AND URZ, UR4 ;\n\
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
    let symbol = "sgemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4";
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
        (
            "sgemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s2",
            (192, 55_296),
        ),
        (
            "sgemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s3",
            (192, 82_944),
        ),
        ("sgemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s2", (128, 36_864)),
        ("sgemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s3", (128, 55_296)),
        ("sgemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4", (96, 29_696)),
        (
            "sgemm_bi_nn_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg1",
            (168, 73_984),
        ),
        (
            "sgemm_bi_nn_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg2",
            (128, 73_984),
        ),
        (
            "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s2_c4",
            (128, 49_408),
        ),
        (
            "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n64_bk32_s4_p8",
            (128, 98_560),
        ),
        (
            "sgemm_bi_nn_sm100_tcgen_tf32_v1_m128n128_bk32_s3_c4",
            (128, 98_560),
        ),
        (
            "sgemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2",
            (128, 49_280),
        ),
        (
            "sgemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3",
            (128, 73_856),
        ),
    ] {
        assert_eq!(tf32_resource_caps(symbol), expected, "{symbol}");
    }
}

fn require_exact_cc(expected: (u32, u32)) {
    let device = mamba_rs::mamba_ssm::gpu::device::GpuDevice::new(0)
        .expect("open the exact qualification GPU");
    assert_eq!(
        device.compute_capability, expected,
        "wrong qualification CC"
    );
}

fn checked_output(mut command: Command, label: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("run {label}: {error}"));
    assert!(
        output.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[derive(Debug, PartialEq)]
enum StrictJsonValue {
    Null,
    Bool(bool),
    Number(String),
    String {
        value: String,
        raw_content: (usize, usize),
    },
    Array(Vec<StrictJsonValue>),
    Object(BTreeMap<String, StrictJsonValue>),
}

struct StrictJsonParser<'a> {
    source: &'a str,
    cursor: usize,
}

impl<'a> StrictJsonParser<'a> {
    fn parse(mut self) -> Result<StrictJsonValue, String> {
        self.skip_whitespace();
        let value = self.value(0)?;
        self.skip_whitespace();
        if self.cursor != self.source.len() {
            return Err(format!("trailing JSON data at byte {}", self.cursor));
        }
        if !matches!(value, StrictJsonValue::Object(_)) {
            return Err("qualification JSON root must be an object".to_owned());
        }
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<StrictJsonValue, String> {
        if depth > 128 {
            return Err("JSON nesting exceeds 128 levels".to_owned());
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'n') => {
                self.keyword(b"null")?;
                Ok(StrictJsonValue::Null)
            }
            Some(b't') => {
                self.keyword(b"true")?;
                Ok(StrictJsonValue::Bool(true))
            }
            Some(b'f') => {
                self.keyword(b"false")?;
                Ok(StrictJsonValue::Bool(false))
            }
            Some(b'"') => {
                let (value, raw_content) = self.string()?;
                Ok(StrictJsonValue::String { value, raw_content })
            }
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number().map(StrictJsonValue::Number),
            Some(byte) => Err(format!(
                "unexpected JSON byte 0x{byte:02x} at byte {}",
                self.cursor
            )),
            None => Err("unexpected end of JSON".to_owned()),
        }
    }

    fn array(&mut self, depth: usize) -> Result<StrictJsonValue, String> {
        self.cursor += 1;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.take(b']') {
            return Ok(StrictJsonValue::Array(values));
        }
        loop {
            values.push(self.value(depth)?);
            self.skip_whitespace();
            if self.take(b']') {
                return Ok(StrictJsonValue::Array(values));
            }
            self.expect(b',')?;
        }
    }

    fn object(&mut self, depth: usize) -> Result<StrictJsonValue, String> {
        self.cursor += 1;
        let mut fields = BTreeMap::new();
        self.skip_whitespace();
        if self.take(b'}') {
            return Ok(StrictJsonValue::Object(fields));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(format!("JSON object key expected at byte {}", self.cursor));
            }
            let (key, _) = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value(depth)?;
            if fields.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate JSON object key {key:?}"));
            }
            self.skip_whitespace();
            if self.take(b'}') {
                return Ok(StrictJsonValue::Object(fields));
            }
            self.expect(b',')?;
        }
    }

    fn string(&mut self) -> Result<(String, (usize, usize)), String> {
        self.expect(b'"')?;
        let raw_start = self.cursor;
        let mut segment_start = self.cursor;
        let mut decoded = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err("unterminated JSON string".to_owned());
            };
            match byte {
                b'"' => {
                    decoded.push_str(&self.source[segment_start..self.cursor]);
                    let raw_end = self.cursor;
                    self.cursor += 1;
                    return Ok((decoded, (raw_start, raw_end)));
                }
                b'\\' => {
                    decoded.push_str(&self.source[segment_start..self.cursor]);
                    self.cursor += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| "unterminated JSON escape".to_owned())?;
                    self.cursor += 1;
                    match escape {
                        b'"' => decoded.push('"'),
                        b'\\' => decoded.push('\\'),
                        b'/' => decoded.push('/'),
                        b'b' => decoded.push('\u{0008}'),
                        b'f' => decoded.push('\u{000c}'),
                        b'n' => decoded.push('\n'),
                        b'r' => decoded.push('\r'),
                        b't' => decoded.push('\t'),
                        b'u' => decoded.push(self.unicode_escape()?),
                        _ => {
                            return Err(format!("invalid JSON escape at byte {}", self.cursor - 1));
                        }
                    }
                    segment_start = self.cursor;
                }
                0x00..=0x1f => {
                    return Err(format!(
                        "unescaped control byte in JSON string at byte {}",
                        self.cursor
                    ));
                }
                0x20..=0x7f => self.cursor += 1,
                _ => {
                    let character = self.source[self.cursor..]
                        .chars()
                        .next()
                        .ok_or_else(|| "invalid UTF-8 in JSON string".to_owned())?;
                    self.cursor += character.len_utf8();
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, String> {
        let high = self.hex_quad()?;
        let scalar = if (0xd800..=0xdbff).contains(&high) {
            if self.source.as_bytes().get(self.cursor..self.cursor + 2) != Some(b"\\u") {
                return Err("high surrogate without a low surrogate".to_owned());
            }
            self.cursor += 2;
            let low = self.hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&low) {
                return Err("high surrogate followed by an invalid low surrogate".to_owned());
            }
            0x1_0000 + ((u32::from(high) - 0xd800) << 10) + (u32::from(low) - 0xdc00)
        } else if (0xdc00..=0xdfff).contains(&high) {
            return Err("orphan JSON low surrogate".to_owned());
        } else {
            u32::from(high)
        };
        char::from_u32(scalar).ok_or_else(|| "invalid JSON Unicode scalar".to_owned())
    }

    fn hex_quad(&mut self) -> Result<u16, String> {
        let end = self
            .cursor
            .checked_add(4)
            .filter(|end| *end <= self.source.len())
            .ok_or_else(|| "short JSON Unicode escape".to_owned())?;
        let mut value = 0_u16;
        for byte in self.source.as_bytes()[self.cursor..end].iter().copied() {
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a' + 10),
                b'A'..=b'F' => u16::from(byte - b'A' + 10),
                _ => return Err("invalid hex digit in JSON Unicode escape".to_owned()),
            };
            value = value * 16 + digit;
        }
        self.cursor = end;
        Ok(value)
    }

    fn number(&mut self) -> Result<String, String> {
        let start = self.cursor;
        self.take(b'-');
        match self.peek() {
            Some(b'0') => {
                self.cursor += 1;
                if self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                    return Err("JSON number has a leading zero".to_owned());
                }
            }
            Some(b'1'..=b'9') => self.take_digits(),
            _ => return Err(format!("invalid JSON number at byte {start}")),
        }
        if self.take(b'.') {
            if !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                return Err("JSON fraction requires a digit".to_owned());
            }
            self.take_digits();
        }
        if self.peek().is_some_and(|byte| matches!(byte, b'e' | b'E')) {
            self.cursor += 1;
            if self.peek().is_some_and(|byte| matches!(byte, b'+' | b'-')) {
                self.cursor += 1;
            }
            if !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                return Err("JSON exponent requires a digit".to_owned());
            }
            self.take_digits();
        }
        Ok(self.source[start..self.cursor].to_owned())
    }

    fn take_digits(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.cursor += 1;
        }
    }

    fn keyword(&mut self, keyword: &[u8]) -> Result<(), String> {
        if self
            .source
            .as_bytes()
            .get(self.cursor..self.cursor + keyword.len())
            == Some(keyword)
        {
            self.cursor += keyword.len();
            Ok(())
        } else {
            Err(format!("invalid JSON keyword at byte {}", self.cursor))
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        if self.take(expected) {
            Ok(())
        } else {
            Err(format!(
                "expected JSON byte 0x{expected:02x} at byte {}",
                self.cursor
            ))
        }
    }

    fn take(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .peek()
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.as_bytes().get(self.cursor).copied()
    }
}

fn parse_strict_json(source: &str) -> Result<StrictJsonValue, String> {
    StrictJsonParser { source, cursor: 0 }.parse()
}

fn strict_json_object(
    value: &StrictJsonValue,
) -> Result<&BTreeMap<String, StrictJsonValue>, String> {
    match value {
        StrictJsonValue::Object(fields) => Ok(fields),
        _ => Err("qualification JSON root must be an object".to_owned()),
    }
}

fn strict_json_string<'a>(
    fields: &'a BTreeMap<String, StrictJsonValue>,
    field: &str,
) -> Result<(&'a str, (usize, usize)), String> {
    match fields.get(field) {
        Some(StrictJsonValue::String { value, raw_content }) => Ok((value, *raw_content)),
        Some(_) => Err(format!("qualification field {field} must be a string")),
        None => Err(format!("qualification field {field} is missing")),
    }
}

fn strict_json_u64(fields: &BTreeMap<String, StrictJsonValue>, field: &str) -> Result<u64, String> {
    match fields.get(field) {
        Some(StrictJsonValue::Number(number)) => number
            .parse()
            .map_err(|_| format!("qualification field {field} must be an unsigned integer")),
        Some(_) => Err(format!("qualification field {field} must be a number")),
        None => Err(format!("qualification field {field} is missing")),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn verify_qualification_digests(report: &str, artifact: &[u8]) -> Result<(), String> {
    let parsed = parse_strict_json(report)?;
    let fields = strict_json_object(&parsed)?;
    let (artifact_digest, artifact_span) = strict_json_string(fields, "artifact_digest")?;
    if &report[artifact_span.0..artifact_span.1] != artifact_digest {
        return Err("artifact_digest must be an unescaped canonical string".to_owned());
    }
    let recomputed_artifact = sha256_hex(artifact);
    if artifact_digest != recomputed_artifact {
        return Err(format!(
            "artifact digest mismatch: report {artifact_digest}, recomputed {recomputed_artifact}"
        ));
    }

    let (report_digest, report_span) = strict_json_string(fields, "report_digest")?;
    if &report[report_span.0..report_span.1] != report_digest || report_digest.len() != 64 {
        return Err("report_digest must be one unescaped 64-byte string".to_owned());
    }
    let mut zeroed = Vec::with_capacity(report.len());
    zeroed.extend_from_slice(&report.as_bytes()[..report_span.0]);
    zeroed.extend_from_slice(&[b'0'; 64]);
    zeroed.extend_from_slice(&report.as_bytes()[report_span.1..]);
    let recomputed_report = sha256_hex(&zeroed);
    if report_digest != recomputed_report {
        return Err(format!(
            "report integrity mismatch: report {report_digest}, recomputed {recomputed_report}"
        ));
    }
    Ok(())
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
        r#"{"artifact_digest":"d25252040204953b4a9926344bf5de38d5bbd36d01e71eb25b4c68a535f99248","report_digest":""#,
        "81cb42897e6148e3486c6ec20e241789d0df30923505a2bd93ec37833e00fdb4",
        r#""}"#,
    );
    assert!(verify_qualification_digests(report, b"artifact-v1").is_ok());
    assert!(verify_qualification_digests(report, b"artifact-v2").is_err());
    let tampered = report.replacen('{', r#"{"extra":true,"#, 1);
    assert!(verify_qualification_digests(&tampered, b"artifact-v1").is_err());
}

fn json_string_field<'a>(fields: &'a BTreeMap<String, StrictJsonValue>, field: &str) -> &'a str {
    strict_json_string(fields, field)
        .unwrap_or_else(|error| panic!("{error}"))
        .0
}

fn assert_hex_digest(fields: &BTreeMap<String, StrictJsonValue>, field: &str) {
    let digest = json_string_field(fields, field);
    assert_eq!(digest.len(), 64, "{field} must contain 256 bits");
    assert!(
        digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must be canonical lowercase hexadecimal: {digest}"
    );
    assert!(
        digest.bytes().any(|byte| byte != b'0'),
        "{field} may not use the all-zero sentinel"
    );
}

fn assert_driver_abi_proof(proof: &str, expected: &BTreeSet<String>, tensor_map_alignment: usize) {
    let mut lines = proof.lines();
    assert_eq!(
        lines.next(),
        Some("MambaBiTf32DriverAbiV1"),
        "driver ABI proof schema"
    );
    let mut records = BTreeMap::new();
    for line in lines {
        let mut fields = line.split('\t');
        let symbol = fields.next().expect("driver ABI symbol");
        let count: usize = fields
            .next()
            .expect("driver ABI parameter count")
            .parse()
            .expect("numeric driver ABI parameter count");
        let layout = fields.next().expect("driver ABI parameter layout");
        assert!(fields.next().is_none(), "extra driver ABI fields: {line}");
        assert_eq!(count, 5, "{symbol} cuFuncGetParamCount");
        let expected_layout = if symbol.contains("_sm80_") {
            "0:8,8:8,16:8,24:8,32:32"
        } else if tensor_map_alignment == 64 {
            "0:8,64:128,192:128,320:8,328:40"
        } else {
            "0:8,128:128,256:128,384:8,392:40"
        };
        assert_eq!(layout, expected_layout, "{symbol} cuFuncGetParamInfo");
        assert!(
            records.insert(symbol.to_owned(), ()).is_none(),
            "duplicate driver ABI proof for {symbol}"
        );
    }
    assert_eq!(
        records.keys().cloned().collect::<BTreeSet<_>>(),
        *expected,
        "driver ABI proof inventory"
    );
}

fn qualification_arguments(
    cc: (u32, u32),
    expected_routes: usize,
    artifact_output: &Path,
    driver_abi_output: &Path,
) -> Vec<String> {
    vec![
        "--exact-cc".to_owned(),
        format!("{}.{}", cc.0, cc.1),
        "--family".to_owned(),
        "all-admitted".to_owned(),
        "--all-routes".to_owned(),
        "--all-m-tiles".to_owned(),
        "--ops".to_owned(),
        "nn,tn,nt".to_owned(),
        "--precision".to_owned(),
        "f32".to_owned(),
        "--preserve-existing-precisions".to_owned(),
        "bf16,f16".to_owned(),
        "--k-cases".to_owned(),
        "0,1,7,8,9,15,16,17,24,31,32,33".to_owned(),
        "--tail-cases".to_owned(),
        "1,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129".to_owned(),
        "--repeat".to_owned(),
        "100".to_owned(),
        "--expected-routes".to_owned(),
        expected_routes.to_string(),
        "--k0-every-symbol".to_owned(),
        "--k0-null-inputs".to_owned(),
        "--driver-abi".to_owned(),
        "--driver-abi-live-query".to_owned(),
        "--driver-abi-proof-output".to_owned(),
        driver_abi_output.display().to_string(),
        "--descriptor-oracles".to_owned(),
        "--sass-cfg".to_owned(),
        "--release-entry-target-checks".to_owned(),
        release_entry_target_count().to_string(),
        "--artifact-output".to_owned(),
        artifact_output.display().to_string(),
        "--report".to_owned(),
        "json".to_owned(),
    ]
}

fn run_hardware_qualification(cc: (u32, u32), expected_routes: usize) {
    require_exact_cc(cc);
    let directory = tempfile::tempdir().expect("TF32 qualification target directory");
    let target = directory.path().join("target");
    let mut build = Command::new("cargo");
    build.args([
        "build",
        "--features",
        "cuda",
        "--bin",
        "gemm-bi-tf32-qualification",
        "--target-dir",
    ]);
    build.arg(&target);
    checked_output(build, "build gemm-bi-tf32-qualification");

    let binary = target.join("debug/gemm-bi-tf32-qualification");
    assert!(binary.is_file(), "missing qualification binary {binary:?}");
    let artifact_output = directory.path().join("qualification-artifact.bin");
    let driver_abi_output = directory.path().join("driver-abi.tsv");
    let arguments =
        qualification_arguments(cc, expected_routes, &artifact_output, &driver_abi_output);
    let mut qualification = Command::new(&binary);
    qualification.args(&arguments);
    let output = checked_output(qualification, "TF32 runtime qualification");
    let report = String::from_utf8(output.stdout).expect("qualification JSON must be UTF-8");
    assert_eq!(
        report.trim(),
        report,
        "qualification report may not contain framing whitespace"
    );
    assert!(
        report.starts_with('{') && report.ends_with('}') && !report.contains('\n'),
        "qualification report must be exactly one JSON object"
    );
    let parsed = parse_strict_json(&report)
        .unwrap_or_else(|error| panic!("invalid qualification JSON: {error}: {report}"));
    let fields = strict_json_object(&parsed).expect("qualification JSON object");
    assert_eq!(
        json_string_field(fields, "schema"),
        "MambaBiTf32QualificationV1"
    );
    assert_eq!(
        json_string_field(fields, "exact_cc"),
        format!("{}.{}", cc.0, cc.1)
    );
    for field in [
        "runtime",
        "correctness",
        "determinism",
        "graph_replay",
        "accuracy",
        "performance",
        "ptx_sass",
        "driver_abi",
        "descriptor_oracles",
        "k0_cfg",
        "sass_cfg",
        "k0_runtime",
        "epilogue_bits",
        "single_owner",
        "tail_canary",
    ] {
        assert_eq!(
            json_string_field(fields, field),
            "pass",
            "qualification gate {field} did not pass"
        );
    }
    for (field, expected) in [
        ("spill_bytes", 0_u64),
        ("numeric_atomics", 0),
        ("k0_a_input_queries", 0),
        ("k0_b_input_queries", 0),
        ("k0_tensor_map_plan_queries", 0),
        ("k0_tensor_map_encodes", 0),
        ("k0_graph_ab_dependencies", 0),
        ("specialized_k0_cfg_entries", 54),
        (
            "release_entry_target_checks",
            release_entry_target_count() as u64,
        ),
        ("routes_qualified", expected_routes as u64),
        ("k0_symbols_qualified", expected_routes as u64),
        ("driver_abi_entries", expected_routes as u64),
    ] {
        assert_eq!(
            strict_json_u64(fields, field).unwrap_or_else(|error| panic!("{error}")),
            expected,
            "qualification count {field}"
        );
    }
    let digest_fields = [
        "artifact_digest",
        "output_digest",
        "zero_reduction_digest",
        "encoded_map_digest",
        "ordered_graph_route_digest",
        "report_digest",
    ];
    for field in digest_fields {
        assert_hex_digest(fields, field);
    }
    for (index, field) in digest_fields.iter().enumerate() {
        for other in &digest_fields[index + 1..] {
            assert_ne!(
                json_string_field(fields, field),
                json_string_field(fields, other),
                "{field} and {other} must use distinct digest domains"
            );
        }
    }
    let artifact = std::fs::read(&artifact_output)
        .unwrap_or_else(|error| panic!("read qualification artifact {artifact_output:?}: {error}"));
    verify_qualification_digests(&report, &artifact)
        .unwrap_or_else(|error| panic!("qualification digest verification failed: {error}"));
    let expected_symbols = expected_hardware_symbols(cc);
    assert_eq!(expected_symbols.len(), expected_routes);
    let driver_abi_proof = std::fs::read_to_string(&driver_abi_output)
        .unwrap_or_else(|error| panic!("read driver ABI proof {driver_abi_output:?}: {error}"));
    let tensor_map_alignment = match loaded_nvrtc_version().0 {
        12 => 64,
        13 => 128,
        version => panic!("unsupported CUDA tensor-map ABI major {version}"),
    };
    assert_driver_abi_proof(&driver_abi_proof, &expected_symbols, tensor_map_alignment);
    let runner_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/gemm_bi_tf32_qualification.rs"),
    )
    .expect("read production TF32 qualification runner");
    assert_code_contains_all(
        &runner_source,
        &["cuFuncGetParamCount", "cuFuncGetParamInfo"],
        "live Driver API ABI qualification",
    );

    for tool in ["memcheck", "racecheck", "initcheck", "synccheck"] {
        let mut sanitizer = Command::new("compute-sanitizer");
        sanitizer.args(["--error-exitcode", "99", "--tool", tool]);
        sanitizer.arg(&binary);
        sanitizer.args(&arguments);
        sanitizer.args(["--suite", "sanitizer"]);
        checked_output(sanitizer, &format!("compute-sanitizer {tool}"));
    }
}

#[test]
#[ignore = "requires exact CC 8.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm80_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 0), 15);
}

#[test]
#[ignore = "requires exact CC 8.6 and the full portable TF32 qualification corpus"]
fn hardware_sm86_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 6), 15);
}

#[test]
#[ignore = "requires exact CC 8.7 and the full portable TF32 qualification corpus"]
fn hardware_sm87_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 7), 15);
}

#[test]
#[ignore = "requires exact Ada CC 8.9 and the full portable TF32 qualification corpus"]
fn hardware_sm89_ada_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 9), 15);
}

#[test]
#[ignore = "requires exact CC 9.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm90a_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((9, 0), 21);
}

#[test]
#[ignore = "requires exact CC 10.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm100_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((10, 0), 51);
}

#[test]
#[ignore = "requires exact CC 10.3 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm103_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((10, 3), 51);
}

#[test]
#[ignore = "requires exact CC 11.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm110_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((11, 0), 51);
}

#[test]
#[ignore = "requires exact CC 12.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm120_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((12, 0), 27);
}

#[test]
#[ignore = "requires exact CC 12.1 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm121_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((12, 1), 27);
}
