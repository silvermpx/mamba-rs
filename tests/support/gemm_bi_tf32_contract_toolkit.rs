//! Shared toolkit of the TF32 contract: the source scanner, the strict
//! JSON reader and digest verifier for the qualification report, the
//! expected symbol census per architecture and the qualification command
//! lines. tests/gemm_bi_tf32_contract.rs exercises it on the host; the
//! hardware gate under tools/qualification runs the same checks against a
//! live board.

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Output};

pub(crate) const MODULE_SOURCE: &str =
    include_str!("../../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
pub(crate) fn assert_contains_all(source: &str, required: &[&str], contract: &str) {
    let mask = source_mask(source);
    for item in required {
        assert!(mask.contains(item), "{contract} is missing {item}");
    }
}
pub(crate) fn assert_code_contains_all(source: &str, required: &[&str], contract: &str) {
    assert_contains_all(source, required, contract);
}
pub(crate) fn braced_scope_at<'a>(source: &'a str, start: usize, label: &str) -> &'a str {
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
pub(crate) fn token_present(source: &str, token: &str) -> bool {
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
pub(crate) fn impl_scopes_for_type<'a>(source: &'a str, type_name: &str) -> Vec<&'a str> {
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
pub(crate) fn method_scope_for_type<'a>(source: &'a str, type_name: &str, method: &str) -> &'a str {
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
    let implementation = matching[0].0;
    let mask = source_mask(implementation);
    function_scope_at(implementation, &mask, matching[0].1, method)
        .unwrap_or_else(|error| panic!("{error}"))
}
pub(crate) fn matching_delimiter(
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
pub(crate) fn source_mask(source: &str) -> String {
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
pub(crate) fn looks_like_character_literal(source: &[u8], quote: usize) -> bool {
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
pub(crate) fn skip_ascii_whitespace(source: &str, mut cursor: usize) -> usize {
    while source
        .as_bytes()
        .get(cursor)
        .is_some_and(u8::is_ascii_whitespace)
    {
        cursor += 1;
    }
    cursor
}
pub(crate) fn identifier_end(source: &str, mut cursor: usize) -> usize {
    while source
        .as_bytes()
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'#')
    {
        cursor += 1;
    }
    cursor
}
pub(crate) fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'#'
}
pub(crate) fn token_at(source: &str, offset: usize, token: &str) -> bool {
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
pub(crate) fn token_offsets(source: &str, token: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find(token) {
        let offset = cursor + relative;
        cursor = offset + token.len();
        if token_at(source, offset, token) {
            offsets.push(offset);
        }
    }
    offsets
}
pub(crate) fn marker_offsets_at_brace_depth(
    source: &str,
    marker: &str,
    expected_depth: u32,
) -> Vec<usize> {
    let mask = source_mask(source);
    let mut offsets = Vec::new();
    let mut depth = 0_u32;
    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut cursor = 0;
    while cursor < mask.len() {
        if depth == expected_depth
            && parentheses == 0
            && brackets == 0
            && mask[cursor..].starts_with(marker)
            && token_at(
                &mask,
                cursor,
                marker.split_ascii_whitespace().next().unwrap(),
            )
        {
            offsets.push(cursor);
        }
        match mask.as_bytes()[cursor] {
            b'{' => depth += 1,
            b'}' => depth = depth.checked_sub(1).expect("balanced source braces"),
            b'(' => parentheses += 1,
            b')' => {
                parentheses = parentheses
                    .checked_sub(1)
                    .expect("balanced source parentheses")
            }
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1).expect("balanced source brackets"),
            _ => {}
        }
        cursor += 1;
    }
    offsets
}
pub(crate) fn function_body_open(
    source: &str,
    function_start: usize,
    name: &str,
) -> Result<usize, String> {
    let name_start = skip_ascii_whitespace(source, function_start + "fn".len());
    let name_end = identifier_end(source, name_start);
    let mut cursor = skip_ascii_whitespace(source, name_end);
    if source.as_bytes().get(cursor) == Some(&b'<') {
        let mut angles = 0_u32;
        loop {
            let Some(byte) = source.as_bytes().get(cursor).copied() else {
                return Err(format!(
                    "function {name} has unterminated generic parameters"
                ));
            };
            match byte {
                b'<' => angles += 1,
                b'>' if source.as_bytes().get(cursor.wrapping_sub(1)) != Some(&b'-') => {
                    angles = angles
                        .checked_sub(1)
                        .ok_or_else(|| format!("function {name} has unbalanced generics"))?;
                    if angles == 0 {
                        cursor += 1;
                        break;
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
    }
    cursor = skip_ascii_whitespace(source, cursor);
    if source.as_bytes().get(cursor) != Some(&b'(') {
        return Err(format!("function {name} is missing its parameter list"));
    }
    cursor = matching_delimiter(source, cursor, b'(', b')')
        .ok_or_else(|| format!("function {name} has an unterminated parameter list"))?
        + 1;

    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut angles = 0_u32;
    while cursor < source.len() {
        match source.as_bytes()[cursor] {
            b'(' => parentheses += 1,
            b')' => {
                parentheses = parentheses
                    .checked_sub(1)
                    .ok_or_else(|| format!("function {name} has an unbalanced signature"))?
            }
            b'[' => brackets += 1,
            b']' => {
                brackets = brackets
                    .checked_sub(1)
                    .ok_or_else(|| format!("function {name} has an unbalanced signature"))?
            }
            b'<' => angles += 1,
            b'>' if source.as_bytes().get(cursor.wrapping_sub(1)) != Some(&b'-') => {
                angles = angles
                    .checked_sub(1)
                    .ok_or_else(|| format!("function {name} has an unbalanced signature"))?
            }
            b'{' => {
                let bang = source[..cursor]
                    .bytes()
                    .rposition(|byte| !byte.is_ascii_whitespace())
                    .filter(|offset| source.as_bytes()[*offset] == b'!');
                let macro_group = bang.is_some_and(|bang| {
                    source[..bang]
                        .bytes()
                        .rposition(|byte| !byte.is_ascii_whitespace())
                        .is_some_and(|offset| is_identifier_byte(source.as_bytes()[offset]))
                });
                if parentheses == 0 && brackets == 0 && angles == 0 && !macro_group {
                    return Ok(cursor);
                }
                cursor = matching_delimiter(source, cursor, b'{', b'}').ok_or_else(|| {
                    format!("function {name} has an unterminated signature group")
                })?;
            }
            b';' if parentheses == 0 && brackets == 0 && angles == 0 => {
                return Err(format!("function {name} has no body"));
            }
            _ => {}
        }
        cursor += 1;
    }
    Err(format!("function {name} is missing its body"))
}
pub(crate) fn function_scope_at<'a>(
    source: &'a str,
    mask: &str,
    function_start: usize,
    name: &str,
) -> Result<&'a str, String> {
    let body_open = function_body_open(mask, function_start, name)?;
    let body_close = matching_delimiter(mask, body_open, b'{', b'}')
        .ok_or_else(|| format!("function {name} has an unterminated body"))?;
    Ok(&source[function_start..=body_close])
}
pub(crate) fn unique_named_item_scope_at_depth<'a>(
    source: &'a str,
    keyword: &str,
    name: &str,
    expected_depth: u32,
) -> Result<&'a str, String> {
    let mask = source_mask(source);
    let mut offsets = Vec::new();
    let mut depth = 0_u32;
    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut cursor = 0;
    while cursor < mask.len() {
        if depth == expected_depth
            && parentheses == 0
            && brackets == 0
            && token_at(&mask, cursor, keyword)
        {
            let name_start = skip_ascii_whitespace(&mask, cursor + keyword.len());
            let name_end = identifier_end(&mask, name_start);
            let item_name = mask[name_start..name_end]
                .strip_prefix("r#")
                .unwrap_or(&mask[name_start..name_end]);
            let inline_module = keyword != "mod"
                || mask.as_bytes().get(skip_ascii_whitespace(&mask, name_end)) == Some(&b'{');
            if item_name == name && inline_module {
                offsets.push(cursor);
            }
        }
        match mask.as_bytes()[cursor] {
            b'{' => depth += 1,
            b'}' => depth = depth.checked_sub(1).expect("balanced source braces"),
            b'(' => parentheses += 1,
            b')' => {
                parentheses = parentheses
                    .checked_sub(1)
                    .expect("balanced source parentheses")
            }
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1).expect("balanced source brackets"),
            _ => {}
        }
        cursor += 1;
    }
    let [offset] = offsets.as_slice() else {
        return Err(format!(
            "expected one direct {keyword} {name} item at brace depth {expected_depth}, found {}",
            offsets.len()
        ));
    };
    if keyword == "fn" {
        function_scope_at(source, &mask, *offset, name)
    } else {
        Ok(braced_scope_at(source, *offset, name))
    }
}
pub(crate) fn skip_ascii_whitespace_back(source: &str, mut cursor: usize) -> usize {
    while cursor > 0 && source.as_bytes()[cursor - 1].is_ascii_whitespace() {
        cursor -= 1;
    }
    cursor
}
pub(crate) fn identifier_start(source: &str, mut cursor: usize) -> usize {
    while cursor > 0 && is_identifier_byte(source.as_bytes()[cursor - 1]) {
        cursor -= 1;
    }
    cursor
}
pub(crate) fn item_prefix_start(source: &str, item_start: usize) -> usize {
    let mut cursor = item_start;
    loop {
        cursor = skip_ascii_whitespace_back(source, cursor);
        if cursor > 0 && source.as_bytes()[cursor - 1] == b')' {
            let mut group_cursor = cursor;
            let mut depth = 0_u32;
            let mut open = None;
            while group_cursor > 0 {
                group_cursor -= 1;
                match source.as_bytes()[group_cursor] {
                    b')' => depth += 1,
                    b'(' => {
                        depth = depth.checked_sub(1).expect("balanced item prefix");
                        if depth == 0 {
                            open = Some(group_cursor);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(open) = open else {
                break;
            };
            let word_end = skip_ascii_whitespace_back(source, open);
            let word_start = identifier_start(source, word_end);
            if &source[word_start..word_end] == "pub" {
                cursor = word_start;
                continue;
            }
            break;
        }

        let word_start = identifier_start(source, cursor);
        if matches!(
            &source[word_start..cursor],
            "pub" | "unsafe" | "async" | "const" | "extern" | "default"
        ) {
            cursor = word_start;
            continue;
        }
        break;
    }
    cursor
}
pub(crate) fn contiguous_item_attributes(source: &str, item_start: usize) -> &str {
    let mask = source_mask(source);
    let prefix_start = item_prefix_start(&mask, item_start);
    let mut start = prefix_start;
    loop {
        start = skip_ascii_whitespace_back(&mask, start);
        if start == 0 || mask.as_bytes()[start - 1] != b']' {
            break;
        }
        let mut cursor = start - 1;
        let mut brackets = 1_u32;
        while cursor > 0 {
            cursor -= 1;
            match mask.as_bytes()[cursor] {
                b']' => brackets += 1,
                b'[' => {
                    brackets -= 1;
                    if brackets == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        let hash_end = skip_ascii_whitespace_back(&mask, cursor);
        if brackets != 0 || hash_end == 0 || mask.as_bytes()[hash_end - 1] != b'#' {
            break;
        }
        start = hash_end - 1;
    }
    &source[start..prefix_start]
}
pub(crate) fn active_production_function_scope<'a>(
    source: &'a str,
    name: &str,
) -> Result<&'a str, String> {
    let function = unique_named_item_scope_at_depth(source, "fn", name, 0)?;
    let item_start = function.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if !attributes.is_empty() {
        return Err(format!(
            "production function {name} must not have cfg or other item attributes"
        ));
    }
    Ok(function)
}
pub(crate) fn active_production_method_scope<'a>(
    source: &'a str,
    type_name: &str,
    method: &str,
) -> Result<&'a str, String> {
    let function = method_scope_for_type(source, type_name, method);
    let item_start = function.as_ptr() as usize - source.as_ptr() as usize;
    let attributes = compact_code(&source_mask(contiguous_item_attributes(source, item_start)));
    if !attributes.is_empty() {
        return Err(format!(
            "production method {type_name}::{method} must not have cfg or other item attributes"
        ));
    }
    Ok(function)
}
pub(crate) fn statement_at(source: &str, start: usize) -> &str {
    let mask = source_mask(&source[start..]);
    let mut depth = 0_u32;
    for (relative, byte) in mask.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth = depth.checked_sub(1).expect("balanced statement braces"),
            b';' if depth == 0 => return &source[start..=start + relative],
            _ => {}
        }
    }
    panic!("unterminated statement")
}
pub(crate) fn named_function_offsets(source: &str, expected_name: &str) -> Vec<usize> {
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
pub(crate) fn loaded_nvrtc_version() -> (i32, i32) {
    let mut major = 0;
    let mut minor = 0;
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(result, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    (major, minor)
}
pub(crate) fn compact_code(source: &str) -> String {
    source_mask(source)
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}
pub(crate) fn expected_sm80_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for (tile, stages) in [
            ("m128n64", &[2, 3][..]),
            ("m64n64", &[2, 3][..]),
            ("m16n32", &[4][..]),
            ("m16n16", &[4][..]),
        ] {
            for stage in stages {
                symbols.insert(format!("{op}_sm80_mma_tf32_{tile}_bk32_s{stage}"));
            }
        }
    }
    symbols
}
pub(crate) fn expected_sm90a_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for warpgroups in [1, 2] {
            symbols.insert(format!(
                "{op}_sm90a_wgmma_tf32_m64n128_bk32_s3_wg{warpgroups}"
            ));
        }
    }
    symbols
}
pub(crate) fn expected_sm100_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for columns in [64, 128] {
            for stages in [2, 3, 4] {
                for schedule in ["c4", "p8"] {
                    symbols.insert(format!(
                        "{op}_sm100_tcgen_tf32_m128n{columns}_bk32_s{stages}_{schedule}"
                    ));
                }
            }
        }
    }
    symbols
}
pub(crate) fn expected_sm120_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["m128n64", "m64n128"] {
            for stages in [2, 3] {
                symbols.insert(format!("{op}_sm120_tma_mma_tf32_{tile}_bk32_s{stages}"));
            }
        }
        symbols.insert(format!("{op}_sm120_tma_mma_tf32_m64n64_bk32_s2"));
    }
    symbols.insert("tn_sm120_tma_mma_tf32_m64n128_bk32_s4_pair".to_string());
    symbols.insert("tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk".to_string());
    symbols.insert("nn_sm120_tma_mma_tf32_m80n32_bk64_s2".to_string());
    symbols
}
pub(crate) fn expected_hardware_symbols(cc: (u32, u32)) -> BTreeSet<String> {
    let mut symbols = expected_sm80_symbols();
    if matches!(cc, (8, 0 | 6 | 7 | 9) | (9, 0) | (10, 0 | 3 | 7) | (11, 0)) {
        symbols.insert("nn_sm80_mma_tf32_m128n128_bk32_s3".to_string());
    }
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
pub(crate) fn checked_output(mut command: Command, label: &str) -> Output {
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
pub(crate) enum StrictJsonValue {
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
pub(crate) struct StrictJsonParser<'a> {
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
pub(crate) fn parse_strict_json(source: &str) -> Result<StrictJsonValue, String> {
    StrictJsonParser { source, cursor: 0 }.parse()
}
pub(crate) fn strict_json_object(
    value: &StrictJsonValue,
) -> Result<&BTreeMap<String, StrictJsonValue>, String> {
    match value {
        StrictJsonValue::Object(fields) => Ok(fields),
        _ => Err("qualification JSON root must be an object".to_owned()),
    }
}
pub(crate) fn strict_json_string<'a>(
    fields: &'a BTreeMap<String, StrictJsonValue>,
    field: &str,
) -> Result<(&'a str, (usize, usize)), String> {
    match fields.get(field) {
        Some(StrictJsonValue::String { value, raw_content }) => Ok((value, *raw_content)),
        Some(_) => Err(format!("qualification field {field} must be a string")),
        None => Err(format!("qualification field {field} is missing")),
    }
}
pub(crate) fn strict_json_u64(
    fields: &BTreeMap<String, StrictJsonValue>,
    field: &str,
) -> Result<u64, String> {
    match fields.get(field) {
        Some(StrictJsonValue::Number(number)) => number
            .parse()
            .map_err(|_| format!("qualification field {field} must be an unsigned integer")),
        Some(_) => Err(format!("qualification field {field} must be a number")),
        None => Err(format!("qualification field {field} is missing")),
    }
}
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}
pub(crate) fn parse_sha256_hex(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be one hexadecimal SHA-256 digest"));
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{label} contains invalid hexadecimal"))?;
    }
    Ok(digest)
}
pub(crate) fn qualification_semantic_digest(
    rows: &[Vec<&str>],
    field: usize,
    domain: &[u8],
) -> Result<[u8; 32], String> {
    let mut digest = mamba_rs::mamba_ssm::gpu::kernel_identity::FramedSha256::new(domain)
        .required(b"route-count", &(rows.len() as u64).to_le_bytes());
    for row in rows {
        digest = digest
            .required(b"symbol", row[0].as_bytes())
            .required(b"value", &parse_sha256_hex(row[field], row[0])?);
    }
    Ok(digest.finish())
}
pub(crate) fn qualification_driver_jit_resource_summary(
    rows: &[Vec<&str>],
) -> Result<(u32, u64, [u8; 32]), String> {
    let mut max_local_bytes = 0;
    let mut exception_count = 0;
    let mut digest = mamba_rs::mamba_ssm::gpu::kernel_identity::FramedSha256::new(
        b"tf32-driver-jit-local-resources.v1",
    )
    .required(b"route-count", &(rows.len() as u64).to_le_bytes());
    for row in rows {
        let route_identity = parse_sha256_hex(row[5], row[0])?;
        let registers = row[13]
            .parse::<u32>()
            .map_err(|_| format!("{} has an invalid register count", row[0]))?;
        let observed = row[14]
            .parse::<u32>()
            .map_err(|_| format!("{} has invalid Driver JIT local bytes", row[0]))?;
        let cap = row[15]
            .parse::<u32>()
            .map_err(|_| format!("{} has an invalid Driver JIT local cap", row[0]))?;
        max_local_bytes = max_local_bytes.max(observed);
        exception_count += u64::from(observed != 0);
        digest = digest
            .required(b"symbol", row[0].as_bytes())
            .required(b"route-identity", &route_identity)
            .required(b"registers", &registers.to_le_bytes())
            .required(b"observed-local-bytes", &observed.to_le_bytes())
            .required(b"approved-local-cap-bytes", &cap.to_le_bytes());
    }
    Ok((max_local_bytes, exception_count, digest.finish()))
}
pub(crate) fn qualification_route_rows<'a>(
    artifact: &'a [u8],
    fields: &BTreeMap<String, StrictJsonValue>,
) -> Result<Vec<Vec<&'a str>>, String> {
    if strict_json_string(fields, "schema")?.0 != "MambaBiTf32QualificationV5" {
        return Err("qualification report schema is not V5".to_owned());
    }
    if strict_json_string(fields, "driver_jit_local_memory")?.0 != "pass" {
        return Err("qualification Driver JIT local-memory gate did not pass".to_owned());
    }
    let artifact = std::str::from_utf8(artifact)
        .map_err(|_| "qualification artifact must be UTF-8".to_owned())?;
    let mut lines = artifact.lines();
    if lines.next() != Some("MambaBiTf32QualificationArtifactV5") {
        return Err("qualification artifact schema is not V5".to_owned());
    }
    for (prefix, report_field) in [
        ("cc\t", "exact_cc"),
        ("suite\t", "suite"),
        ("repeat\t", "repeat"),
    ] {
        let value = lines
            .next()
            .and_then(|line| line.strip_prefix(prefix))
            .ok_or_else(|| format!("qualification artifact is missing {prefix:?}"))?;
        if report_field == "repeat" {
            if value.parse::<u64>().ok() != Some(strict_json_u64(fields, report_field)?) {
                return Err("qualification artifact repeat differs from its report".to_owned());
            }
        } else if value != strict_json_string(fields, report_field)?.0 {
            return Err(format!(
                "qualification artifact {report_field} differs from its report"
            ));
        }
    }
    let boundary_cases = lines
        .next()
        .and_then(|line| line.strip_prefix("boundary_cases_per_route\t"))
        .ok_or_else(|| "qualification artifact is missing its boundary case count".to_owned())?
        .parse::<u64>()
        .map_err(|_| "qualification boundary case count is not numeric".to_owned())?;
    if boundary_cases != 47 {
        return Err(format!(
            "qualification artifact has {boundary_cases} boundary cases per route instead of 47"
        ));
    }
    let artifact_set = lines
        .next()
        .and_then(|line| line.strip_prefix("artifact_set\t"))
        .ok_or_else(|| "qualification artifact is missing its artifact set".to_owned())?;
    parse_sha256_hex(artifact_set, "artifact_set")?;
    let driver_abi = lines
        .next()
        .and_then(|line| line.strip_prefix("driver_abi\t"))
        .ok_or_else(|| "qualification artifact is missing its Driver ABI digest".to_owned())?;
    parse_sha256_hex(driver_abi, "driver_abi")?;
    if driver_abi != strict_json_string(fields, "driver_abi_digest")?.0 {
        return Err("qualification artifact Driver ABI differs from its report".to_owned());
    }

    let rows = lines
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let route_count = strict_json_u64(fields, "routes_qualified")?;
    if rows.len() as u64 != route_count {
        return Err(format!(
            "qualification artifact has {} route rows, report declares {route_count}",
            rows.len()
        ));
    }
    let total_boundary_cases = strict_json_u64(fields, "boundary_cases")?;
    if total_boundary_cases != route_count * boundary_cases {
        return Err("qualification boundary case counts are inconsistent".to_owned());
    }
    let mut symbols = BTreeSet::new();
    for row in &rows {
        if row.len() != 16 {
            return Err(format!(
                "qualification route row has {} fields instead of 16",
                row.len()
            ));
        }
        if row[0].is_empty() || !symbols.insert(row[0]) {
            return Err(format!(
                "qualification route symbol is empty or duplicated: {:?}",
                row[0]
            ));
        }
        for field in 1..=10 {
            parse_sha256_hex(row[field], row[0])?;
        }
        if row[2] != row[1] {
            return Err(format!(
                "{} graph output digest differs from eager output",
                row[0]
            ));
        }
        if row[5] != row[6] {
            return Err(format!(
                "{} eager route digest differs from graph route",
                row[0]
            ));
        }
        for field in [11, 12] {
            let timing = row[field]
                .parse::<f32>()
                .map_err(|_| format!("{} has an invalid timing", row[0]))?;
            if !timing.is_finite() || timing <= 0.0 {
                return Err(format!("{} has a non-positive timing", row[0]));
            }
        }
        let registers = row[13]
            .parse::<u32>()
            .map_err(|_| format!("{} has an invalid register count", row[0]))?;
        if registers == 0 {
            return Err(format!("{} has a zero register count", row[0]));
        }
        let observed_local_bytes = row[14]
            .parse::<u32>()
            .map_err(|_| format!("{} has invalid Driver JIT local bytes", row[0]))?;
        let approved_local_cap = row[15]
            .parse::<u32>()
            .map_err(|_| format!("{} has an invalid Driver JIT local cap", row[0]))?;
        if observed_local_bytes > approved_local_cap {
            return Err(format!(
                "{} reports {observed_local_bytes} Driver JIT local bytes above its {approved_local_cap}-byte cap",
                row[0]
            ));
        }
    }
    let (max_local_bytes, exception_count, resource_digest) =
        qualification_driver_jit_resource_summary(&rows)?;
    if strict_json_u64(fields, "max_driver_jit_local_memory_bytes")? != u64::from(max_local_bytes)
        || strict_json_u64(fields, "driver_jit_local_memory_exception_count")? != exception_count
    {
        return Err("qualification Driver JIT resource aggregates differ from route rows".into());
    }
    let expected_resource_digest = parse_sha256_hex(
        strict_json_string(fields, "driver_jit_resource_digest")?.0,
        "driver_jit_resource_digest",
    )?;
    if resource_digest != expected_resource_digest {
        return Err("Driver JIT resource digest differs from qualification route rows".into());
    }

    if matches!(strict_json_string(fields, "exact_cc")?.0, "12.0" | "12.1") {
        let sm120_symbols = expected_sm120_symbols();
        let sm120_rows = rows
            .iter()
            .filter(|row| sm120_symbols.contains(row[0]))
            .collect::<Vec<_>>();
        let zero_rows = sm120_rows
            .iter()
            .filter(|row| row[14] == "0" && row[15] == "0")
            .count();
        let portable_nonzero = rows
            .iter()
            .any(|row| !sm120_symbols.contains(row[0]) && (row[14] != "0" || row[15] != "0"));
        if sm120_rows.len() != sm120_symbols.len()
            || zero_rows != sm120_symbols.len()
            || portable_nonzero
        {
            return Err(
                "SM120 qualification must contain every SM120 route with zero local memory".into(),
            );
        }
    }
    Ok(rows)
}
pub(crate) fn verify_qualification_digests(
    report: &str,
    artifact: &[u8],
    driver_abi_proof: &[u8],
) -> Result<(), String> {
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

    let (driver_abi_digest, driver_abi_span) = strict_json_string(fields, "driver_abi_digest")?;
    if &report[driver_abi_span.0..driver_abi_span.1] != driver_abi_digest {
        return Err("driver_abi_digest must be an unescaped canonical string".to_owned());
    }
    let recomputed_driver_abi = sha256_hex(driver_abi_proof);
    if driver_abi_digest != recomputed_driver_abi {
        return Err(format!(
            "Driver ABI digest mismatch: report {driver_abi_digest}, recomputed {recomputed_driver_abi}"
        ));
    }

    if artifact.starts_with(b"MambaBiTf32QualificationArtifactV5\n")
        || fields.contains_key("boundary_output_digest")
    {
        let rows = qualification_route_rows(artifact, fields)?;
        for (artifact_field, report_field, domain) in [
            (
                1,
                "output_digest",
                b"tf32-qualification-all-output.v1".as_slice(),
            ),
            (
                3,
                "zero_reduction_digest",
                b"tf32-qualification-all-zero-reduction.v1".as_slice(),
            ),
            (
                4,
                "boundary_output_digest",
                b"tf32-qualification-all-boundary-output.v1".as_slice(),
            ),
            (
                7,
                "tensor_map_identity_digest",
                b"tf32-qualification-all-encoded-maps.v1".as_slice(),
            ),
            (
                6,
                "ordered_graph_route_digest",
                b"tf32-qualification-all-graph-routes.v1".as_slice(),
            ),
            (
                8,
                "staged_guarded_digest",
                b"tf32-qualification-all-staged-guarded.v1".as_slice(),
            ),
            (
                9,
                "exceptional_values_digest",
                b"tf32-qualification-all-exceptional-values.v1".as_slice(),
            ),
            (
                10,
                "cross_m_invariance_digest",
                b"tf32-qualification-all-cross-m.v1".as_slice(),
            ),
        ] {
            let expected =
                parse_sha256_hex(strict_json_string(fields, report_field)?.0, report_field)?;
            let actual = qualification_semantic_digest(&rows, artifact_field, domain)?;
            if actual != expected {
                return Err(format!(
                    "{report_field} does not match the qualification artifact route rows"
                ));
            }
        }
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
pub(crate) fn qualification_arguments(
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
        "--k-cases".to_owned(),
        "0,1,7,8,9,15,16,17,24,31,32,33,65,97,129,257".to_owned(),
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
        "--artifact-output".to_owned(),
        artifact_output.display().to_string(),
        "--report".to_owned(),
        "json".to_owned(),
    ]
}
pub(crate) fn sanitizer_qualification_arguments(
    cc: (u32, u32),
    expected_routes: usize,
    directory: &Path,
    tool: &str,
) -> Vec<String> {
    let artifact = directory.join(format!("qualification-artifact-{tool}.bin"));
    let driver_abi = directory.join(format!("driver-abi-{tool}.tsv"));
    qualification_arguments(cc, expected_routes, &artifact, &driver_abi)
}
pub(crate) fn sanitizer_command(binary: &Path, arguments: &[String], tool: &str) -> Command {
    let mut command = Command::new("compute-sanitizer");
    command.args([
        "--error-exitcode",
        "99",
        "--report-api-errors",
        "no",
        "--tool",
        tool,
    ]);
    command.arg(binary);
    command.args(arguments);
    command.args(["--suite", "sanitizer"]);
    command
}
pub(crate) fn validate_cuda12_driver_abi_lookup_contract(source: &str) -> Result<(), String> {
    let driver_call = active_production_function_scope(source, "driver_call")?;
    let expected_driver_call = r#"
        fn driver_call(
            result: cudarc::driver::sys::CUresult,
            operation: &str
        ) -> Result<(), String> {
            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!(
                    "{operation}: {:?}",
                    cudarc::driver::result::DriverError(result)
                ))
            }
        }
    "#;
    let driver_call_code = compact_code(&source_mask(driver_call));
    let expected_driver_call_code = compact_code(&source_mask(expected_driver_call));
    if driver_call_code != expected_driver_call_code {
        return Err(format!(
            "Driver result validation must keep its exact live implementation: {driver_call_code:?} != {expected_driver_call_code:?}"
        ));
    }

    let driver_proc_address = active_production_function_scope(source, "driver_proc_address")?;
    let expected_driver_proc_address = r#"
        fn driver_proc_address(symbol: &str, cuda_version: i32) -> Result<*mut c_void, String> {
            let symbol = CString::new(symbol).expect("static CUDA Driver symbol");
            let mut address = std::ptr::null_mut();
            let mut status =
                cudarc::driver::sys::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            let result = unsafe {
                cudarc::driver::sys::cuGetProcAddress_v2(
                    symbol.as_ptr(),
                    &mut address,
                    cuda_version,
                    cudarc::driver::sys::CUdriverProcAddress_flags::CU_GET_PROC_ADDRESS_DEFAULT as u64,
                    &mut status,
                )
            };
            driver_call(result, &format!("resolve {}", symbol.to_string_lossy()))?;
            if status
                != cudarc::driver::sys::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS
                || address.is_null()
            {
                return Err(format!(
                    "resolve {} returned {status:?} at address {address:p}",
                    symbol.to_string_lossy(),
                ));
            }
            Ok(address)
        }
    "#;
    if compact_code(&source_mask(driver_proc_address))
        != compact_code(&source_mask(expected_driver_proc_address))
    {
        return Err("Driver symbol lookup must keep its exact live implementation".into());
    }

    let checked = active_production_method_scope(source, "Tf32DriverAbi", "checked")?;
    let expected_checked = r#"
        fn checked(
            parameter_count: usize,
            parameters: Vec<(usize, usize)>
        ) -> Result<Self, String> {
            if parameter_count == 0 {
                return Err("TF32 Driver ABI has no parameters".into());
            }
            if parameter_count != parameters.len() {
                return Err(format!(
                    "TF32 Driver ABI count is {parameter_count}, but {} layouts were queried",
                    parameters.len()
                ));
            }
            if parameter_count > 64 {
                return Err(format!(
                    "TF32 Driver ABI reports an implausible parameter count {parameter_count}"
                ));
            }

            let mut previous_end = 0;
            let mut checked = Vec::with_capacity(parameter_count);
            for (index, (offset, size)) in parameters.into_iter().enumerate() {
                if size == 0 {
                    return Err(format!("TF32 Driver ABI parameter {index} has zero size"));
                }
                if index == 0 && offset != 0 {
                    return Err(format!(
                        "TF32 Driver ABI first parameter starts at offset {offset}"
                    ));
                }
                if offset < previous_end {
                    return Err(format!(
                        "TF32 Driver ABI parameter {index} overlaps its predecessor"
                    ));
                }
                previous_end = offset.checked_add(size).ok_or_else(|| {
                    format!("TF32 Driver ABI parameter {index} extent overflows usize")
                })?;
                checked.push(Tf32DriverParameterAbi { offset, size });
            }
            Ok(Self {
                parameter_count,
                parameters: checked.into_boxed_slice(),
            })
        }
    "#;
    if compact_code(&source_mask(checked)) != compact_code(&source_mask(expected_checked)) {
        return Err("Tf32DriverAbi::checked must keep its exact live implementation".into());
    }

    let census = active_production_function_scope(source, "census_tf32_driver_abi")?;
    let expected_census = r#"
        fn census_tf32_driver_abi(
            ctx: &CudaContext,
            module_kind: ModuleKind,
            extensions: bool,
            ptx: &str,
        ) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
            let symbols: Vec<&'static str> = super::contract::tf32_route_specs_for(module_kind, extensions)
                .map(|spec| spec.symbol)
                .collect();
            if symbols.is_empty() {
                return Ok(BTreeMap::new());
            }

            type GetParamInfo = unsafe extern "C" fn(
                cudarc::driver::sys::CUfunction,
                usize,
                *mut usize,
                *mut usize,
            ) -> cudarc::driver::sys::CUresult;

            let module = DriverModule::load(ctx, ptx)?;
            let get_parameter_info: GetParamInfo =
                unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
            let mut census = BTreeMap::new();
            for symbol in symbols {
                let name = CString::new(symbol).expect("static TF32 symbol");
                let function = unsafe { cudarc::driver::result::module::get_function(module.raw(), name) }
                    .map_err(|error| format!("load {module_kind:?}/{symbol} for Driver ABI: {error:?}"))?;
                let label = format!("{module_kind:?}/{symbol}");
                let abi = query_tf32_driver_parameter_abi(
                    &label,
                    tf32_driver_parameter_count(module_kind, symbol),
                    |index, offset, size| unsafe { get_parameter_info(function, index, offset, size) },
                )?;
                if census.insert(symbol, abi).is_some() {
                    return Err(format!(
                        "{module_kind:?} Driver ABI census contains duplicate symbol {symbol}"
                    ));
                }
            }
            module.unload()?;
            Ok(census)
        }
    "#;
    if compact_code(&source_mask(census)) != compact_code(&source_mask(expected_census)) {
        return Err("Driver ABI census must keep its exact live-query data flow".into());
    }
    let initializer_marker = "let get_parameter_info: GetParamInfo =";
    let initializer_offsets = marker_offsets_at_brace_depth(census, initializer_marker, 1);
    let [initializer_offset] = initializer_offsets.as_slice() else {
        return Err(format!(
            "expected one direct Driver ABI function-pointer initializer, found {}",
            initializer_offsets.len()
        ));
    };
    let initializer = statement_at(census, *initializer_offset);
    let expected_initializer = r#"
        let get_parameter_info: GetParamInfo = unsafe {
            std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?)
        };
    "#;
    if compact_code(&source_mask(initializer)) != compact_code(&source_mask(expected_initializer)) {
        return Err("Driver ABI lookup must use the exact direct initializer".into());
    }
    let expected_call = "driver_proc_address(\"cuFuncGetParamInfo\", 12_040)";
    let call_offset = initializer
        .find(expected_call)
        .ok_or_else(|| "missing CUDA 12-compatible Driver API lookup call".to_string())?;
    let initializer_mask = source_mask(initializer);
    if !token_at(&initializer_mask, call_offset, "driver_proc_address") {
        return Err("Driver API lookup must be executable code".into());
    }
    let version_offset = call_offset
        + expected_call
            .find("12_040")
            .expect("CUDA Driver API version in expected call");
    if &initializer_mask[version_offset..version_offset + "12_040".len()] != "12_040" {
        return Err("CUDA Driver API version must be executable code".into());
    }
    let census_mask = source_mask(census);
    if token_offsets(&census_mask, "get_parameter_info").len() != 2 {
        return Err(
            "Driver ABI function pointer must have one direct downstream invocation".into(),
        );
    }
    let symbol_loops = marker_offsets_at_brace_depth(census, "for symbol in symbols", 1);
    let [symbol_loop_offset] = symbol_loops.as_slice() else {
        return Err(format!(
            "expected one direct Driver ABI symbol loop, found {}",
            symbol_loops.len()
        ));
    };
    let symbol_loop = braced_scope_at(census, *symbol_loop_offset, "for symbol in symbols");
    let expected_symbol_loop = r#"
            for symbol in symbols {
                let name = CString::new(symbol).expect("static TF32 symbol");
                let function = unsafe { cudarc::driver::result::module::get_function(module.raw(), name) }
                    .map_err(|error| format!("load {module_kind:?}/{symbol} for Driver ABI: {error:?}"))?;
                let label = format!("{module_kind:?}/{symbol}");
                let abi = query_tf32_driver_parameter_abi(
                    &label,
                    tf32_driver_parameter_count(module_kind, symbol),
                    |index, offset, size| unsafe { get_parameter_info(function, index, offset, size) },
                )?;
                if census.insert(symbol, abi).is_some() {
                    return Err(format!(
                        "{module_kind:?} Driver ABI census contains duplicate symbol {symbol}"
                    ));
                }
            }
    "#;
    if compact_code(&source_mask(symbol_loop)) != compact_code(&source_mask(expected_symbol_loop)) {
        return Err("Driver ABI symbol loop must bind every live query directly to census".into());
    }
    let downstream_marker = "let abi = query_tf32_driver_parameter_abi";
    let downstream_offsets = marker_offsets_at_brace_depth(symbol_loop, downstream_marker, 1);
    let [downstream_offset] = downstream_offsets.as_slice() else {
        return Err(format!(
            "expected one direct Driver ABI query in the symbol loop, found {}",
            downstream_offsets.len()
        ));
    };
    let downstream = statement_at(symbol_loop, *downstream_offset);
    let expected_downstream = r#"
        let abi = query_tf32_driver_parameter_abi(
                    &label,
                    tf32_driver_parameter_count(module_kind, symbol),
                    |index, offset, size| unsafe { get_parameter_info(function, index, offset, size) },
                )?;
    "#;
    if compact_code(&source_mask(downstream)) != compact_code(&source_mask(expected_downstream)) {
        return Err("Driver ABI query must directly invoke the resolved function pointer".into());
    }

    let terminal_probe = active_production_function_scope(source, "query_driver_parameter_abi")?;
    let expected_terminal_probe = r#"
        fn query_driver_parameter_abi(
            label: &str,
            parameter_count: usize,
            mut get_parameter_info: impl FnMut(
                usize,
                &mut usize,
                &mut usize
            ) -> cudarc::driver::sys::CUresult,
        ) -> Result<Tf32DriverAbi, String> {
            let mut parameters = Vec::with_capacity(parameter_count);
            for index in 0..parameter_count {
                let mut offset = 0;
                let mut size = 0;
                let result = get_parameter_info(index, &mut offset, &mut size);
                driver_call(result, &format!("cuFuncGetParamInfo {label}[{index}]"))?;
                parameters.push((offset, size));
            }

            let mut extra_offset = 0;
            let mut extra_size = 0;
            let extra = get_parameter_info(
                parameter_count,
                &mut extra_offset,
                &mut extra_size
            );
            if extra == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "{label} exposes more than {parameter_count} Driver ABI parameters"
                ));
            }
            if extra != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE {
                driver_call(
                    extra,
                    &format!("cuFuncGetParamInfo {label}[{parameter_count}] sentinel"),
                )?;
            }
            Tf32DriverAbi::checked(parameters.len(), parameters)
                .map_err(|error| format!("{label}: {error}"))
        }
    "#;
    if compact_code(&source_mask(terminal_probe))
        != compact_code(&source_mask(expected_terminal_probe))
    {
        return Err(format!(
            "Driver ABI terminal probe must keep its exact fail-closed flow: {:?} != {:?}",
            compact_code(&source_mask(terminal_probe)),
            compact_code(&source_mask(expected_terminal_probe))
        ));
    }
    let tf32_wrapper = active_production_function_scope(source, "query_tf32_driver_parameter_abi")?;
    let expected_tf32_wrapper = r#"
        fn query_tf32_driver_parameter_abi(
            label: &str,
            parameter_count: usize,
            get_parameter_info: impl FnMut(usize, &mut usize, &mut usize) -> cudarc::driver::sys::CUresult,
        ) -> Result<Tf32DriverAbi, String> {
            query_driver_parameter_abi(label, parameter_count, get_parameter_info)
        }
    "#;
    if compact_code(&source_mask(tf32_wrapper)) != compact_code(&source_mask(expected_tf32_wrapper))
    {
        return Err(
            "TF32 Driver ABI wrapper must bind the exact production parameter count".into(),
        );
    }
    let splitk_census = active_production_function_scope(source, "census_tf32_splitk_driver_abi")?;
    let splitk_loops = marker_offsets_at_brace_depth(
        splitk_census,
        "for spec in super::contract::tf32_splitk_specs_for(extensions)",
        1,
    );
    let [splitk_loop] = splitk_loops.as_slice() else {
        return Err("split-K Driver ABI census must keep one exact symbol loop".into());
    };
    let splitk_loop = braced_scope_at(splitk_census, *splitk_loop, "split-K Driver ABI loop");
    let splitk_queries =
        marker_offsets_at_brace_depth(splitk_loop, "let abi = query_driver_parameter_abi", 1);
    let [splitk_query] = splitk_queries.as_slice() else {
        return Err("split-K Driver ABI loop must keep one direct generic query".into());
    };
    let splitk_query = statement_at(splitk_loop, *splitk_query);
    let expected_splitk_query = r#"
        let abi = query_driver_parameter_abi(&label, 7, |index, offset, size| unsafe {
            get_parameter_info(function, index, offset, size)
        })?;
    "#;
    if compact_code(&source_mask(splitk_query)) != compact_code(&source_mask(expected_splitk_query))
    {
        return Err("split-K Driver ABI query must bind the seven-parameter contract".into());
    }
    let extra_queries =
        marker_offsets_at_brace_depth(terminal_probe, "let extra = get_parameter_info(", 1);
    let success_checks = marker_offsets_at_brace_depth(
        terminal_probe,
        "if extra == cudarc::driver::sys::CUresult::CUDA_SUCCESS",
        1,
    );
    let terminal_checks = marker_offsets_at_brace_depth(
        terminal_probe,
        "if extra != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE",
        1,
    );
    let checked_tails = marker_offsets_at_brace_depth(
        terminal_probe,
        "Tf32DriverAbi::checked(parameters.len(), parameters)",
        1,
    );
    let ([extra_query], [success_check], [terminal_check], [checked_tail]) = (
        extra_queries.as_slice(),
        success_checks.as_slice(),
        terminal_checks.as_slice(),
        checked_tails.as_slice(),
    ) else {
        return Err("Driver ABI terminal probe must keep one direct fail-closed sequence".into());
    };
    if !(extra_query < success_check
        && success_check < terminal_check
        && terminal_check < checked_tail)
    {
        return Err("Driver ABI terminal probe checks are out of order".into());
    }
    let terminal_mask = source_mask(terminal_probe);
    let function_close = terminal_mask
        .rfind('}')
        .ok_or_else(|| "Driver ABI terminal probe is missing its closing brace".to_string())?;
    let checked_tail_source = &terminal_probe[*checked_tail..function_close];
    let expected_checked_tail = r#"
        Tf32DriverAbi::checked(parameters.len(), parameters)
            .map_err(|error| format!("{label}: {error}"))
    "#;
    if compact_code(&source_mask(checked_tail_source))
        != compact_code(&source_mask(expected_checked_tail))
    {
        return Err("Driver ABI checked result must be the sole returned tail expression".into());
    }
    let extra_query_statement = statement_at(terminal_probe, *extra_query);
    let expected_extra_query = r#"
        let extra = get_parameter_info(
            parameter_count,
            &mut extra_offset,
            &mut extra_size
        );
    "#;
    if compact_code(&source_mask(extra_query_statement))
        != compact_code(&source_mask(expected_extra_query))
    {
        return Err("Driver ABI terminal query must probe the first undeclared parameter".into());
    }
    let success_branch = braced_scope_at(
        terminal_probe,
        *success_check,
        "Driver ABI terminal success rejection",
    );
    let expected_success_branch = r#"
        if extra == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "{label} exposes more than {parameter_count} Driver ABI parameters"
            ));
        }
    "#;
    if compact_code(&source_mask(success_branch))
        != compact_code(&source_mask(expected_success_branch))
    {
        return Err("Driver ABI terminal success must reject an extra parameter".into());
    }
    let terminal_branch = braced_scope_at(
        terminal_probe,
        *terminal_check,
        "Driver ABI terminal status rejection",
    );
    let expected_terminal_branch = r#"
        if extra != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            driver_call(
                extra,
                &format!("cuFuncGetParamInfo {label}[{parameter_count}] sentinel"),
            )?;
        }
    "#;
    let terminal_code = compact_code(&source_mask(terminal_branch));
    let expected_terminal_code = compact_code(&source_mask(expected_terminal_branch));
    if terminal_code != expected_terminal_code {
        return Err(format!(
            "Driver ABI terminal probe must accept only CUDA_ERROR_INVALID_VALUE: {terminal_code:?} != {expected_terminal_code:?}"
        ));
    }
    if token_present(&source_mask(source), "cuFuncGetParamCount") {
        return Err("CUDA 12-compatible Driver ABI may not use cuFuncGetParamCount".into());
    }
    Ok(())
}
pub(crate) fn assert_cuda12_driver_abi_lookup_contract() {
    validate_cuda12_driver_abi_lookup_contract(MODULE_SOURCE)
        .unwrap_or_else(|error| panic!("{error}"));
}
/// Per-op census of the routes the tool qualifies on a device: the portable
/// set plus the device module's TF32 routes. The exact-F32 SM120 routes are
/// not TF32 candidates.
pub(crate) struct QualifiedOpCensus {
    pub(crate) nn: usize,
    pub(crate) tn: usize,
    pub(crate) nt: usize,
}
pub(crate) fn qualified_op_census(cc: (u32, u32), expected_routes: usize) -> QualifiedOpCensus {
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::tf32_qualification_route_specs;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;
    let mut census = QualifiedOpCensus {
        nn: 0,
        tn: 0,
        nt: 0,
    };
    for spec in tf32_qualification_route_specs(cc).expect("supported qualification CC") {
        match spec.op {
            ResolvedGemmOp::Nn => census.nn += 1,
            ResolvedGemmOp::Tn => census.tn += 1,
            ResolvedGemmOp::Nt => census.nt += 1,
        }
    }
    assert_eq!(
        census.nn + census.tn + census.nt,
        expected_routes,
        "qualified route census for CC {cc:?}"
    );
    census
}
