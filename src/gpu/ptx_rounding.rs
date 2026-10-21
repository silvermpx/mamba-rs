//! Every floating-point add, subtract and multiply rounds as written.
//!
//! NVRTC compiles the kernels to PTX and the driver assembles that PTX into
//! machine code. PTX leaves an `add`, `sub` or `mul` on a floating-point type
//! that carries no rounding mode free for the assembler to fuse with a
//! neighbour into one FMA, which rounds once instead of twice. Whether it
//! fuses is the assembler's choice, and the assembler is the one the
//! driver build carries, so the same kernel could round its last bit
//! differently from one driver to the next.
//!
//! Before a module is handed to the driver, each such instruction gets the
//! round-to-nearest-even mode it computes anyway (`add.f32` becomes
//! `add.rn.f32`). PTX gives an instruction with an explicit rounding mode
//! exactly that rounding and never fuses it. The FMAs NVRTC already wrote,
//! and the ones the kernel source spells out, stay as they are; nothing
//! else in the module changes. The artifact identities still name the PTX
//! NVRTC produced: this rewrite is a pure function of it.

/// The floating-point types whose `add`, `sub` and `mul` the assembler may
/// fuse when no rounding mode is given.
const FLOAT_TYPES: [&str; 7] = ["f32", "f64", "f16", "f16x2", "bf16", "bf16x2", "f32x2"];

/// The modifiers PTX allows between the opcode and the type of these
/// instructions besides a rounding mode.
const PLAIN_MODIFIERS: [&str; 2] = ["ftz", "sat"];

/// The opcode of one PTX instruction line with the rounding mode inserted,
/// or `None` when the line is not a floating-point `add`, `sub` or `mul`
/// without one.
fn pinned_instruction(line: &str) -> Option<String> {
    let body_start = line.len() - line.trim_start().len();
    let mut rest = &line[body_start..];
    // A guard predicate (`@%p1` or `@!%p1`) may precede the opcode.
    let mut prefix_end = body_start;
    if rest.starts_with('@') {
        let guard_len = rest.find(char::is_whitespace)?;
        let after_guard = &rest[guard_len..];
        let spaces = after_guard.len() - after_guard.trim_start().len();
        prefix_end += guard_len + spaces;
        rest = &rest[guard_len + spaces..];
    }
    let token_len = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let token = &rest[..token_len];
    let (head, ty) = token.rsplit_once('.')?;
    let (opcode, middle) = head.split_once('.').unwrap_or((head, ""));
    if !matches!(opcode, "add" | "sub" | "mul")
        || !FLOAT_TYPES.contains(&ty)
        || !(middle.is_empty() || middle.split('.').all(|m| PLAIN_MODIFIERS.contains(&m)))
    {
        return None;
    }
    let mut pinned = String::with_capacity(line.len() + 3);
    pinned.push_str(&line[..prefix_end]);
    pinned.push_str(opcode);
    pinned.push_str(".rn");
    pinned.push_str(&token[opcode.len()..]);
    pinned.push_str(&rest[token_len..]);
    Some(pinned)
}

/// The PTX with a rounding mode on every floating-point `add`, `sub` and
/// `mul`. The string comes back as it was when no instruction lacks one.
pub(crate) fn pin_rounding(ptx: String) -> String {
    if !ptx.lines().any(|line| pinned_instruction(line).is_some()) {
        return ptx;
    }
    let mut out = String::with_capacity(ptx.len() + ptx.len() / 64);
    for line in ptx.split_inclusive('\n') {
        let (body, newline) = match line.strip_suffix('\n') {
            Some(body) => (body, "\n"),
            None => (line, ""),
        };
        match pinned_instruction(body) {
            Some(pinned) => out.push_str(&pinned),
            None => out.push_str(body),
        }
        out.push_str(newline);
    }
    out
}

/// How many floating-point `add`, `sub` and `mul` instructions in the PTX
/// carry no rounding mode.
#[cfg(test)]
pub(crate) fn unpinned_instructions(ptx: &str) -> usize {
    ptx.lines()
        .filter(|line| pinned_instruction(line).is_some())
        .count()
}

#[cfg(test)]
mod tests {
    use super::{pin_rounding, unpinned_instructions};

    #[test]
    fn bare_float_arithmetic_takes_round_to_nearest_and_nothing_else_moves() {
        let ptx = "\
.version 9.2
.target sm_120
\tmul.f32 \t%f3, %f1, %f2;
\tadd.ftz.f32 \t%f4, %f3, %f1;
\t@%p1 sub.f64 \t%fd2, %fd1, %fd0;
\t@!%p2 mul.sat.f32 %f5, %f4, %f4;
\tadd.f16x2 %r1, %r2, %r3;
\tmul.bf16 %rs1, %rs2, %rs3;
\tfma.rn.f32 \t%f6, %f1, %f2, %f3;
\tadd.rn.f32 \t%f7, %f1, %f2;
\tmul.lo.s32 \t%r4, %r5, %r6;
\tadd.s64 \t%rd1, %rd2, %rd3;
\tmul.wide.u32 \t%rd4, %r1, %r2;
\tadd.cc.u32 %r7, %r8, %r9;
\tsub.f32x2 %rd5, %rd6, %rd7;
// mul.f32 in a comment
";
        assert_eq!(unpinned_instructions(ptx), 7);
        let pinned = pin_rounding(ptx.to_string());
        assert_eq!(unpinned_instructions(&pinned), 0);
        let expected = "\
.version 9.2
.target sm_120
\tmul.rn.f32 \t%f3, %f1, %f2;
\tadd.rn.ftz.f32 \t%f4, %f3, %f1;
\t@%p1 sub.rn.f64 \t%fd2, %fd1, %fd0;
\t@!%p2 mul.rn.sat.f32 %f5, %f4, %f4;
\tadd.rn.f16x2 %r1, %r2, %r3;
\tmul.rn.bf16 %rs1, %rs2, %rs3;
\tfma.rn.f32 \t%f6, %f1, %f2, %f3;
\tadd.rn.f32 \t%f7, %f1, %f2;
\tmul.lo.s32 \t%r4, %r5, %r6;
\tadd.s64 \t%rd1, %rd2, %rd3;
\tmul.wide.u32 \t%rd4, %r1, %r2;
\tadd.cc.u32 %r7, %r8, %r9;
\tsub.rn.f32x2 %rd5, %rd6, %rd7;
// mul.f32 in a comment
";
        assert_eq!(pinned, expected);
    }

    #[test]
    fn ptx_without_bare_arithmetic_comes_back_unchanged() {
        let ptx =
            ".target sm_89\n\tfma.rn.f32 %f1, %f2, %f3, %f4;\n\tadd.u32 %r1, %r2, %r3;".to_string();
        let pointer = ptx.as_ptr();
        let pinned = pin_rounding(ptx);
        assert_eq!(pinned.as_ptr(), pointer);
    }
}
