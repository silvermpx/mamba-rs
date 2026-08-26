//! Host-side pin scan for the inference kernel family's epilogues.
//!
//! Every epilogue in kernels/gemm_bi_fixed computes alpha and beta
//! through explicit rounding intrinsics (__fmul_rn / __fadd_rn /
//! __fmaf_rn). A bare `alpha * acc` or `val += beta * c` leaves ptxas
//! free to contract multiplies and adds per target under --fmad=true,
//! which would make the epilogue bits architecture- and
//! toolchain-dependent - invisible while one arch is qualified with one
//! toolchain, and a silent re-golden the moment either moves. This scan
//! needs no GPU: it reads the sources and rejects any multiplication
//! spelled outside an intrinsic.

use std::path::Path;

fn strip_comments(line: &str) -> String {
    let line = match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    };
    // Kernel macros carry per-line /* ... */ comments; drop their bodies.
    let mut out = String::new();
    let mut rest = line;
    loop {
        match rest.find("/*") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(i) => {
                out.push_str(&rest[..i]);
                match rest[i..].find("*/") {
                    Some(j) => rest = &rest[i + j + 2..],
                    None => break,
                }
            }
        }
    }
    out
}

#[test]
fn fixed_family_epilogues_spell_alpha_and_beta_through_intrinsics() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/gemm_bi_fixed");
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("kernel dir") {
        let path = entry.expect("dir entry").path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext != "cu" && ext != "cuh" {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("kernel source");
        for (i, raw) in text.lines().enumerate() {
            let line = strip_comments(raw);
            for var in ["alpha", "beta"] {
                let mut from = 0;
                while let Some(pos) = line[from..].find(var) {
                    let at = from + pos;
                    from = at + var.len();
                    // A word boundary on the left keeps identifiers like
                    // shifted_gamma or halpha out of scope.
                    if at > 0 && line.as_bytes()[at - 1].is_ascii_alphanumeric() {
                        continue;
                    }
                    let tail = line[at + var.len()..].trim_start();
                    if tail.starts_with('*') {
                        offenders.push(format!(
                            "{}:{}: {}",
                            path.file_name().unwrap().to_string_lossy(),
                            i + 1,
                            raw.trim()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "bare alpha/beta multiplication in the fixed family - route it \
         through __fmul_rn/__fmaf_rn so ptxas cannot re-contract the \
         epilogue per target:\n{}",
        offenders.join("\n")
    );
}
