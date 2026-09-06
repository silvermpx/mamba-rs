# Ada RNA full-corpus static review

## Scope and identity

- Base/HEAD reviewed: `f3248a16a2b64d4790f6fb7a99a04f0d3bdd39ad`.
- Candidate source: `tests/gemm_bi_fixed_correctness.rs` only.
- Candidate SHA-256: `7f0566985b7c80ed858258104f166fc46bc7328cd8548eab778fae94e4eb977f`.
- Governing documents read in full: `ada-rna-cross-toolkit-qualification-plan.md` and `internal/experiments/fixed-rna-production-integration.md`.
- Review mode: static and read-only apart from this ignored note. No build, test execution, SSH, GPU, process-control, source, index, or HEAD operation was performed.

## Findings

No static code-review issues found in the candidate diff.

## Plan alignment

### Real corpus regression

The non-GPU regression asserts the exact ordered corpus returned by the same helper consumed by the GPU qualification path. It would detect any future loss, reordering, or dimension drift in the tail/A-E families.

### Exact shapes and A-E boundaries

The helper contains exactly the specified six ordered entries: tail `(6018,36,132)`, hot A `(4622,384,1928)`, hot B `(4622,768,2304)`, hot C `(4622,1928,384)`, hot D `(2049,768,2304)`, and hot E `(2049,2304,768)`. The forced run no longer depends on `actual_auto` to receive B-E.

### Tail prefixes and hot-family boundaries

The full 21-entry tail prefix/row list is preserved. Every non-tail family uses the specified seven entries covering M=1/16/17, the boundary-minus-two/minus-one/boundary triplet, and the whole-row `M-1,row=1` view.

### C4/C16 and row views

Every tail and hot-family `(M,row)` entry expands to output offsets 1 and 4, exercising C4 and C16. The row-offset views remain whole-row A views and the expected slices remain within their full-shape references.

### Exceptional and bias modes

Each shape is exercised in finite and exceptional modes and with bias absent and present. Exceptional A, B, and bias payloads remain in the shared path.

### Five incumbent rungs

All five existing Fixed TF32 rungs are compared bit-for-bit against RNA-wide for the full output and for every guarded prefix/view group.

### Guard and input equality

The guarded output comparison still covers the complete allocation, including prefix and suffix sentinels, for incumbent, RNA eager, RNA graph, and actual-AUTO runs where enabled. Immutable A, B, and bias checks remain after each shape/input corpus; guarded misaligned-input equality remains in the actual-AUTO-only section.

### Eager and graph gates

Two independently re-poisoned RNA eager repeats and two independently re-poisoned RNA graph replays remain for every view group. The graph symbol, ABI bundle, geometry, and dynamic-shared-memory assertions are unchanged. The separate unsafe-input and K0 test is unchanged by this diff.

### Actual AUTO isolation

The actual-AUTO selection/eager/graph checks and misaligned-input fallback section remain conditional on `actual_auto`; their assertions are unchanged. The distinct actual-AUTO test remains available for the CUDA 13.2 qualification run only, while the `fixed_tf32_rna_wide_` force filter still selects exactly the force corpus and unsafe-input/K0 tests.

### Success marker and countability

The marker is emitted once per completed view group only after all five incumbent comparisons, both re-poisoned RNA eager checks, both re-poisoned RNA graph checks, and—when enabled—the relevant actual-AUTO checks succeed. Its fields make the group independently countable and unique. Static cardinality for one forced run is `(21 + 5*7) * 2 C offsets * 2 input modes * 2 bias modes = 448` markers.

### Source scope and claims

The tracked diff against the base touches only `tests/gemm_bi_fixed_correctness.rs`; no production source is changed. `git diff --check` is clean. The candidate adds no timing or performance claim.

## Runtime qualification

### Evidence integrity and TDD

The final raw bundle under `internal/perf/ada-rna-full-toolkit-qualification-20260906/raw/` was inspected directly rather than relying on the qualification summary or normalized `.groups` files.

- The 21-entry `SHA256SUMS` file hashes to `348935b9e11389dcb4c643c2333d69258b45e05d91f66d71076383d508055278`; an independent `shasum -a 256 -c` check passed for every entry.
- The 167-input source manifest hashes to `44f1a4632512169d56fea483df599507e7cd718cb4853640f2eb3ff733002105`. An independent check against this worktree passed for all 167 entries, including candidate source hash `7f0566985b7c80ed858258104f166fc46bc7328cd8548eab778fae94e4eb977f`.
- The CUDA 13.2 RED log records the intended failure: for `actual_auto=false`, the helper returned only tail and hot A while the regression expected tail plus hot A-E. It reached the behavioral assertion after a successful build and exited 101 with `0 passed; 1 failed`; it is not an environment, dependency, compilation, or synchronization failure.
- The CUDA 13.2 GREEN log records the same regression passing with `1 passed; 0 failed` and exit 0.

### Six qualification runs

No review issue was found in the six raw qualification logs:

- CUDA 12.8 cold and warm: each `2 passed; 0 failed`, exit 0.
- CUDA 13.0 cold and warm: each `2 passed; 0 failed`, exit 0.
- CUDA 13.2 force: `2 passed; 0 failed`, exit 0.
- CUDA 13.2 actual AUTO: the exact wrapper reports `1 passed; 0 failed`, exit 0.

Every raw log has exactly 448 marker substrings and 448 unique marker strings. Each has tail=168 and hot A/B/C/D/E=56 apiece; all four exceptional/bias combinations have 112 entries; C offsets 1 and 4 have 224 entries apiece. The six `(shape_m,k,n)` tuples and the unique `(case,view_m,row_offset)` sets exactly match the static corpus: the 21 tail entries and seven entries for each hot family. No malformed marker record was found.

The five force marker streams are byte-identical (SHA-256 `54f82ffcd92cc1e0424b85bf63024074f77f7a288e6c130b0d0000c9d943378c`) and contain only `actual_auto=false`. The CUDA 13.2 actual-AUTO stream contains only `actual_auto=true`; replacing that one field produces the same force-stream hash, proving the remainder of its 448-record corpus is exact. Every extracted raw stream separately matches its corresponding `.groups` file.

All five force logs explicitly report `fixed_tf32_rna_wide_rejects_unsafe_inputs_and_handles_k0` as `ok`, so the qualification includes the retained unsafe-input and K0 gate rather than only marker-producing views. The CUDA 13.2 actual-AUTO pass executes the unchanged conditional selection, eager/graph bit, admitted-symbol ABI/geometry/resource, and misaligned-fallback assertions established by the static review. No actual-AUTO wrapper was invoked by the 12.8 or 13.0 commands. The plan requires one CUDA 13.2 force run plus this separate actual-AUTO run; it does not require an additional 13.2 force-warm run.

### Builds, runtime, and cache identity

The three build logs report successful dedicated correctness executables with exit 0 and distinct recorded hashes. Each run records the matching feature and explicit toolkit, target, library, and private cache paths. Runtime evidence identifies RTX 6000 Ada, CC 8.9, driver 595.45.04, Rust/Cargo 1.98.0, and NVRTC resolving inside CUDA 12.8, 13.0, and 13.2 respectively with recorded library hashes.

The cache comparison log is internally complete and consistent: each toolkit has exactly three new and three frozen cache blobs; all nine cache-key filenames match their frozen comparator; every new/frozen blob-hash pair is equal; and every recorded byte comparison is `true`. This confirms the report's per-toolkit cache-key/blob retention claim from the archived comparison evidence without inventing a fresh remote observation.

### Format and scope

Independent local `rustfmt --edition 2024 --check` and `git diff --check` both exit 0. `git diff --name-only f3248a16` lists only `tests/gemm_bi_fixed_correctness.rs`. The evidence and report contain no new performance result or claim; reported durations are test runtimes.

## Final verdict

No static or qualification-evidence issues found. The test-only full-corpus task is ready to commit with the selected evidence and report. This verdict qualifies the forced corpus across CUDA 12.8/13.0/13.2 and confirms the existing CUDA 13.2 actual-AUTO assertions; it does not authorize, implement, or imply wider AUTO admission, cross-toolkit AUTO mapping, or a production/epoch/revision change.
