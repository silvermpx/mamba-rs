# Context-routed F32 GEMMs: Task 902 verification

Base: `de8b463ec1ad37448a448bc4e39a6415b789808a`.
Source commit: `5ba6c5e2bacdc6274c732c1741d92a7f42f2e540`.
Independent spec/quality review: approved, no Critical/Important/Minor findings.
Review: `review.md`, by `/root/context_routing_review`. Its execution-only
Cannot-verify items are adjudicated by the root-owned receipts below; root
observed these commands and checked the source-manifest/byte comparisons.
Board: RTX 6000 Ada, CC8.9, CUDA13.2, driver595.45.04.
GPU UUID: `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.
Root owns all execution below; no performance measurements were made.

## Result

The raw NN and tied NT regressions now execute custom deterministic GEMMs.
M3 F32/typed projections and tied/untied heads use the context dispatcher.
The task does not claim complete Inference/M3 physical graph inventory or
deterministic half-input tied heads: those are subsequent tasks.

| Check | Result | Receipt |
|---|---|---|
| Tests-first actual Ada NN/NT calls | Both fail at the forbidden vendor boundary, as expected | `red/` |
| Actual Ada raw-wrapper tests | 7 passed, 0 failed, 0 ignored; 43.13s | `green-fix1/routing-tests.log` |
| Exact-final CUDA+HF all-target compile | PASS, 9.79s | `host-final/cuda-hf-check.log` |
| Exact-final prepared-control tests | 54 passed, 0 failed, 6 explicitly GPU-ignored; 0.01s | `host-final/prepared-control-tests.log` |
| Exact-final CUDA+HF Rustdoc, broken links denied | PASS, 2.34s, one pre-existing private-link warning | `host-final/rustdoc.log` |
| Frozen remote source integrity | PASS | `host-final/source-after.log` |
| Final local source equals verified remote manifest | PASS | `host-final/local-source-verification.log` |
| Scoped rustfmt and Git diff whitespace | PASS | Root commands below |
| All CUDA/header bytes relative to base | Unchanged | `git diff --exit-code de8b463e -- kernels` |

Final host packet completed at `2026-09-10T12:14:38Z`, runner exit0.
The six ignored existing GPU tests were not silently counted as passes.
The seven actual GPU regressions cover NN/NT routing, both deterministic
families, asymmetric CPU references, bit repeats, interior inputs, guarded
outputs, bias/zero-reduction handling and invalid spans.

## Exact snapshots and scope of reuse

- RED test SHA: `97e3a6e9c0ca045e43971f4415e30cad2dea10a989881325134ba0f6b2934c68`.
- Runtime snapshot: `/root/mamba-context-routing-green-fix1.2SaKOK`.
- Final host snapshot: `/root/mamba-context-routing-host-final.LT9yig`.
- Final tests: `c1318c9a2a5a165d06cd8d8c88b3c02cb20f54102422133162a42d30a4cf660d`.
- Final launch: `024c08b8432acdaaa1fbcd0093474aa40d5e0fad6f78643c92e57f7d80ca311a`.
- Final module exports: `0079d773acb0f240cb550446cebe840d5914679d83f8748300192bdf8e9c24c7`.
- Final blas: `03238d9444bbe0c7d0beca28fc2a37b40426a6ce2a90de194f31436c669628b8`.
- Final M3: `f7fe69c8ebada14a8cbc7304b8706c5e640d9c750b8c1609d48cfed7ef0e70f9`.
- Final LM3: `f5edc250c8e48c17e14e14f2de935c03b854aa02d47fd727e1fc649f90935f4c`.

The runtime and final snapshots differ only in `launch.rs`: the unused thin
cached NN wrapper is now test-only, and one source sentry was repaired.
Root compared the actual runtime source, preserved in
`source-snapshots/launch_green_fix1.rs`, against the final source. No runtime
launch body, CUDA byte, argument order, or dispatcher choice changed after
the seven successful GPU tests. They were not rerun for the test-only edits.

## Failures and corrections

`green-initial/` records E0364 from re-exporting a GPU-scoped validator at
crate scope. The export was narrowed. The resulting unused cached-NN import
was removed, and the helper used only by colocated tests became test-only.
No warning suppression was added.

`green-fix1/` records the successful GPU batch followed by a source-sentry
failure: 53 unit passes, one failure. The sentry expected four physical-route
producer calls, but the base already contained six. It also used the first
`#[cfg(test)]` token as a boundary, which is unsuitable when a test helper
appears earlier. The repair names the actual test module and six producers;
semantic argument/resource assertions remain. The final unit batch passes.

One manual post-failure checksum command initially used the wrong remote cwd.
Its failed output is retained as `source-after-wrong-cwd.log`; verification
from the actual immutable source directory passed all561 recorded files.
This was not source drift or a GPU failure.

## Commands and remaining boundaries

All CUDA commands run on Ada with the environment recorded by each runner:

```text
cargo check --locked --release --features cuda,hf --all-targets
cargo test --locked --release --features cuda --test gemm_context_routing --no-run
<routing-binary> --include-ignored --nocapture --test-threads=1
cargo test --locked --release --features cuda --lib --no-run
<lib-binary> ::prepared_f32_launch_tests:: --nocapture --test-threads=1
RUSTDOCFLAGS='-D rustdoc::broken_intra_doc_links' cargo doc --locked --release --features cuda,hf --no-deps --lib
```

Local checks: `rustfmt --edition 2024 --check` on the six changed Rust files;
`git diff --check`; `git diff --exit-code de8b463e -- kernels`; and
`shasum -a 256 -c host-final/source.sha256` from the worktree root.

Existing deprecated benchmark/test setters remain for the API/test migration.
The existing Rustdoc private-item warning is at `contract.rs:2834`.
Legacy public safe raw-pointer vendor compatibility APIs remain an explicit
API-audit item. No new RTX5090 validation or speed claim is made here.
