# Release qualification

This page is the maintainers' release runbook. CI compiles, lints and
runs the host tests, but it owns no GPU: a green pipeline proves nothing
about kernel behaviour. A release is qualified on
GPU boxes against the declared targets and the lane table, and the evidence
it produces is data, not scrollback.

## Targets and lanes

Every target is declared in `Cargo.toml` (`autotests` and `autobenches` are
off) and takes exactly one lane in `qual/lanes.toml`. The host-side check
`tests/qual_lane_census.rs` runs in CI and fails when a declared target has
no lane, a gate file hides an ignored arm, a bench is not harness-free under
`benches/`, a qualification tool is not under `tools/qualification/` behind
the `qualification` feature, or anything outside the declared source,
test, bench and tool directories is reachable from the manifest.

| lane | what | how it runs |
|---|---|---|
| host | regressions that need no GPU and no CUDA feature | `cargo test --no-default-features` (CI, any machine) |
| gate | CUDA regressions with no ignored arm | `qual/run.sh gate` (`cargo test --release --features cuda`) |
| contract | CUDA regressions whose ignored arms need a specific board or a long run | `qual/run.sh contract` (each target with `-- --ignored`; the first red stops) |
| record | manual instruments that need model assets or foreign hardware, still under `tests/` | `qual/run.sh record` lists them; run one at a time |
| bench | timing instruments under `benches/`, no verdict | `cargo bench --features cuda --bench <name> [-- <instrument>]` |
| qualification | hardware, toolkit and inventory instruments under `tools/qualification/` | `cargo test --release --features "cuda hf qualification" --test <name> -- --ignored`, one per board and toolkit |

Kernel-candidate stands, scouts and diagnostics are kept in the
maintainers' archive with an index of their origin; they are outside the
crate and its targets.

## Evidence

Bit-gate printers record their cells when `MAMBA_RS_ACCEPTANCE_TSV` names a
file: boundary sets of the invariance matrix, the tensor-core goldens,
training run digests, the decode chain, the serve hash table, the scan
ledger. Two captures compare with:

```
cargo run --release --example acceptance_diff -- base.tsv candidate.tsv
```

which prints every moved or one-sided cell and exits non-zero on any
difference. A release candidate's capture must be byte-identical to the last
accepted one for every cell whose kernels the release does not intentionally
change; an intentional change re-records its cells in the same commit that
makes it.

Every performance table that is published records the board, driver,
toolkit, source hash, GEMM mode, storage precision and the family that
served, comparator (Fast TF32, Fast f32 compute or Pedantic f32), output
dtype, eager or graph path, and
the window count and aggregation. Boards are reported separately.

It also records how the route was admitted on the board it was timed on.
A route the board holds a frozen cohort for is a measured winner there; a
route the board admitted by the first-use bit proof carries the speed
evidence of the board it was measured on and none of its own, and a table
may not present the two as the same claim.

## Order

1. Host: `cargo fmt --all -- --check`, `cargo clippy --workspace
   --all-targets -- -D warnings`, `cargo test --no-default-features`, and
   `cargo check --workspace --all-targets` with `cuda`, `cuda,hf`,
   `cuda,hf,gemm-blas,nccl` and each plus `qualification`, all without a
   rustc warning.
2. `qual/run.sh gate` green on each qualification board.
3. `qual/run.sh contract` on each board, with the evidence capture on;
   the lane runs every target and names each red at its end, and the only
   reds allowed are the targets written for another board, which assert
   that board's compute capability. `acceptance_diff` against the previous
   release's capture clean, or every difference accounted for by an
   intentional, documented change.
4. The qualification tools that back the published tables, one per board
   and toolkit (`qual/run.sh qualification` lists them), with raw logs kept.
5. Package: `cargo package --allow-dirty --no-verify`, extract the `.crate`
   into a fresh directory and inspect it directly: no archive or local
   tooling material; every declared target path and fixture
   present; `cargo test --lib --no-default-features --no-run` and
   `cargo test --no-default-features` pass offline inside the extracted
   tree; one qualification tool and one bench build from it.
6. `cargo publish --locked --dry-run`: the dependency resolution CI tested
   is the one that ships.
7. The previous release has been measured against the release candidate
   on the same board with identical settings, kernel by kernel and as
   whole training and inference steps, and the result is in the changelog
   and on the benchmark page; if it has not run, the changelog says so.
8. Tag only after the release commit's CI is green and a maintainer has
   approved publication.
