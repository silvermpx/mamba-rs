# Task 2 report: SM89 half Triad AUTO activation

## Outcome

`TriadSm89Half` now owns exactly the 11 frozen Ada half-Triad AUTO cells in
`task-2-brief.md`. The selector is limited to CC 8.9, 142 SMs, exact NVRTC
12.8/13.0/13.2 PTX identities, current compiler/composer/numeric/schedule
revisions, canonical contiguous shapes, homogeneous F16/BF16 operands, the
production scalar contract, and the operation-specific bias contract.

NN and NT launch their measured six symbols from the isolated module. A missing
or resource-excluded symbol returns only its own request to the existing SM80
fallback. SM90/SM100/SM120 precedence remains before this selector. There are no
TN cells and no public policy or mode was added.

Physical launch identity uses `ModuleKind::TriadSm89Half`, backend
`Sm89Mma16HalfS3V1`, `MmaSyncF32V1`, and private route revision 1. Graph
rehydration resolves the exact special symbol from the same isolated module.
The global tuning revision 45 and frozen TF32 identities are unchanged.

## TDD evidence

- Registry RED: `auto_registry_is_exactly_the_eleven_proven_cells` observed an
  empty left-hand registry before the literal 11-cell table was added.
- Resource RED: the Bxor resource test observed register cap 167 before the
  measured CUDA 12.8 cap 168 was installed.
- Selector RED: all-toolkit positive coverage declined the first CUDA 12.8 NN
  BF16 cell while the selector was a stub.
- Gate RED: wrong compute capability was admitted before device/toolkit/compiler/
  artifact/request fail-closed gates were implemented.
- Identity RED: the half identity resolver rejected the first SM89-half base
  before the six exact symbol mappings and isolated owner were added.

## Verification

- `cargo test --features cuda,cudarc/fallback-latest --lib sm89_half -- --nocapture`
  — 35 passed.
- `cargo test --features cuda,cudarc/fallback-latest --lib half_physical_trace_tests -- --nocapture`
  — 4 passed.
- `cargo test --features cuda,cudarc/fallback-latest --lib physical_launch_tests -- --nocapture`
  — 19 passed.
- `cargo test --no-default-features --test sm89_half_source_freeze -- --nocapture`
  — 22 passed.
- `cargo test --features cuda,cudarc/fallback-latest --lib --no-run` — passed.
- `cargo test --features cuda,cudarc/fallback-latest --test gemm_bi_sm89_half_selector_qualification --no-run`
  — passed.
- Scoped `rustfmt --check` on the touched production files and qualification
  test — passed. Repo-wide `cargo fmt --check` is not used because root owns
  unrelated dirty discovery/SM120 WIP.
- `git diff --check` — passed.

## GPU handoff

No GPU test or timing was run in this task. On an exclusive RTX 6000 Ada, run
the ignored actual-AUTO qualification independently under each selected CUDA
toolkit:

```text
cargo test --features cuda,cudarc/fallback-latest \
  --test gemm_bi_sm89_half_selector_qualification \
  sm89_half_actual_auto_qualification -- --ignored --nocapture
```

It emits live compiler/artifact identity and per-symbol resource JSON, then for
all 11 cells requires one exact `TriadSm89Half` node with the expected symbol,
tile, grid, block and dynamic shared memory; eager/graph manifest equality; a
captured graph replay; and intact red zones.

The CUDA 12.8/13.0 cohorts carry correctness/source/PTX/resource qualification,
not Fast timing claims. CUDA 13.2 retains the recorded strict Fast evidence.
