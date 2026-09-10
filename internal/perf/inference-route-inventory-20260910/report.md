# Inference route inventory verification

Task904, base `6c282b3c23637551c0a9a121cabf3df505b218c6`.
Status: focused verification complete; source7d890e6c plus fix086716e3;
independent spec/quality review and scoped fix re-review accepted.
Device: RTX6000Ada, CC8.9, CUDA13.2. No new performance measurements.

## RED on unchanged production source

Frozen integration test SHA256:
`a0347517cc292f217f276c4fd94d03959482795e0acc72967f9c49f58038060c`.
Snapshot: `source-snapshots/gemm_inference_route_inventory_red.rs`.
Immutable source: `/root/mamba-inference-inventory-red.R7o5y8`.
Raw packet: `red/`, copied from
`/root/inference-inventory-red-evidence-20260910`.

All three public forward calls succeed and then fail the intended empty-route
assertion `Inference NN launch must be inventoried`. They do not fail at a
vendor boundary or before executing the existing Inference kernel.

| Exact test | Exit | Actual runtime |
|---|---|---|
| `deterministic_inference_public_f32_forward_records_nn_route` |101|5.25s|
| `deterministic_inference_public_bf16_forward_records_nn_route` |101|5.09s|
| `deterministic_inference_public_bf16_to_f32_forward_records_nn_route` |101|5.00s|

Each case ran one test, zero ignored. Release CUDA compilation passed41.53s.
The wrapper runner requires this exact assertion and failing exit, so runner0
means the intended RED was observed, not that these regression tests passed.
Completed `2026-09-10T13:32:05Z`; remote source-after and root local manifest
comparison passed before GREEN source edits. Idle telemetry is retained for
each exact invocation. Root authorized production implementation only after
reading all three failing receipts.

The existing CUDA-only M3 unused-accessor warning is retained in build.log.
This packet does not establish complete physical recording, model graph
guards, fresh5090 behavior or a performance improvement.

## Intermediate production compilation

The first coherent production wiring was copied as a tar snapshot with SHA256
`5c48a097129171220faeb8792a876541e0767307144c57a56b6b75912958acfb`
to fresh `/root/mamba-inference-inventory-compile.gIarDm` while the implementer
continued test work locally. `cargo check --locked --release --features cuda
--lib` passed6.49s on CUDA13.2, runner0 and immutable source-after checks pass.
Raw packet is `compile/`; only the existing CUDA-only M3 accessor warning
appeared. No GPU test was run against this intermediate snapshot. Final GREEN
must use the later coherent source/test freeze, not imply this is final source.

## Focused host verification of the test-bearing snapshot

Snapshot archive SHA256:
`9a48ac6e43f6d4442b11a2d1f85e6fa98b631fcd15b8ce9d58dc6642c83e2bdc`.
Immutable tree: `/root/mamba-inference-inventory-host.1kb4vH`.
Raw packet: `host/`, copied from
`/root/inference-inventory-host-evidence-20260910`.

`cargo test --locked --release --features cuda --lib --no-run` passed in
1m01s. The three focused host filters then passed: seven terminal identity
tests, one append-only tag test, and one cold architecture preparation guard
test. Total: nine passed, zero failed, zero ignored. Runner exit was zero;
the immutable source-after manifest passed. Existing deprecated setter and
M3 accessor warnings remain visible in the build receipt.

This snapshot includes the actual GPU test bodies, but this batch did not
execute them. The implementer continues final source review; GPU acceptance
and independent review are still pending.

## Final source verification

Production source frozen by the implementer, archived as SHA256
`dc08196f19a21efc68fa8485072d848b35ed72f5981c9248fdaa191f24c3e740`,
then tested in `/root/mamba-inference-inventory-green.Z6uxvD`.
CUDA+HF all-target check passed9.83s; CUDA-only lib check passed.
Four integration tests passed22.71s. The fifth failed because its expectation
used the canonical F32 API for an empty output, which that existing API rejects
before dispatch. No production change was needed or made after this freeze.

The corrected test preserves that public rejection and separately checks the
direct `inference_forward` no-op across all five supported storage triples.
Only this test/import changed. Its SHA256 is
`00603815f2869f9e057681b3e6c55abba24a6e6fd3ebf0b4e0ad93dfc479f53c`.
It passed4.56s in `/root/mamba-inference-inventory-empty.2d5Ic9`.
Raw failed and corrected receipts remain in `green/` and `empty/`.

The remaining unchanged-production library packet is `lib/`:

| Check | Actual result |
|---|---|
| Exact terminal identity/mutation host tests |8 passed|
| Append-only tag and cold-architecture guards |2 passed|
| Context host tests |19 passed; one GPU test ignored here and executed below|
| Actual observed terminals versus unrecorded output bits |1 passed,4.77s|
| Actual typed matvec storage/TC/K0 matrix |1 passed,4.68s|
| Actual recorder borrow-conflict guard |1 passed,4.58s|
| Library test compilation, CUDA+HF |passed53.53s|
| Rustdoc with broken links denied |passed2.40s|

Accepted total across these packets:29 host tests and8 actual GPU tests.
This is not a claim of a fresh single5/5 integration run: four original
integration cases passed in the first batch; the corrected fifth passed alone.
Each explicit GPU unit ran1 test with0 failures and0 ignored. Library and
corrected-test runners exited0; source-after checks passed. Root compared all
564 final source/test/kernel files and both Cargo inputs to the local worktree.
Scoped rustfmt and diff checks passed. CUDA/header bytes, admission/dispatcher
tables and compiler identities were not changed by this task.

Existing deprecated-test and private Rustdoc-link warnings remain visible.
SM90/100/120 terminal records have host coverage here, not fresh GPU acceptance.
No performance measurement or complete model-graph/no-vendor claim is made.
Source commit: `7d890e6c6f3bb32eb701f503caa2c7364ddbee69`.

## Independent review fix1: aligned empty architecture request

Review found that an aligned homogeneous-half M0 request on a loaded cold
SM90/SM100 lane could reach architecture probing before the terminal no-op.
The original Ada no-op fixture's K37 did not expose that decision. Same worker
extracted the existing decision unchanged and added a regression through that
production seam. Root observed the intended RED: eager-preparation error where
the empty request required success without a probe. Raw packet: `fix1-red/`;
the tests-first source is frozen in `source-snapshots/inference-empty-arch-red.rs`.

The fix is an early M0/N0 return before the architecture enabled/probe callback.
It preserves nonempty decision logic. Root's focused GREEN packet passed the
new empty test, its nonempty companion and the existing cold guard (3/3).
The actual Ada direct no-op test, extended with aligned K64 alongside K37,
passed4.69s. Both wrapper exits and source-after manifests passed; all564 final
files plus Cargo inputs match the local worktree. See `fix1-green/`, `fix1-gpu/`.

Source commit: `086716e38e0b554009ecf034ffc50e9911790c19`.
Final changed source SHA256:
`6dd5ebd747f05be92c0fa8515fad7357d619aa2da4ddc1eb703ecec2e2ad0287`.
Final integration SHA256:
`72023b1bc5ad17928f4fc1b1e6161df40c67cc91e2cb78751eeb50ae9e58e1d9`.
Independent scoped re-review accepted I1 with no new findings. The original passing packet remains
valid for unchanged terminal behavior; no full matrix was rerun for this guard.
