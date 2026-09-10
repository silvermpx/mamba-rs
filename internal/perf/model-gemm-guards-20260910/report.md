# Model GEMM capture and replay guards

Source base: `b7397d4fd53e52c0d070a76b9e91ee3ed51fabc4`.
Original verified source: `96640d627d45cfe68f3c413a2e9bd18eacde8d80`.
Final accepted source: `a651b13395a6403de2852ab4edbe3ff2e606dc68`.
Task905 is complete. Independent review found one stale-permit edge; the
tests-first correction passed focused verification and scoped re-review.
The RED/failure packets below remain failures, not acceptance or performance
results. No new SM120 execution is claimed.

## Acceptance summary

- 15 distinct actual GPU groups passed across the original and correction
  packets:48 complete M1/M3 backbone cases,24 tied/untied head cases, missing
  and stale-permit controls, vendor count/deny and explicit vendor replay,
  both public route-drift cases, direct native-half inventory/coexistence,
  and physical native-half/exact-fallback graphs.
- 14 distinct exact host tests passed: guard restoration, launch scanner,
  exact struct/owner inventory, half identity/environment invariants, physical
  enqueue topology and negative mutations, provenance scopes and owner boundaries.
- CUDA+HF all-target and CUDA-only library compilation passed. Rustdoc passed
  with one pre-existing private-link warning in unchanged `contract.rs`; old
  deprecated-setter warnings remain visible for release cleanup. Scoped fmt
  and `git diff --check` passed. No lint allowance was added.
- Kernel CUDA/header bytes, selectors, qualification admissions and compiler
  identities were not changed. This is model graph integration, not a new
  speed result or a claim of full prefill/training/API release completion.

## Tests-first Ada packet

Frozen source directory: `/root/mamba-model-gemm-guards-red.fZfxf9`.
Only the two tests-bearing files differ from the reviewed904 source:

- `src/mamba3_siso/gpu/inference.rs`:
  `93e7a1fd9b1a6e1b08ab53aba5db6342a594370f6289fc367a5cb62aa055105c`.
- `src/mamba_ssm/gpu/graph_capture.rs`:
  `6b9f6083a61670db780020ab6561b7ad838cf5cb3cd66a35af45932314984712`.

Both files are preserved under `source-snapshots/red/`. Raw logs, exact
test list, source/build manifests, binary hash, runner and per-case GPU idle
checks are under `red/`. CUDA13.2, RTX6000Ada, private warmed JIT/kernel caches.

The CUDA+HF release library test build passed in54.61s. The exact explicitly
ignored tests each ran one real case and failed at the intended assertion:

| Test suffix | Observed failure | Time |
| --- | --- | --- |
| `m3_capture_requires_successful_public_eager_manifest` | Warmed private M3 body, but no successful public eager permit; actual capture incorrectly returned `Ok(())`. |4.71s|
| `deterministic_inference_half_work_rejects_empty_graph_plan` | Real BF16 GEMM produced the expected `[32.0;16]`; the old guard incorrectly accepted a missing plan for this nonzero Deterministic/Inference workload. |4.48s|

Both test exits were101, with one failure and zero ignored. Runner exit0
means the expected RED protocol completed, not that these tests passed.
Source and Cargo input manifests remained unchanged. The sole source writer
was authorized to implement GREEN after root inspected both failures.

## Compile-ready and final candidate

The coherent eight-file implementation compiled on Ada with
`cargo check --locked --release --features cuda,hf --all-targets` in9.16s.
The immutable directory was `/root/mamba-model-gemm-guards-green.Wu94ox`;
receipts are under `compile/`. Runner exit0 and source-after checks passed.
Pre-existing deprecation/unused warnings are not suppressed.

Root then authorized the ninth file, `tests/gemm_bi_tf32_contract.rs`, to
migrate its obsolete model-owner assertions and exact guarded-launch scanner.
The original eight hashes did not change. The five external prefill/training
owners retain their old helper; this task's runtime acceptance is decode/heads,
not a claim to have migrated those other owners.

Final candidate directory: `/root/mamba-model-gemm-guards-final.7dt0nG`.
Its focused GREEN packet is in progress: three exact host tests and thirteen
actual GPU tests, including grouped48 backbone and24 head cases, negative
controls and both public Inference/Triad route-drift controls. These are planned
case counts until the raw exits and logs are inspected. Existing903/904 terminal
matrices are not rerun by this packet.

## Initial candidate findings

The CUDA+HF all-target and CUDA-only library checks passed, and all three
release test binaries built successfully in1m04s. Two exact host tests passed.
The owner-map host test then failed because its source parser matched
`Mamba3GpuInferenceMixedScratch` instead of the complete model type name.
Only the host contract file was corrected, with a prefix-collision regression;
its replacement SHA is
`348740c465e3fd2167556f4941f21c6a4bef96cfc4de5db1718d552ce05b4216`.
That host fix still awaits a run. No kernel or runtime change is implied.

Root continued on the unchanged original runtime binaries. Four exact GPU
groups passed: M3 eager-permit admission, half Inference missing-plan rejection,
both vendor modes' count/deny controls, and shared zero-work/vendor/poison guards.
The M1 matrix then failed at Triad/tensor-cores/B1/BF16 native: the recorded
projection groups were only `(1,64,18)` instead of the eight expected groups.
Earlier printed matrix rows are not counted as an overall M1 pass. This finding
is under diagnosis; it is not yet evidence of bad numeric output or performance.
The remaining eight original GPU groups are being collected separately without
recompiling or rerunning the four successful groups.

The remaining packet completed. Overall across the two immutable runtime
packets:10 exact GPU groups passed and3 failed. Both complete head matrices
passed (12 M1 +12 M3), as did both failed-step controls, explicit vendor graph
replay, and Inference route-drift rejection. The three failures are the M1
matrix, M3 matrix, and Triad route-drift fixture; all reproduce the same missing
native-half terminal inventory with tensor cores enabled. Source and binary
hash checks at the end of the remaining packet passed. Raw packets are under
`initial-green/`, `initial-gpu/` and `initial-gpu-remaining/`.

Diagnosis: context-aware homogeneous-half TC dispatch enters
`enqueue_half_gemm` with `NoPhysicalObserver`. That terminal emitted a physical
record only for `O::ENABLED` and did not publish a context GEMM route. Matvec
projections already recorded correctly, explaining the lone M1 shape. The
minimal fix is in the shared native-half launch environment/terminal, not the
model's expected inventory or numerical contract. It remains pending validation.

## Native-half correction packet

Frozen runtime directory: `/root/mamba-model-gemm-half-fix.xrQlBl`.
`gemm_bi_triad/launch.rs` SHA
`88f8d1a0ff1a30c83fcc00e5c946fab11dc92dfcbceaa52b8eeb65285e2dc34f`;
`blas.rs` SHA
`bcb5b3c393f4fca6e54a993e6c729c538d53ee64c96163f79e7073ee72f94729`.
Other runtime files stayed unchanged. CUDA+HF/CUDA-only compilation passed;
the test binaries built in57.15s. Seven focused host tests passed across the
build and resumed packets. Two host regressions still need correction: the
old environment-size test expects two pointers instead of the new three,
and the existing physical provenance census lacks previously introduced904
private/test-only scopes. Neither finding changes runtime arithmetic.

Actual GPU results from the corrected runtime:

| Exact group | Result | Time |
| --- | --- | --- |
| Direct typed-half projection, context/physical coexistence and capacity control | PASS |4.74s|
| Complete M1 matrix,30 cases | PASS |128.19s|
| Complete M3 matrix,18 cases | PASS |81.22s|
| Triad public graph route-drift rejection | PASS |18.83s|
| Physical native-half/exact-fallback graph,NN/TN/NT | PASS |26.86s|

All five GPU groups in this correction packet passed. The enclosing runner
exited1 solely for the stale host environment-size expectation; it is not
reported as an all-green packet. Source and binary hashes remained unchanged.
No repeated performance benchmark was run. Both original head matrices and
unaffected guard results remain separately attributed to their original packet.

## Host-only finalization

The native-half source's only later change is its cfg(test) environment-size
expectation, from two pointers to three. Root compared against the actual
runtime snapshot; no production line changed. Its final SHA is
`0f32c23b99dc812700a1bed0979d9856374b8be1fd89548b7b873947cc9137b9`.

The first final host packet passed5/8 exact tests, including that size test,
and Rustdoc completed in2.52s. Its existing private-link warning in untouched
`contract.rs` is retained for documentation cleanup; this is not warning-free
release acceptance. Three provenance tests hit one source-scanner false positive:
an existing `-> RecordedPhysicalTrace {` return signature was mistaken for a
forged struct literal. Only the contract target was corrected. The final
`e12519d450910daf8cf16beab09d8008e2a91b58e52c14ccbc85cde9044e15b9`
snapshot is rerunning those three host tests, with positive return-signature
and negative real-literal cases. No GPU/Rustdoc repeat is needed for this change.

The three final host tests now pass: provenance18.22s, cfg/private/count and
return/literal controls5.01s, private-method authorities2.02s. The final
`provenance-fix/` runner exited0; source and Cargo-input checks passed. All
source changes are committed at the SHA above. Failed packets are preserved
separately instead of being overwritten with successful results.

## Independent review follow-up

Review found that M1 mixed cleared a path's eager permit only when no graph
was installed. After warming both paths and capturing one, a failed attempt
on the other path could leave its old permit available for recapture. The
existing test expected that recapture to succeed; it did not cover the required
clear-before-any-attempt rule. This is a permit-lifecycle gap, not evidence of
wrong numeric output or a mismatched physical graph launching. A tests-first
correction is pending for that case; source96640d62 is not final task acceptance.

The expanded M1 regression reproduced the review finding in9.15s after a
59.05s library build (`permit-red/`, testexit101, expected-RED runner0). Its
tests-only source is preserved under `source-snapshots/permit-red/`.

The correction removes only the conditional wrapper around each matching
manifest clear at all ten entries (six M1, four M3). Captures, installed plans,
pointer/scratch checks and numerical launches are unchanged. Final hashes:
M1`69c53b2482489423a017a28d534b98d8cde4a13e6757a20b444e20850de6310e`;
M3`4017f1096db49382f0766ffef3fb2281138e3f8a33c313ac6f4b9418494c62d3`.
`permit-green/` passes the expanded M1 lifecycle test42.75s, the M3 lifecycle
test9.56s, and the exact owner-map host test0.03s. All eight installed-opposite
graph cases pass; positive recapture, installed-plan preservation and existing
scratch/poison controls remain. CUDA+HF/CUDA-only builds, Rustdoc and final
source/Cargo manifest checks pass; runner0. Scoped re-review accepted the fix:
one finding addressed, no new Critical/Important breakage. The original review,
accepted fix review and implementation report are archived in `review/`.

Final acceptance is model decode/head graph integration only. Constructor/API
and the remaining prefill/training owners are Task906; shipping test layout,
public documentation and final release verification remain later work.
