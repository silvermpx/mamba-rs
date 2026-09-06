# Task 1 fix round 1: scoped static re-review

## Verdict

**APPROVED statically for fix round 1.** All four requested findings are addressed in the reviewed source, and I found no regression introduced by the fixes. This verdict covers specification and code quality only; it does not complete Task 1 because functional GPU corpora, sanitizers, retained cohorts, and final evidence validation are still pending.

Review basis: `half-swizzle-force-fix1-checkpoint.md`, `half-swizzle-force-fix1-review.diff`, and `half-swizzle-force-fix1-modules-review.diff`, compared with the original static-review package/report and `ada-half-swizzle-force-brief.md`. The nine source hashes are those recorded in the fix package; production launcher/kernel/holder files other than the added `modules.rs` test remain unchanged from the prior reviewed snapshot.

## Per-finding disposition

### ADDRESSED — P1 measured-callsite graph contract and eager-only capture

`tests/gemm_bi_fixed_performance.rs:11166-11240,11242-11273,11576-11634,12871-12944,12964-13051`

The half physical contract now receives and validates:

- exact dtype-specific pipeline/swizzle symbol and one-node inventory
- flat grid `ceil(M/128) * ceil(N/128),1,1`, block `256,1,1`, and per-tile shared bytes
- live Driver offsets/sizes `(0,8),(8,8),(16,8),(24,8),(32,32)` plus terminal-sixth rejection
- the four captured C/A/B/bias pointer values against the measured operands
- exact bundle bits `[1.0,0,M,N,K,K,N,N]` against the measured shape

`fixed_explicit_vendor_graph_inventory` derives these facts from the captured node itself before calling the pure contract. The identity predicate now forces untimed capture for both Ada half tiles under an eager-only path, as well as retaining graph-path/RNA behavior. The timing call site validates the AUTO identity graph when AUTO is an Ada half tile and the forced identity graph whenever the forced tile is pipeline or swizzle; graph timing reuses those already validated graphs. Eager-only output carries the available validated identity inventory instead of silently bypassing it.

The mutation test covers bad grid, shortened/wrong ABI, accepted sixth parameter, pointer mismatch, and wrong bundle in addition to symbol/block/shared/dtype. The eager-only predicate test covers both half tiles. No timing is authorized by these checks; they only bind physical identity to the measured operands as required.

### ADDRESSED — P2 re-poison rounding output between candidates

`tests/gemm_bi_fixed_sm89_pipeline.rs:683-768`

The test now builds a poison result whose every output element is the bitwise complement of the oracle and uploads it immediately before each pipeline/swizzle launch. Missing stores can no longer inherit correct bytes from the preceding candidate. Prefix/suffix guards remain their original values. The added finite-only A row and finite-output assertion prevent the corpus from degenerating into only NaN/overflow results. The corrected `let mut output` resolves the intermediate E0596 noted in the checkpoint.

### ADDRESSED — P2 production-bound exhaustive layout proof

`tests/cuda/gemm_bi_fixed_sm89_swizzle_layout.cpp:1-88`; `tests/arch_compile_gates.rs:19-74`

The new C++ proof includes `kernels/gemm_bi_fixed/sm89_half_swizzle_layout.cuh` directly. It exhaustively covers both stages and all 256-thread/four-slice copy chunks; checks 16-byte destination alignment, bounds, scalar/async agreement, collision/hole freedom; reconstructs A x4 and B x2-transposed `ldmatrix` lane values for every stage/warp/atom/issue; and verifies bank uniqueness for every matrix row group. Its reported expected totals—4,096 chunks, 98,304 fragment halves, 1,536 groups, 65,536 staging bytes, and 69,632 total bytes—follow from those loops. The Rust source-contract test pins the direct production-header include so the proof cannot quietly revert to the experiment helper.

### ADDRESSED statically; runtime still pending — forced hot A-E prefix/view/graph coverage

`tests/gemm_bi_fixed_sm89_pipeline.rs:79-217`

The existing hot-cell helper now accepts an optional forced tile. The old AUTO wrapper retains its CUDA-13.2 selection assertion, while the new forced-swizzle wrapper deliberately bypasses only that AUTO assertion. All other checks are shared: exact hot A-E dimensions, BF16/F16, bias absent/present, finite/exceptional corpora, exact hot and adjacent/tail M values, row-offset input views, aligned/misaligned output views with guards, raw oracle equality, exact physical graph contract, poisoned replay, and input/bias immutability. The forced path calls `fixed_forward_with_tile(..., Tc128Sm89Swizzle)` and asserts that returned identity.

This closes the static plan gap. It is not evidence that the ignored GPU test passed on any toolkit; those runs remain part of the pending runtime review.

## Added `modules.rs` composition-retention test

`src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:11331-11366`

The test strips the unchanged Fixed base from the SM89 composition, checks the five exact suffix boundaries in order, and compares the complete suffix byte-for-byte with composition of the incumbent pipeline, exact-N64, RNA-wide, production layout, and swizzle fragments in that order. It correctly detects composer omission, reordering, insertion, or transformation of the current fragments. It does not independently freeze historical file contents—source review/hashes establish that the three incumbent fragment files were not edited—but that is the proper division for this test. Existing non-Ada/compute-120/Triad exclusion checks remain intact. I found no tautology that would allow a different actual suffix order or extra composed bytes to pass.

## Fix-adjacent regression review

No new breakage found.

- The graph inventory still rejects non-kernel nodes for half candidates and still supports vendor non-kernel nodes. Five successful parameter queries plus a rejected sixth establish exact arity; pointer slots and bundle storage are checked non-null before reads.
- Capturing identity graphs before numeric-family rejection does not execute them and does not contaminate timings. Graph mode reuses the same objects; eager-only mode retains them solely as untimed evidence.
- On CUDA 12.8/13.0, portable `Tc128` AUTO is not unnecessarily forced through the Ada-half descriptor, while forced pipeline/swizzle is. On 13.2, incumbent pipeline AUTO and forced swizzle are both bound.
- The helper refactor preserves the original AUTO test behavior: only the forced wrapper skips the exact 13.2 AUTO assertion, and it still requires CC8.9/142 SM.
- The direct production host proof uses stage offsets matching the CUDA storage arrangement and keeps bank calculations stage-independent, which is valid because each stage displacement is a multiple of the 32-bank period.
- The new composer equality uses the same production fragment objects and allowed-include stripping as the actual composer, so the expected source models the real composition contract.

Minor wording such as the generic “pipeline” error label for either half tile is cosmetic and does not impair diagnosis because the message includes symbol, geometry, ABI, pointers, and bundle.

## Tests and evidence actually inspected

I inspected the code for:

- the performance pure graph-contract validator, eager-only identity predicate, mutation tests, node introspection, and both eager/graph timing call sites
- the forced hot A-E wrapper/shared helper and the corrected rounding-edge candidate loop
- the complete new production-header C++ layout proof and its Rust binding assertion
- the exact SM89 suffix composition assertion in `modules.rs`

I did not execute tests, builds, SSH, or GPU work. The checkpoint reports the direct C++ proof green, focused graph/resource/PTX tests green, the timing-verifier suite at 10 runs/48 assertions green, and final release host/static/warm suites on CUDA 12.8/13.0/13.2 (including the separate corrected 12.8 rerun) with zero exits. Those are supplied evidence, not independently reproduced results.

## Remaining Task-1 scope

The CUDA 12.8/13.0/13.2 full functional corpora, actual forced hot A-E test results, all four compute-sanitizer modes, retained incumbent pipeline/RNA/AUTO/Triad cohort results, final cold/warm cache/source/artifact identity reconciliation, manifest/log-key validation, and the worker's final report remain unverified here. No AUTO routing or tuning epoch change is present or approved, and no timing/performance conclusion follows from this static fix review.
