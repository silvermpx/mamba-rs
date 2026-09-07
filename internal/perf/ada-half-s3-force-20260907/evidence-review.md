# Task6A final evidence review

## Verdicts

**Spec compliance: Approved.** All Task6A source, functional, retained-route, artifact and handoff requirements are closed by the reviewed evidence.

**Task quality: Approved.** No material source or evidence issue remains. Source/fix1 approval stands; this report does not re-review unchanged source or establish performance/AUTO admission.

## Functional matrix

Evidence paths below are relative to `internal/perf/ada-half-s3-force-20260907/` unless otherwise stated.

| Required gate | CUDA12.8 | CUDA13.0 | CUDA13.2 |
| --- | --- | --- | --- |
| Library/static force/source/SM89 compile | 644 / 48 / 2 / 1 pass | 644 / 48 / 2 / 1 pass | 644 / 48 / 2 / 1 pass |
| Independent cold and warm holders | Both pass | Both pass | Both pass |
| Full half pipeline/S2/S3 corpus | 11 pass | 11 pass | 11 pass |
| Retained RNA-wide | 2 pass | 2 pass | 2 pass |
| Retained exact N64 | 5 applicable pass in continuation | 5 applicable pass | All6 pass, including AUTO |
| Retained TF32-C AUTO | Requires13.2 | Requires13.2 | 1 pass |
| Explicit eager S3 physical identity | 8 records, zero rejects | 8 records, zero rejects | 8 records, zero rejects |
| Memcheck/racecheck/synccheck | Clean, all exit0 | Clean, all exit0 | Clean, all exit0 |
| PTX/SASS/resources/cache proof | Pass,182 registers/dtype | Pass,182 registers/dtype | Pass,188 registers/dtype |
| SM120 compile contracts / actual SM89 cohorts | Final13.2 gate | Final13.2 gate | 2 compile tests / 2 live cohort tests pass |

The common matrix stages have explicit successful exit records in `matrix-cuda128-attempt1.log:14,25,36,47,58,69,80,91`, `matrix-cuda130-attempt1.log:14,25,36,47,58,69,80,91,102`, and `matrix-cuda132-attempt1.log:14,25,36,47,58,69,80,91,102,113`. Toolkit completion is recorded at `matrix-cuda128-continuation1.log:127`, `matrix-cuda130-attempt1.log:215`, and `matrix-cuda132-attempt1.log:270`; the last also records `FINAL_MATRIX132_EXIT=0` at line271.

The library result retains46 intentionally ignored tests and the force-static result retains63; these are not represented as executed GPU tests. The separately invoked hardware suites supply the task-specific live coverage.

The actual full-half test results name the S3 hot A–E, raw alpha/beta/stride, full prefix/view/replay, public rejection, rounding and incumbent AUTO gates (`cuda130-attempt1/half-full.log:5` through its summary at line22, and the corresponding12.8/13.2 logs). Their previously reviewed code validates exact output bits and actual captured parameters, with independent output reset/readback before replay. The public overflow-rejection fix is included in all three final full suites.

## Preserved12.8 failure and continuation

`cuda128-attempt1/retained-exact.log:10` shows the sole initial failure is the existing AUTO test's explicit toolkit precondition, `(12,8)` versus `(13,2)`, not an observed numeric failure. Its failed summary remains at line610 and the wrapper preserves exit101 (`matrix-cuda128-attempt1.log:102`).

A focused source-precondition check confirms the exact-AUTO test requires13.2 (`tests/gemm_bi_fixed_sm89_exact_n64.rs:1226`), as does TF32-C (`tests/gemm_bi_fixed_performance.rs:1359`). The continuation skips exactly the former on12.8 and completes all five applicable tests (`matrix-cuda128-continuation1.log:13,14,21`). The13.0 run applies the same scoped skip, while13.2 runs all six exact tests and TF32-C successfully (`cuda132-attempt1/retained-exact.log:1302`, `cuda132-attempt1/retained-tf32-c.log:7`). No test was weakened or deleted to make the continuation pass.

`run-continuation128.sh:9,10,16` retains the original target and attempt1 cache and requires that cache to exist. Only the evidence directory is new. Root independently reconciled the same frozen source/build inputs and saved artifact identities. The initial failure is retained alongside the successful continuation rather than overwritten.

## ABI, physical identity and resource closure

The source/fix1 reviews already established independent cold/warm Driver ABI census, exact five-argument layout including terminal probe, optional holder/rejection, and physical parameter decoding. All three final cold/warm tests succeed; root independently verified complete compiler/artifact identity equality between each pair. Both typed S3 resource rows are present in every final full-half log (lines15–16): local0, static0, maxthreads256, dynamic98304 and occupancy1; register counts are182/182/188 by toolkit.

The eight raw eager records in each `eager-physical.log` cover BF16/F16 × bias/no-bias × both execution orders. I inspected their decoded records: every forced route is `Tc128Sm89S3`, bit/repeat checks are true, each physical graph contains exactly one corresponding typed S3 kernel, no non-kernel node, grid592×1×1, block256×1×1, and98304 shared bytes. Each completion record reports8 records,0 rejected, passed=true. These are physical-identity controls with one timing window; no speedup or vendor-win claim is inferred.

The per-toolkit identity JSONs match the saved artifact proofs: shared composed Fixed source digest `205f58b56429b8e74f3ac1e7ab9f0cf6a9bf193ffeb87b254bd3ab00e6b02ca5`, targetsm89, actual NVRTC versions, known library domains, CC8.9/142SM and revision42. Saved PTX/artifact digests are:

- 12.8: `60977db33de28d807ac7dd3dafe2d176914f7b7eab589917a47988050712352c`.
- 13.0: `d1aa6e33a612d99cd44a1e9c7eebe495c05eeef2212bf2cf4296db55eedb86a9`.
- 13.2: `8b89aadf7d456b78ab249ff614568730946006eefd7143e35bce1e158741aff1`.

`analyze-artifact.py:8` validates the cache envelope magic/version/type, payload length/digest and byte equality to the saved PTX. Its per-dtype loop at line20 selects each exact S3 symbol's own SASS/resource/ptxas block, enforces cap188 with zero stack/static/local and zero spills, rejects local-memory and numeric atomic/reduction instructions, and requires commit/wait0/wait1/HMMA/LDSM/async-copy/barrier instructions. Both typed records in all three `artifact-proof.log` files pass. Raw `resources.log:5` and `ptxas.log:3,8` agree with those resource/spill conclusions. This is offline ptxas/SASS inspection of the exact production NVRTC PTX, complemented by the live Driver census; it is not misrepresented as a captured Driver-JIT cubin.

## Sanitizers and retained cohorts

Every selected sanitizer executes the actual final pipeline binary, runs the named bounded smoke test successfully, and reports zero errors; racecheck also reports zero warnings. See line8 of `cuda128-continuation1/sanitizer-{memcheck,racecheck,synccheck}.log`, `cuda130-attempt1/sanitizer-{memcheck,racecheck,synccheck}.log`, and `cuda132-attempt1/sanitizer-{memcheck,racecheck,synccheck}.log`. Wrapper exits are independently recorded, using error-exitcode99. The runner uniquely resolves and hashes the binary before use (`run-toolkit.sh:51`).

Retained RNA tests complete both actual-AUTO groups per toolkit; the13.2 raw cohort log actually reports served TriadSm80 symbols for five cells (`cuda132-attempt1/retained-cohort.log:5`), and the bias cohort reports the qualified wide epilogues (`retained-bias-cohort.log:5`). Both summary lines12 pass. The two SM120 logs each pass their compile contract at line7; these are compile-only evidence, not live SM120 behavior. `final-triad-cache-byte-equivalence.log:1` records a changed Fixed artifact but byte-identical retained Triad cache objects for each toolkit and ends in comparison exit0.

## Runner semantics, identities and release

The selected final runner uses fresh attempt/cache directories and noclobber logs, records command status, and stops on failure. The continuation intentionally reuses the existing12.8 cache. Identity subcommands run with errexit/pipefail and explicit stage exits; artifacts and hashes likewise have checked exit paths. The proof invoked in both runners is `analyze-artifact.py`, not the unused Ruby alternative.

Root independently reverified the seven frozen source hashes, all356 recorded source/build inputs for each completed toolkit, saved PTX hashes and full cold/warm identity equality. This review uses that work for byte identity and independently checks the coverage/resource/physical semantics above. AUTO remains revision42 throughout the raw physical records and source approval remains in force.

`lane-release.log:1` records release at2026-09-07T02:48:19Z. The exact GPU is `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, RTX6000 Ada CC8.9, driver595.45.04, utilization0%; `COMPUTE_APPS=` is empty and both identity/no-app checks exit0. This proves the recorded lane release, not perpetual future idleness.

## Final report and rooted manifest

Read the complete consolidated `final-report.md`; its final source/cap/guard status, per-toolkit counts, preserved failures, compile-only SM120 limitation and timing-only next step agree with the independently inspected raw results. The original task-report file and evidence copy both hash to `fa21201590a09e86f926ae80ffb6c9a1272ab3d6c45abdd0bc37e72f32670339`, independently checked during this review.

`SHA256SUMS` hashes to `9899ead9a0d4e9a5ec29fcb4eae68dfc34b9b450ae432ef88d14c182c7992aff`, also independently checked. Root independently recomputed all122 rooted entries from the worktree root and reported all OK; `manifest-verification.log` records the complete successful list. The manifest excludes itself and its verification output while including the final report copy, authoritative archives and raw evidence.

`archive-verification.json:2` records PASS and all three toolkit rows:356 source/build inputs each,6+7+7 test binaries and3+3+3 cache envelopes, all Fixed identities matched. Inspection of `verify-archives.py` confirms it hashes actual archive-member bytes against the corresponding final identity log and matches each Fixed compile key/PTX digest against physical-control identity. AppleDouble metadata alone is excluded from source inputs. Initial archive/preparation-check failures are retained and explicitly distinguished from the authoritative v2 source/12.8 archives. No archive or test rerun was needed for this review.

## Findings and acceptance scope

No material finding in the completed functional/resource/artifact evidence. I1 remains closed by fix1. The historical12.8 precondition failure is explained and covered by the completed applicable continuation and13.2 AUTO run.

No remaining unverified Task6A requirement. The supported handoff is **ready for production paired timing**; Task6A does not authorize an AUTO/revision change or establish production speedup/cuBLAS victory. Actual SM120 runtime remains outside this Ada compile-only validation scope.

Review actions were local read-only inspection and parsing of existing evidence. No GPU, SSH, build/test reruns, source edits, staging, commits or agents were used. Only this authorized review report was written.
