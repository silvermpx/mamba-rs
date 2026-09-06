# Ada homogeneous-half swizzle Task 1 final evidence review

## Verdict

**APPROVE for Task 1 force-only production qualification.** I found no blocking
specification or evidence-quality gap in the final package. The evidence supports
the report's bounded claim: the homogeneous BF16/F16 swizzle is production-loadable
and force-routable on the named CC8.9/142-SM board under matching CUDA 12.8, 13.0
and 13.2, with the reviewed functional/resource/identity contracts. It does **not**
support an AUTO admission, a routing-epoch change, a robust speed claim, universal
cross-GPU identity, or the separate F32-output mixed sibling.

This was a read-only evidence review. I did not rerun tests, build, use SSH, or
re-review the source; source was treated as the previously approved fix-1 hashes.

## Scope coverage checked

- **Release and static gates — covered.** The authoritative CUDA 12.8 rerun records
  636 library passes, 43 non-ignored performance passes, both focused static gates,
  the SM89 compile gate, holder/resource proof, and six zero exits
  (`final-release-host-static-warm-cuda128-rerun.log:688-830`). CUDA 13.0 and 13.2
  record the same 636/43 totals and five zero exits
  (`final-release-host-static-warm-cuda130.log:744-887`,
  `final-release-host-static-warm-cuda132.log:744-887`). The README correctly
  selects the 12.8 rerun and leaves the earlier unsuffixed failure as chronology
  (`README.md:10-14`).

- **Functional corpus — covered on all three toolkits.** Each final rounding,
  cross-rung/full-forced, and forced hot-A-E log has one pass and exit 0. The 12.8
  representatives are at `final-rounding-edges-cuda128.log:13-15`,
  `final-full-forced-corpus-cuda128.log:13-15`, and
  `final-forced-hot-a-e-cuda128.log:13-15`; the 13.0/13.2 files have the same final
  result structure. This covers the named runtime tests whose static implementation
  was approved in the fix-1 review; the terse harness transcripts are not being
  treated as an independent restatement of every assertion.

- **Cold/warm and resources — covered.** CUDA 12.8 cold-publishes the expected
  Fixed, TriadScalar and TriadSm80 artifacts and exits 0
  (`cuda128-cold-live-guard.log:7-16`), then the warm holder hits the same three
  artifacts and reports both typed resources (`cuda128-warm-live-resources.log:7-18`).
  CUDA 13.0 and 13.2 cold-publish all three artifacts, execute the holder, and report
  both dtype resources (`cuda130-cold-live-resources.log:62-73`,
  `cuda132-cold-live-resources.log:62-73`); their final release holder runs are warm
  hits. The observed contract is local/static shared 0, max threads 256, dynamic
  shared 69,632, occupancy 1, and registers 177/177/180, within the 224 cap.

- **Retained routes — covered.** Every `final-retained-rna-auto-cuda*.log` records
  two passes and exit 0 (for example, 12.8 at lines 469-471). I locally parsed each
  transcript: each has exactly 448 `RNA_WIDE_QUALIFIED_GROUP` rows, all 448 are
  unique and `actual_auto=true`, and the sorted row sets have the same SHA-256
  `03b8d1258810167bccd1009b6e85e2517c1360b85bf3d21ea70901fb195154e0`
  across all three toolkits. The six additional 13.2 retained gates each pass and
  end with `EXIT_STATUSES=0,0,0,0,0,0`
  (`final-retained-extra-cuda132.log:5-1369`). Main's independent summary also binds
  the six prior Triad cache-blob hashes
  (`main-independent-verification.json:115-123`).

- **Sanitizers — covered with correct API-error semantics.** Each authoritative
  final toolkit log runs memcheck, racecheck, initcheck and synccheck; all four test
  invocations pass, mem/init/sync report zero errors, racecheck reports zero hazards,
  and the aggregate exit is 0 (12.8 example:
  `final-sanitizers-cuda128-rerun.log:1-65`; equivalent 13.0/13.2 files). The exact
  command in the report retains `--error-exitcode 99` while setting
  `--report-api-errors no` (`ada-half-swizzle-force-report.md:148-161`), so device
  memory/race/init/sync findings remain fatal. I counted the preserved unsuppressed
  12.8 diagnostic: exactly 99 `CUDA_ERROR_INVALID_VALUE` reports, all from the
  deliberate terminal `cuFuncGetParamInfo` arity probe, no other `Program hit`
  type, followed by `ERROR SUMMARY: 99 errors` and result 99
  (`final-sanitizers-cuda128.log:6-1311`). It is diagnostic evidence, not a failed
  memory-safety gate; the rerun is the authoritative final sanitizer result.

- **Identity and copy provenance — covered.** Local digest validation found 48/48
  matches in `SHA256SUMS.raw` and 174/174 matches in
  `source-manifest-final.sha256`, with no missing files. The 174 source entries are
  set-identical to the source portion of `main-live-sha256.log`. The source manifest
  explicitly includes both new CUDA/layout files and all reviewed Rust/test inputs
  (`source-manifest-final.sha256:20-21,99,104,113,167,170,172`). The remote identity
  transcript records the expected absence of Git metadata, then the source hashes,
  three private mode-0700 caches per toolkit, 14 relevant release binaries and all
  nine cache blobs (`final-identities.log:1-44`). Thus the local baseline
  `86fd7f5605ef5aba77f4e51211af362edb929003` remains the only HEAD claim; no remote
  Git identity is inferred.

- **Composed/header identity — covered.** The report consistently binds the
  26-fragment, 688,275-byte Fixed source SHA and per-toolkit invocation, artifact,
  header-manifest and NVRTC-library identities
  (`ada-half-swizzle-force-report.md:193-220`). The distinct new header-manifest
  digests are correctly explained as including paste analysis of the changed
  composed source, rather than being reused from the RNA checkpoint. The three new
  `identity-cuda{128,130,132}.json` controls exactly match all shared identity fields
  in all 24 eager records and all three completion records; individual records also
  bind tuning revision 41. These controls contain no timing-derived admission.

- **Independent checks — consistent.** The copied integration report and static
  fix review are byte-identical to their authoritative ledger versions. Main's
  CUDA 13.0 transcripts independently show hot A-E 1/1 in 11.27 s
  (`main-hot-a-e-cuda130.log:2-5`) and holder/resources 1/1 in 2.94 s with exact
  Fixed/Triad cache hits (`main-holder-cache-cuda130.log:2-13`). The documented
  no-op trace-variable typo affects only the first transcript's logging, not its
  configured cache path or result; the holder rerun uses the correct trace variable
  (`main-independent-verification.json:135-148`).

## Claim-boundary checks

The eager smoke is properly constrained. Each toolkit emits exactly eight records:
BF16/F16 x bias off/on x both execution orders, all on `hot_a`, `path=eager`, and
`windows=1`; all raw/repeat/AUTO bit flags pass and every forced graph has exactly
one 256-thread, 69,632-byte node with the typed swizzle symbol and flat grid. Each
completion reports 8 accepted, 0 rejected, passed true (12.8 completion at
`final-eager-identity-smoke-cuda128.log:20-25`). Although the records necessarily
contain CUDA-event numbers, one sample makes p50 equal p95 and supplies no robust
speed evidence. The report and README explicitly call this identity/bit-path smoke,
not timing admission (`ada-half-swizzle-force-report.md:121-135,242-253`;
`README.md:24-27`).

The newly appended checkpoint paragraphs preserve that boundary. The handoff calls
the route a force checkpoint and sends the 21/101-window paired census to the next
task; the retirement ledger says no current-module timing admission exists and
keeps both Tc128 and pipeline non-retirable. Both correctly separate internal
p50+p95 admission from separately reported vendor gaps and reserve mixed F32 output
for independent qualification.

## Packaging observation (non-blocking)

`SHA256SUMS.raw` correctly freezes the worker's 48 raw logs that existed at README
finalization. The later main transcripts, identity-control JSON files, report/review
copies, and this review are intentionally outside that raw-log manifest. As already
planned by main, the forthcoming root evidence manifest should bind these
post-finalization files. This does not weaken the already validated 48-entry raw
manifest or the 174-entry source manifest, but it is the remaining packaging step
before the evidence directory itself is called fully frozen.

