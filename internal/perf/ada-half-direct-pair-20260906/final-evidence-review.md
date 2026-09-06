# Ada half direct-pair final evidence review

## Verdict

**APPROVE — Task 2 evidence is complete; no blocking finding.** The four fresh
v2 whole runs satisfy the brief's 21/101 measurement scope on CUDA 12.8 and
13.0, bind the approved source/harness and external revision-41 identities, and
support the reported direct winner matrix. This verdict establishes only the
forced pipeline-versus-swizzle ranking. It does not authorize an AUTO selector,
tuning-revision, CUDA, loader, or dispatcher change.

The reviewed final report SHA-256 is
`f111b949c2970e8390fc5deeabfa1ccbb14dea35ff206b75a07564299ff5d951`;
`README.resume.md` is
`74f6f367bb00db6f6224c1e978f1c956e7336a1c4d8c4fe9df0e4bcb8177c711`;
and `SHA256SUMS.resume` is
`2799274c6f55f43aa56267cf8c087bd3cd00df2db7fcd3bd4d05292862e98f43`.
All 22 additive manifest entries passed `sha256sum -c`. I performed no SSH/GPU,
did not rerun the approved Rust/Ruby suites, and did not re-review the approved
harness, analyzer, or wrapper source.

## Evidence checks

- **Complete fresh cohorts:** each of the four `direct-cuda*-w*-v2.log` files
  contains exactly 81 recognized JSON objects: 80 unique records spanning 2
  dtypes x 5 cells x 2 bias states x 2 paths x 2 orders, plus one exact-schema
  completion with `records:80`, `rejected:0`, and `passed:true`. The two screens
  have exactly 21 positive samples per arm/record; the two final runs have 101.
  Every test result and wrapper benchmark status is zero
  (`ada-half-direct-pair-report.md:205-239`).

- **Closed schema, correctness, and physical identity:** I independently checked
  every record/completion key set against the approved verifier constants. All
  records bind homogeneous BF16/F16, exact A-E dimensions, NN/alpha1/beta0,
  CUDA-event timing, forced pipeline/swizzle route names, actual old AUTO Tc128,
  PEDANTIC F32 reference, finite in-tolerance errors, iteration range 1..4096,
  and all required bit/replay booleans. Both graph inventories have one kernel,
  zero non-kernel nodes, the exact typed symbol, flat shape-derived grid, block
  256, and shared bytes 71,680/69,632. The already-approved live harness supplies
  the non-serialized five-argument/pointer/bundle gate.

- **External identities:** all record and completion identity fields match the
  committed external controls byte-for-byte: CUDA 12.8 control SHA-256
  `8ab62c7f1da94b7f61ef81efac7c4d54ae8f98830748e6e8bee27e94558763e3`
  and CUDA 13.0 control SHA-256
  `0489790af9824b9459b4fe9f56641fd884052e70c2b8c0b946aa9252eeaef301`.
  This includes CC8.9, 142 SM, known library, compiler target, Fixed source,
  invocation/artifact/header/library-domain digests, and tuning revision 41.
  Each strict verifier output binds the correct raw-log and identity-file hash
  and says `required_census_complete:true` and
  `auto_admission_authorized:false`.

- **Raw quantiles and report matrix:** I recomputed each arm p50 and both
  samplewise ratio directions' p50/p95 directly from every raw array using the
  specified order statistic, then took the worst value across eager/graph and
  both orders per dtype/cell/bias. All emitted quantiles matched within 1e-9.
  All 40 final table pairs in `ada-half-direct-pair-report.md:254-275` match the
  independent recomputation to the printed six decimals. CUDA 12.8 and 13.0
  each have the same 17 robust swizzle winners; BF16 A0, F16 A0, and F16 A1 have
  no direct winner, and pipeline is not a robust winner in those three. The
  21-window screens independently produce 18 swizzle winners on each toolkit,
  correctly reported as screen rather than final ranking.

- **Telemetry and run separation:** all four v2 logs start with exact UUID/model/
  CC, 0% GPU and memory utilization, empty compute-app output, Driver 8.9/142,
  and successful PRE date/query/app/Driver statuses. All POST queries, identity,
  app inventory, Driver checks, wrapper/test statuses, and final date statuses
  succeed. POST preserves recent utilization (2%, 9%/1%, 71%, or 34%) without
  rejecting it, consistent with the reviewed phase-specific NVML ruling; every
  POST compute-app inventory is empty. Timestamps show the four qualifying
  windows are sequential, not overlapping. Commands in each raw log pin the
  toolkit, target, private cache, exact filters, absent vendor tiles, source,
  binary, and control hashes.

- **Fail-closed wrapper evidence:** the live injected-failure transcript records
  failed GPU/app queries, PRE telemetry exit 70, and no benchmark start. The
  following positive live preflight records exact wrapper hash, target identity,
  0%/0%, empty apps, Driver 8.9/142, and all-zero statuses
  (`ada-half-direct-pair-report.md:179-203`). This complements the previously
  approved ten-case actual-function regression, including GPU-only and
  memory-only PRE rejection/POST acceptance.

- **Excluded evidence stays excluded:** the three unsupported-field runs remain
  bound by the original `analysis/excluded-timing-runs.json`. The later complete
  `direct-cuda128-w21-corrected.log` remains additive but is explicitly excluded
  in `README.resume.md:38-41` and the final report at lines 170-177 because v1
  returned 70 on its POST residual-utilization rule. Its valid 80-record payload
  is not used in the final matrix, and no run was rewritten or spliced.

- **Final reconciliation:** `final-remote-source-reconciliation-v2.log` supplies
  the clean integrated 174/174 source proof, including the reviewed performance
  test. `final-remote-reconciliation-v2.log` separately verifies wrapper/test
  sources, all three release binaries, both controls, nine cache blobs and 0700
  cache modes, plus final idle target identity; all recorded substantive statuses
  are zero. The first transform's changed-test-line formatting warning is
  truthfully bounded: that line passes its separate exact check in the same log,
  while the clean companion is authoritative for the full 174-source set
  (`README.resume.md:43-53`; report lines 284-300).

- **Archive and release boundary:** the additive README accurately says it does
  not modify files covered by the original manifests and scopes the resume
  manifest to restored-access additions. The report separately preserves the
  prior functional three-toolkit gates, makes no vendor-win or AUTO claim, states
  that no remote jobs remain, and explicitly releases the Ada lane
  (`ada-half-direct-pair-report.md:277-305`).

No Task 2 evidence correction or performance rerun is required. Any later AUTO
implementation remains a distinct revision-42 task with its own selector,
actual-AUTO, identity, and post-promotion performance gates.
