# Ada Fixed deterministic TF32: three-toolkit AUTO qualification

2026-09-06, RTX6000Ada CC8.9/142SM, driver595.45.04, CUDA12.8/13.0/13.2.
Basec3564636; source manifest
`21f828115c983114175615bf6b6c044c461b811d3cc11aad787697dff3f7a574`.

The existing Fixed RNA-wide winner is now selected by actual AUTO on all
three explicitly qualified NVRTC versions, with every existing shape,
alignment, holder, known-library, device, dtype and policy gate retained.
Routing epoch41 invalidates previously captured graphs; recapture is required.
No CUDA body, composer/loader, numeric/ABI/schedule or other selector changed.
Ordinary fallback kernels are retained; nothing is deletion-ready.

## Measured outcome

All builds and functional work finished before quiet101 timing. Five planned
invocations produce120records:32+8 on12.8,32+8 on13.0,40 on13.2, no rejects.
Every shape/bias case wins against the correct previous AUTO tile at both
samplewise p50 and p95 in all eager/graph/order cohorts. Ratios below are the
worst paired p95 across those cohorts; less than1 means lower time.

| CUDA | Own wins | Range AUTO/old | FAST wins | AUTO/FAST for winning cases | Worst remaining AUTO/FAST |
| --- | ---: | ---: | --- | --- | ---: |
|12.8|10/10|0.648678--0.972474|A1|0.956254|1.479948 (E0)|
|13.0|10/10|0.654364--0.992156|A1|0.975619|1.487687 (E0)|
|13.2|10/10|0.629951--0.918121|A1,B1|0.908820,0.982497|1.345904 (E0)|

Shapes(M,K,N): A4621/384/1928; B4621/768/2304; C4621/1928/384;
D2048/768/2304; E2048/2304/768. Suffix0/1 is absent/present bias.
C's previous route is M128S2 on12.8/13.0 and M64S2 on13.2; all others use
M64S2. FAST is explicitly `CUBLAS_COMPUTE_32F_FAST_TF32`, with bias broadcast
included in timing. PEDANTIC is only the numerical reference, not the timing
denominator.13.0 D1's internal p95 margin is only0.8%, not a large gain.

These results leave9/9/8 TF32 FAST gaps by toolkit. Half/mixed, exactF32,
Triad and other GPUs are not closed by this checkpoint. There are no fresh
RTX5090 results here and no new full architecture-suite or clippy run.

## Correctness, reachability and identity

- Each toolkit passes634 library tests,43 performance-static tests and the
  exact two actual-AUTO GPU wrappers, with448 expected unique view groups.
  The corpus retains all five old-rung bit comparisons, both bias states,
  finite/exceptional values, prefixes/row views/C4/C16, guarded buffers,
  immutable inputs, eager repeats and poisoned graph replays.
- CUDA13.2 retained Ada Triad cohort tests2/2 and C-prefix1/1 pass. Main reran
  the full13.0 actual-AUTO pair independently:2/2,448groups,38.69s.
- Main and reviewer independently checked120 raw timing records, physical
  AUTO and forced-old symbols/geometry, epoch41, known matching NVRTC,
  explicit FAST/bias semantics and all required bit flags.
- Main checked168 local/remote source inputs,162 unchanged inputs against
  base, six live correctness/performance binaries and all nine cache blobs.
  `main-remote-check.log` records the exact remote command, paths and hashes;
  `main-independent-verification.json` records cache keys and their comparison
  basis in the preceding full-toolkit qualification bundle.

## Evidence and preserved failures

`raw/` is an immutable copy of39 worker artifacts plus its SHA256SUMS
(manifest SHA3173d5830bb4272ed088abe37c76e102635c581cefe701f2c1bcd8baa39acf38).
No raw-log whitespace was rewritten. `integration-report.md` preserves the
full acquisition report, commands, six source hashes, resources, binary/cache
identities and per-cell ratios. `review.md` contains independent review.
Main proof is in the `main-*` files; `main-timing-verification.json` is a
separate samplewise recomputation, not an inverted forced/AUTO quantile.

The intended selector, epoch40 and old hot-A GPU RED failures are retained.
`red-selector-cuda128.log` selected zero tests and is explicitly not RED proof.
`green-lib-cuda128.log` is the preserved632pass/2failure first full-suite run:
two current-epoch fixtures were then corrected. Initial green build logs use
pre-amend manifest0f8762c1dada48707d404a39ba1357b3b5d65e8c11f198998f27e9b84281e483;
the final library recompiles and subsequent targets use final21f828... source.
The final six binary hashes were independently checked live by main.

From the worktree root, reproduce main's read-only timing validation:

```sh
ruby internal/perf/ada-rna-toolkit-auto-20260906/main-verify-timing.rb 12.8 41 \
  internal/perf/ada-rna-toolkit-auto-20260906/raw/timing101-cuda128-abde-m64.log \
  internal/perf/ada-rna-toolkit-auto-20260906/raw/timing101-cuda128-c-m128.log
ruby internal/perf/ada-rna-toolkit-auto-20260906/main-verify-timing.rb 13.0 41 \
  internal/perf/ada-rna-toolkit-auto-20260906/raw/timing101-cuda130-abde-m64.log \
  internal/perf/ada-rna-toolkit-auto-20260906/raw/timing101-cuda130-c-m128.log
ruby internal/perf/ada-rna-toolkit-auto-20260906/main-verify-timing.rb 13.2 41 \
  internal/perf/ada-rna-toolkit-auto-20260906/raw/timing101-cuda132-all-m64.log
```

`main-verify-corpus.rb final <logs...>` verifies the exact two AUTO wrappers
and448-key set per log. The helper's `legacy` mode is only for its historical
one-wrapper self-check. Final validation never uses that mode. The timing
helper rejects wrong partitions, missing/repeated records, stale identities
and any failed internal promotion; its JSON is diagnostic even on failure.
