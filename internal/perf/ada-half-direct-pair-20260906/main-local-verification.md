# Main verification — partial checkpoint, 2026-09-06

This checkpoint preserves a tested harness and diagnostic evidence. It does not
finish direct timing qualification, choose the fastest production route, change
AUTO, or claim that inference/Triad beats cuBLAS FAST everywhere.

## Source and local checks

- Parent commit: `98ee1c2606ac59221c8a1271f04390043b27d5a6`.
- Only shipped source change: `tests/gemm_bi_fixed_performance.rs`, SHA-256
  `3873a40ac14a05615c5758bb3181948e6e4944486400387552de08a8ecb90d3a`.
- The prior 174-input source inventory differs only at that test file; the
  other 173 qualification inputs are unchanged. Production modules, CUDA
  kernels, loader, dispatcher and tuning revision 41 remain unchanged.
- Fresh direct `rustfmt --edition 2024 --check`, `bash -n`, `shellcheck`, and
  source/document `git diff --check` passed locally.
- Archived `analysis/test-verify-half-direct-pair.rb`: fresh 9 tests,
  48 assertions, zero failures/errors/skips.
- The worker partial manifest's 29 entries independently checked successfully.
  The enclosing `SHA256SUMS` also covers later analysis, reports and reviews;
  run `shasum -a 256 -c
  internal/perf/ada-half-direct-pair-20260906/SHA256SUMS` from the worktree root.

## Functional evidence boundary

Main independently checked the exact eight-record eager and eighty-record full
one-window smokes for all three toolkits: closed schema, complete unique roster,
external identity controls, both physical graph identities and raw-bit checks.
Matching nonignored performance suites report 46 passes and 63 ignored per
toolkit. The three focused helper tests pass per toolkit. These are functional
checks, not performance admission or quiet-device telemetry qualification.
The source harness and closed-schema analyzer fix have independent approvals;
their archived reviews retain their original, narrower review-time boundaries.

The three excluded multi-window attempts remain byte-preserved with reasons and
hashes in `analysis/excluded-timing-runs.json`. Any accompanying analytical JSON
or verifier output is diagnostic only. The prior committed three-arm census
is unaffected; its confirmed results remain in the separate census archive.

## Pending work and access

The final strengthened wrapper needs a live failure probe and positive preflight,
followed by fresh complete 21/101 runs on CUDA 12.8 and 13.0, strict analysis and
final remote source/binary/cache reconciliation. None is claimed by this commit.
Normal SSH to Ada failed before remote execution, exit 255, with its configured
IPv6 destination routed through `utun6`. Root independently reproduced this and
found no same-host IPv4 in scoped project records. No route, VPN, SSH or host-key
configuration was changed. The worker released Ada; no job remains active.
Resume only after access is restored or the owner supplies this machine's IPv4.
