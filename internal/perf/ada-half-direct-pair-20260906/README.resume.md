# Restored-access direct-pair evidence

This additive evidence completes the measurement gates left pending in the
committed partial archive.  It does not modify any file covered by the original
`SHA256SUMS` or `SHA256SUMS.partial` manifests.

## Qualified runs

The valid whole-run logs are:

- `direct-cuda128-w21-v2.log`
- `direct-cuda130-w21-v2.log`
- `direct-cuda128-w101-v2.log`
- `direct-cuda130-w101-v2.log`

Each log was produced by `run-direct-pair-v2.sh`, contains exactly 80 records
and one passing completion with zero rejections, and has successful PRE and
POST telemetry plus benchmark exits.  The matching
`verify-direct-cuda*-w*-v2.log` files are strict analyzer outputs.  The final
101-window result is the same on CUDA 12.8 and CUDA 13.0: the swizzle arm is the
direct p50+p95 winner in 17 of 20 row/cell/bias cohorts.  The three cohorts
without a robust winner are BF16 A0, F16 A0 and F16 A1.  This measurement stage
does not authorize an AUTO change.

## Wrapper correction and controls

`run-direct-pair-v2.sh` retains fail-closed identity, hash, command, active-app,
Driver and benchmark checks.  PRE additionally requires 0% GPU and memory
utilization.  POST requires successful exact-identity telemetry and an empty
compute-app inventory while recording, but not rejecting, recent utilization
attributable to the just-finished owned workload.

`test-run-direct-pair-v2.sh` has ten passing literal assertions, including
GPU-only and memory-only residual samples.  `telemetry-fail-closed-v2-final.log`
proves injected telemetry failure exits 70 before benchmark start;
`telemetry-positive-preflight-v2.log` proves the successful live preflight.

`direct-cuda128-w21-corrected.log` is preserved but excluded: the preceding v1
policy rejected a successful 80-record owned run because the immediate POST
sample contained 14% recent GPU utilization with no active compute process.
Earlier excluded logs remain governed by the committed partial README.

## Final reconciliation

`final-remote-reconciliation-v2.log` verifies the exact reviewed v2 wrapper and
test source, three frozen release binaries, two external identity controls, all
nine frozen cache blobs, cache modes, GPU UUID, driver, CC 8.9/142 SM and idle
GPU state.  Its first source-manifest transform used `awk`, which emitted a
format warning only for the replaced performance-test line; that line also
passed a separate exact hash check in the same log.  The companion
`final-remote-source-reconciliation-v2.log` uses a format-preserving transform
and cleanly verifies all 174 source inputs, including that changed test, with no
warning.  All reconciliation and SSH exits are zero.

`SHA256SUMS.resume` covers every additive file listed here and all preserved
restored-access diagnostic transcripts.
