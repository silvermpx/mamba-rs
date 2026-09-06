# Ada half pipeline/swizzle direct-pair evidence

This directory records the bounded Task 2 continuation that directly compares
the two forced Ada homogeneous-half candidates.  It does not change or admit
AUTO.  Production CUDA/module/dispatcher state is the committed Task 1 state;
the only shipped source diff is the ignored performance-test workflow in
`tests/gemm_bi_fixed_performance.rs`.

## Functional and TDD evidence

- `red-rust-direct-helpers-cuda128.log`: intended missing-helper compile RED,
  exit 101.
- `green-rust-direct-helpers-final-cuda128-rerun.log`: final three focused Rust
  tests pass.  The preceding `green-rust-direct-helpers-final-cuda128.log`
  preserves a local zsh wrapper-variable mistake after the remote tests passed.
- `red-verifier-closed-schema.log` and `green-verifier-closed-schema.log`:
  supplied extra-field RED, then focused 1/6 and full 9/48 GREEN.
- `build-helper-nonignored-cuda130.log` and `cuda132.log`: exact matching build,
  three focused helpers and 46 nonignored performance tests all pass.  CUDA
  12.8's final helper and nonignored suite are in the dedicated logs.
- `focused-eager-smoke-cuda*.log`: 8 direct records per toolkit; mandatory live
  graph identities are validated even though the selected path is eager only.
- `full-smoke-cuda*.log`: full rows/cells/bias/path/order coverage, one window,
  exactly 80 records and one successful completion per toolkit.  One window is
  functionality only, never timing admission.
- `old-census-smoke-cuda*.log`: the unchanged V2 three-arm workflow still runs
  against the matching toolkits (8 records on 12.8/13.0, 4 on 13.2, matching
  the prior candidate sets).

## Telemetry correction and excluded attempts

`direct-cuda128-w21.log`, `direct-cuda130-w21.log`, and
`direct-cuda128-w101.log` are preserved diagnostic attempts and are excluded
from performance qualification.  Their wrapper requested unsupported
`nvidia-smi` field `multiprocessor_count`, causing both pre/post GPU queries to
fail while the benchmark exit remained zero.  Their adjacent `verify-*w21.log`
files establish schema/numeric/graph behavior only; they are not accepted
timing evidence.

`run-direct-pair.sh` is the corrected reusable fail-closed wrapper.  It uses
supported GPU telemetry, checks the exact UUID/CC and zero utilization, rejects
any active compute application, independently queries CUDA Driver attributes
CC 8.9/142 SM, checks source/binary/external-control hashes, and propagates all
pre/post telemetry and test failures.

The three failure-probe logs preserve the correction sequence precisely.
`telemetry-fail-closed-check.log` stopped before telemetry because the isolated
directory lacked the committed external control.  After that control was copied
byte-for-byte, `telemetry-fail-closed-check-final.log` reached injected
`/bin/false` telemetry and exited 70 before `BENCHMARK_START`, while also
exposing an invalid `date --iso-8601=nanoseconds` argument.
`telemetry-fail-closed-check-final2.log` uses the corrected `ns` argument and
records `PRE_DATE_EXIT=0`, `PRE_NVIDIA_SMI_EXIT=1`, `PRE_APPS_EXIT=1`,
`PRE_DRIVER_EXIT=0`, and `PRE_TELEMETRY_EXIT=70`, with no benchmark start.  The
final wrapper additionally rejects nonzero utilization and nonempty active-app
output; its live probe and fresh whole timing runs remain pending restoration of
Ada SSH connectivity.  Superseded diagnostics remain preserved.

## Final timing evidence

Fresh fail-closed 21/101 logs for CUDA 12.8 and 13.0, strict verifier outputs,
identity/cache/source reconciliation, checksums and the final per-cell report
will be added after the current connectivity interruption.  CUDA 13.2 requires
functional matching proof only and has no direct timing sweep.
