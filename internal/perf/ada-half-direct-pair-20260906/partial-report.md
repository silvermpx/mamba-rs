# Ada half forced direct-pair report

Status: implementation source-stable and all required three-toolkit functional
gates pass.  The required quiet 12.8/13.0 21/101 measurements are blocked by
loss of IPv6 connectivity to Ada after three telemetry-invalid diagnostic
attempts.  This report records the verified partial boundary; it is not an AUTO
admission or direct-performance conclusion.

## Scope and identity

Main advanced HEAD from the production baseline
`6570ce872aafe77bb94ecc5b6fee27fd48c054f7` to the evidence-only census
checkpoint `98ee1c2606ac59221c8a1271f04390043b27d5a6`.  Production CUDA, module
composition, loader, dispatcher and tuning revision remain unchanged.  The only
tracked direct-pair source diff is `tests/gemm_bi_fixed_performance.rs`:

- file SHA-256: `3873a40ac14a05615c5758bb3181948e6e4944486400387552de08a8ecb90d3a`
- diff SHA-256: `8c1ebabb4d99867d753b44d163c9d1c84071f379c166e810c1f1e441f6f9f23a`
- size: 622 insertions, 18 deletions; the deletions are the old filter body
  moved into the new pure strict helper
- local and isolated Ada copies have the same file hash
- direct `/Users/silvermpx/.cargo/bin/rustfmt --edition 2024 --check` and
  `git diff --check` pass

The bounded analysis-helper fix is separate from shipped source.  Current
SHA-256 values are:

- `verify-half-direct-pair.rb`: `c08e3535d127c618948c34f401b187c0422b79b0def97f962a8623369641f68f`
- `test-verify-half-direct-pair.rb`: `d3026c4dff692c76e4b745846c172ac60c9a22c9ca77aa189ef5e376d0d963b6`

Its exact closed-schema RED/GREEN and rationale are in
`half-direct-verifier-report.md`.  No old census verifier or raw log changed.

## Implementation

The ignored release-only `fixed_ada_half_forced_direct_pair` workflow has only
the two forced timed arms, `Tc128Sm89Pipeline` and `Tc128Sm89Swizzle`.  It uses
both explicit execution orders and computes both ratio directions from the
same-index raw samples.  AUTO and F32 PEDANTIC are untimed correctness proofs.
Every candidate output is independently overwritten with the bitwise complement
of its expected bytes before repeat/path/order checks.

Both measured callsites are captured and validated even for an eager-only run.
The validator receives the live tile, dtype, operands and shape and verifies the
single kernel, exact symbol, flat grid, block 256, shared bytes, five Driver
arguments with rejected sixth argument, four captured pointers and bundle.
Both graphs and buffers outlive all timing windows.  Each path warms each arm
128 times, calibrates independently from the existing 16-launch/~5 ms helper,
and emits one closed-schema V1 record per row/cell/bias/path/order.

Strict filters reject empty, duplicate and foreign values; any presence of
`MAMBA_FIXED_VENDOR_TILES` fails.  Rows are restricted to homogeneous BF16 and
F16, cells A-E, bias 0/1, and eager/graph paths.  Supported runtime identity is
exactly CC 8.9, 142 SMs, known NVRTC 12.8/13.0/13.2, tuning revision 41.

## Test-driven evidence

Rust RED on isolated CUDA 12.8 source/target:

```text
cargo test --release --features cuda,cudarc/cuda-12080 \
  --test gemm_bi_fixed_performance fixed_ada_direct_pair_ -- --nocapture

error[E0425]: cannot find function `fixed_ada_direct_pair_ordered_window`
error[E0425]: cannot find function `fixed_ada_direct_pair_ratio_quantiles`
error[E0425]: cannot find function `fixed_ada_direct_pair_filter`
error[E0425]: cannot find function `fixed_ada_direct_pair_reject_vendor_tiles`
RED_EXIT=101
```

After the minimal helpers/workflow were implemented, the focused final command
passed:

```text
cargo test --release --features cuda,cudarc/cuda-12080 \
  --test gemm_bi_fixed_performance fixed_ada_direct_pair_ -- --nocapture

3 passed; 0 failed; 0 ignored; 106 filtered out
EXIT=0
```

The tests use literal expectations: reversed order records actual closure
execution as `swizzle,pipeline`; paired p50/p95 includes a tail for which the
reverse p95 is demonstrably not the inverse forward p95; and empty, duplicate,
unknown rows plus any vendor-tile environment value are rejected.

Raw transcripts:

- `internal/perf/ada-half-direct-pair-20260906/red-rust-direct-helpers-cuda128.log`
- `internal/perf/ada-half-direct-pair-20260906/green-rust-direct-helpers-final-cuda128-rerun.log`

The first final wrapper run also had all three remote tests pass, then the local
zsh wrapper tried to assign read-only variable `status`.  That wrapper-only
failure is preserved in `green-rust-direct-helpers-final-cuda128.log`; the exact
remote command was rerun with `test_exit` and exited zero above.

The supplied verifier regression first observed the intended failure:

```text
1 runs, 1 assertions, 1 failures
auto_samples_us. RuntimeError expected but nothing was raised.
```

Exact record, completion, and external-control key sets were then enforced.
Focused GREEN is 1 run/6 assertions; full GREEN is 9 runs/48 assertions.  Raw
logs are `red-verifier-closed-schema.log` and
`green-verifier-closed-schema.log` in the evidence directory.

## Functional evidence so far

CUDA 12.8 focused eager-only smoke used rows BF16/F16, hot A, bias 0/1,
`MAMBA_FIXED_VENDOR_PATHS=eager`, and one window.  It exited zero with 8 exact
records and one successful completion in 4.40 seconds.  Despite selecting only
eager execution, every record contains exact live pipeline and swizzle graph
inventories.  AUTO is the expected `Tc128`; both forced results match its raw
storage bits and the repeat checks, and finite normalized errors match the
PEDANTIC bounds.  Raw log SHA-256 is
`648e4a98b5c6f7db2c40a4b9e753d42b17053e1e327cfd2bfc0e239df947fd8b`.

CUDA 12.8 complete functional smoke selected both paths, both rows, A-E, both
bias states and one window.  It exited zero with exactly 80 records and one
successful completion in 9.60 seconds.  Raw log SHA-256 is
`f901693a3283526ca37b24f62fa623b948a419be7390191b5c61c0283bd48aef`.
These one-window runs establish functionality only and are not timing evidence.

The exact remote runtime environment pins `CUDA_HOME`, `CUDA_PATH`,
`LD_LIBRARY_PATH` and `PATH` to `/usr/local/cuda-12.8`, pins
`CARGO_TARGET_DIR=/root/target-ada-half-direct-pair-cuda128-20260906`, and pins
`MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-half-direct-pair-cuda128-20260906`.
The run explicitly unsets `MAMBA_FIXED_VENDOR_TILES`, requires exact CC 8.9,
and invokes the frozen release test binary directly with `--exact --ignored
--nocapture --test-threads=1`.  Full commands, pre/post telemetry, hashes and
exit status are preserved verbatim in `focused-eager-smoke-cuda128.log` and
`full-smoke-cuda128.log`.

## Completed three-toolkit gates

All matching feature builds and tests use only SSH Ada and the isolated source,
targets and private caches.  The final release test binaries are:

| Toolkit | Binary SHA-256 | Helpers | Nonignored performance | Eager/full direct smoke | Old census smoke |
|---|---|---:|---:|---:|---:|
| 12.8 | `b887ffbd2d5c91ebd7cd1dcd8eeb3daad5c9400f5b967f58dc46150328067a78` | 3/3 | 46/46 | 8/80 | 8 |
| 13.0 | `d21a1a3cb284979289e05f042529be7f98f924e79147aa527766812f794540d5` | 3/3 | 46/46 | 8/80 | 8 |
| 13.2 | `422499ce8485e998412effac9ab172e6341704225536a51c90d73678aeb3d421` | 3/3 | 46/46 | 8/80 | 4 |

Every direct smoke has one successful completion with `rejected:0`; all three
full smokes cover exactly 80 unique row/cell/bias/path/order records.  The old
census counts match its two-candidate 12.8/13.0 and one-candidate 13.2 scopes.
The live records bind the committed per-toolkit Fixed source, invocation,
artifact, header-manifest and known-library-domain controls and tuning revision 41.
CUDA 13.2 has no timing obligation beyond these passing functional gates.

## Telemetry-wrapper correction in progress

The initial 12.8/13.0 21-window attempts and an already-running 12.8 101-window
attempt are excluded from performance qualification.  Their wrapper requested
unsupported `nvidia-smi` field `multiprocessor_count`; both immediate GPU queries
failed, while the wrapper propagated only the successful benchmark exit.  The
raw logs are preserved unchanged as diagnostics and will not be spliced or used
for admission.  Their numerical/graph payloads remain useful only as provisional
debugging output.

`internal/perf/ada-half-direct-pair-20260906/run-direct-pair.sh` is the corrected
reusable wrapper.  It checks the supported GPU telemetry result, exact UUID/CC,
zero pre/post GPU and memory utilization, empty active-compute inventory, and
read-only CUDA Driver attributes CC 8.9/142 SM.  It also verifies immutable
source, test binary and committed external-control hashes before telemetry and
propagates every pre/post telemetry or benchmark failure.

The three preserved failure-probe logs have distinct outcomes.
`telemetry-fail-closed-check.log` stopped before telemetry because the isolated
remote directory initially lacked the committed external control.
`telemetry-fail-closed-check-final.log` reached injected `/bin/false` telemetry
and exited 70 before `BENCHMARK_START`, but it also exposed the wrapper's invalid
`date --iso-8601=nanoseconds` argument.  After changing that argument to `ns`,
`telemetry-fail-closed-check-final2.log` records `PRE_DATE_EXIT=0`,
`PRE_NVIDIA_SMI_EXIT=1`, `PRE_APPS_EXIT=1`, `PRE_DRIVER_EXIT=0`, and
`PRE_TELEMETRY_EXIT=70`, with no benchmark start.  The final wrapper was then
strengthened to reject nonzero utilization and nonempty active-app output; its
live failure probe could not reach Ada because connectivity failed before the
remote command.  The final local wrapper SHA-256 is
`9770e2f4ffedae71d65952ce456a1a36722ec0bc9a29d52f4d2531832e1d014a`;
`bash -n` and `shellcheck` pass.  It still requires final remote injection and
successful live preflight before use.  Fresh whole 21/101 runs will use only
this corrected wrapper and new log names.

## Blocked boundary

After the already-running invalid 12.8/101 attempt finished, every SSH command
failed before remote execution against the configured literal IPv6 address
`2a01:4f8:1060:20d5::2` (`AddressFamily inet6`) with exit 255 and
`Inappropriate ioctl for device` or `Operation not supported on socket`.
Read-only routing inspection shows that destination routed through `utun6`.
Main independently reproduced the same normal-SSH failure and found no scoped
alternate IPv4 address.  No SSH, VPN, route, host-key or security configuration
was changed, and no alternate GPU host was used.  No worker GPU job is active.

The exclusive Ada lane is explicitly released, with no active worker job.
Task completion still requires after connectivity is restored: syncing and
testing the final corrected wrapper; fresh complete fail-closed 21 and 101 runs
on CUDA 12.8 and 13.0; strict independent verification; exact final remote
source/binary/cache reconciliation; all twenty per-toolkit conclusions; and
final evidence checksums.  Until those gates run, there is no fastest-route or
AUTO-admission claim.
