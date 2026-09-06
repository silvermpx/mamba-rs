# Ada half forced direct-pair report

Status: complete.  The implementation is source-stable, all required
three-toolkit functional gates pass, and fresh fail-closed whole 21/101 direct
measurements on CUDA 12.8 and 13.0 have passed strict independent verification.
This is a forced-arm performance conclusion only; it does not authorize an AUTO
change.

## Scope and identity

Main advanced HEAD from the production baseline
`6570ce872aafe77bb94ecc5b6fee27fd48c054f7` through the evidence-only census
checkpoint `98ee1c2606ac59221c8a1271f04390043b27d5a6` to the partial direct-pair
checkpoint `21eebcd1abe2685ad498496036931ba443ee5f80`.  Production CUDA, module
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

## Telemetry correction and restored-access qualification

The initial 12.8/13.0 21-window attempts and the initial 12.8 101-window
attempt remain excluded.  Their wrapper requested unsupported `nvidia-smi`
field `multiprocessor_count` and did not propagate the telemetry failure.  The
three earlier failure-probe logs and the IPv6 outage boundary remain documented
in the committed partial archive; none was rewritten, spliced or used for a
performance conclusion.

After access was restored, `resume-preflight-identities.log` proved host
`ada-6000`, GPU UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, driver
595.45.04, 0% GPU/memory utilization, no compute applications and Driver CC
8.9/142 SM.  The test source, all three release binaries, both external timing
controls and all nine private cache blobs matched their frozen hashes.

The corrected v1 wrapper then exposed a narrower behavioral RED.  Its first
fresh whole CUDA 12.8 21-window test emitted 80 valid records and a successful
completion, but the wrapper returned 70 because immediate POST telemetry
sampled 14% recent GPU utilization from the just-finished owned workload while
the compute-app inventory was empty.  That complete attempt is preserved as
`direct-cuda128-w21-corrected.log` (SHA-256
`99d018976e709d820f5a5ef8fa07b92dbfa0cdb9a47a672906b4232f2c95a589`)
and excluded.

The reviewed phase-specific policy is implemented in additive
`run-direct-pair-v2.sh` (SHA-256
`f6a23119cae22f36528c1bd4b461f0083bf63458e7020f12ad8bdbbb5046e71d`).
PRE requires exact identity, 0% GPU and memory utilization, an empty active-app
inventory and successful date/Driver checks.  POST requires successful exact-
identity telemetry, an empty active-app inventory and the same Driver identity,
while recording rather than rejecting recent utilization.  Date, command,
application, Driver, hash and benchmark failures remain fail-closed.

The literal `test-run-direct-pair-v2.sh` regression covers quiet PRE/POST
acceptance; GPU14%/memory0% and GPU0%/memory14% as PRE reject/POST accept; wrong
UUID rejection in both phases; and query-failure propagation in both phases.
All ten assertions, `bash -n`, ShellCheck and `git diff --check` pass.  The test
SHA-256 is
`e5d674b34841142ef3e73338bbe919de8a7e0807adf27f251b6bb5b75245aa4e`;
the final transcript SHA-256 is
`7196176ab42ed907d1effde9d3718765ed54fdbdc2b270ad4262db4fcdb73c87`.

The live injected-failure probe has `PRE_NVIDIA_SMI_EXIT=1`,
`PRE_APPS_EXIT=1`, `PRE_TELEMETRY_EXIT=70`, no `BENCHMARK_START`, and SSH exit
70.  Its log is `telemetry-fail-closed-v2-final.log` (SHA-256
`0cca3a8a2348483b3c3731b30d2c3fb977e405c1711eeea408263e28370f7342`).
The subsequent successful live preflight has every telemetry exit zero, exact
UUID/CC/SM, 0% utilization and no applications; its log SHA-256 is
`c2c4f7ecede210ea7d3b3fc35c1a0fc7bbb09a8595b8feba22823cda8b31fde4`.

## Fresh whole-run commands and exits

All SSH/rsync calls used `BatchMode=yes`, `ConnectTimeout=10`,
`ServerAliveInterval=10`, `ServerAliveCountMax=2` and
`ConnectionAttempts=1`.  The four qualifying remote commands were:

```text
/root/mamba-ada-half-direct-pair-20260906/run-direct-pair-v2.sh cuda128 21
/root/mamba-ada-half-direct-pair-20260906/run-direct-pair-v2.sh cuda130 21
/root/mamba-ada-half-direct-pair-20260906/run-direct-pair-v2.sh cuda128 101
/root/mamba-ada-half-direct-pair-20260906/run-direct-pair-v2.sh cuda130 101
```

The wrapper pins the matching CUDA root, release target and private cache,
requires exact CC 8.9, unsets `MAMBA_FIXED_VENDOR_TILES`, selects both rows,
A-E, bias 0/1, eager/graph and invokes the frozen binary directly as:

```text
gemm_bi_fixed_performance-* fixed_ada_half_forced_direct_pair \
  --exact --ignored --nocapture --test-threads=1
```

Every whole run has 80 unique closed-schema records, one passing completion,
zero rejections, and PRE/POST/benchmark/SSH exit zero.  The 21-window runs each
took 26 seconds; the 101-window runs took 93 seconds on CUDA 12.8 and 94 seconds
on CUDA 13.0.  Strict local and independent main verification accepted all four
raw logs, their external identities, paired samplewise quantiles, bits, graphs,
shapes, arguments and exact cohort keys.

| Toolkit/windows | Raw SHA-256 | Verifier SHA-256 | Screen/final swizzle wins |
|---|---|---|---:|
| 12.8/21 | `9416747de3350a712e20602f0d8154eda20970735835f3533f8afecc59939b3f` | `7082f99a3573cb86aa077f6ff427ca2035400df9b08e46f790d7c9f5f5f75607` | 18/20 |
| 13.0/21 | `fa6d7a980bfd9ae89903fe7283dcab125e0ebf7f18d78901cf94f43220c6e32a` | `3f5f060009161531feb6b530526174b1e5e985d1cbb780c70d674f0436e9f9b2` | 18/20 |
| 12.8/101 | `e40cede4f41df7e889315fdb6a14764ee080ace0668f57eba1e608f092048ea5` | `e08f0e6804fd6f8d9dac158ae247017ba78c07e4b05ae420cf9c55608db206bf` | 17/20 |
| 13.0/101 | `ee180dd25038c1ca4cac5e11a3eaf4c2fab06215b8af68b6cc333bea3cd90a41` | `4ad1082ffcda85497adf4a7b410c993237ceba7d22929b4f640d8e3caec28b2e` | 17/20 |

The exact verifier invocation form was:

```text
ruby analysis/verify-half-direct-pair.rb identity-cudaNNN.json \
  WINDOWS direct-cudaNNN-wWINDOWS-v2.log
```

## Final 101-window direct result

Ratios below are worst across eager/graph and both execution orders, computed
samplewise before quantiles.  A direct swizzle winner requires both p50 and p95
strictly below 1.0.  `none` means neither forced arm met that two-quantile rule.

| dtype | cell | bias | CUDA 12.8 swizzle/pipeline p50 / p95 | winner | CUDA 13.0 swizzle/pipeline p50 / p95 | winner |
|---|---|---:|---:|---|---:|---|
| bf16 | hot_a | 0 | 0.999758 / 1.000779 | none | 0.999592 / 1.000613 | none |
| bf16 | hot_a | 1 | 0.995133 / 0.996327 | swizzle | 0.995067 / 0.995961 | swizzle |
| bf16 | hot_b | 0 | 0.828448 / 0.847014 | swizzle | 0.839802 / 0.848647 | swizzle |
| bf16 | hot_b | 1 | 0.835013 / 0.846651 | swizzle | 0.841876 / 0.848272 | swizzle |
| bf16 | hot_c | 0 | 0.992624 / 0.995791 | swizzle | 0.990058 / 0.994301 | swizzle |
| bf16 | hot_c | 1 | 0.984743 / 0.987320 | swizzle | 0.981452 / 0.986546 | swizzle |
| bf16 | hot_d | 0 | 0.866488 / 0.867368 | swizzle | 0.866667 / 0.867544 | swizzle |
| bf16 | hot_d | 1 | 0.865345 / 0.866424 | swizzle | 0.865500 / 0.866358 | swizzle |
| bf16 | hot_e | 0 | 0.936602 / 0.937362 | swizzle | 0.936646 / 0.937118 | swizzle |
| bf16 | hot_e | 1 | 0.938736 / 0.939448 | swizzle | 0.938857 / 0.939434 | swizzle |
| f16 | hot_a | 0 | 1.000981 / 1.005588 | none | 1.002178 / 1.005206 | none |
| f16 | hot_a | 1 | 0.996539 / 1.000275 | none | 0.998616 / 1.001090 | none |
| f16 | hot_b | 0 | 0.850593 / 0.854767 | swizzle | 0.851806 / 0.855933 | swizzle |
| f16 | hot_b | 1 | 0.848921 / 0.854272 | swizzle | 0.849693 / 0.854025 | swizzle |
| f16 | hot_c | 0 | 0.985885 / 0.990819 | swizzle | 0.985522 / 0.993111 | swizzle |
| f16 | hot_c | 1 | 0.985743 / 0.992826 | swizzle | 0.987315 / 0.994145 | swizzle |
| f16 | hot_d | 0 | 0.866539 / 0.867391 | swizzle | 0.866756 / 0.867373 | swizzle |
| f16 | hot_d | 1 | 0.865500 / 0.866580 | swizzle | 0.865544 / 0.866551 | swizzle |
| f16 | hot_e | 0 | 0.971153 / 0.985978 | swizzle | 0.974181 / 0.986884 | swizzle |
| f16 | hot_e | 1 | 0.974324 / 0.987664 | swizzle | 0.975345 / 0.989150 | swizzle |

Thus the swizzle arm is the robust direct winner in the same 17 of 20 cohorts
on both toolkits: all B-E dtype/bias cohorts plus BF16 A1.  BF16 A0, F16 A0 and
F16 A1 have no direct winner.  These are internal pipeline-versus-swizzle
conclusions only.  Vendor gaps remain separate, and the analyzer explicitly
reports `auto_admission_authorized:false`; no AUTO, epoch or production file was
changed in this task.

## Final immutable reconciliation and release

`final-remote-source-reconciliation-v2.log` cleanly verifies all 174 isolated
source inputs, substituting only the reviewed direct-pair test hash into the
frozen Task 1 manifest.  Its SHA-256 is
`63e4b97c6dea2153134538ffb148e40ca9b2ac4fef71dc5f1afd2fa853c04d38`;
all 174 checks and SOURCE/SSH exits are zero.

`final-remote-reconciliation-v2.log` (SHA-256
`7bde5a62b8bfcd8546c53497463cc39617f6825680a2778ecfc7736d795a53bc`)
verifies the exact v2 wrapper and test source, three release binaries, both
external controls, all nine cache blob filenames and hashes, cache modes 0700,
exact UUID/driver/CC8.9/142 SM, 0% GPU/memory utilization and an empty compute-
app inventory.  Every substantive check and SSH exit is zero.  Its first
source transform emitted a single formatting warning for the replaced test
line; that line passed separately in the same log, and the clean companion log
above supplies the authoritative integrated 174/174 proof.

The original partial manifests and archived bytes are unchanged.  The additive
`README.resume.md` and `SHA256SUMS.resume` describe and bind the restored-access
artifacts.  No remote jobs remain, no more remote calls are planned, and the
exclusive Ada build/GPU lane is explicitly released.
