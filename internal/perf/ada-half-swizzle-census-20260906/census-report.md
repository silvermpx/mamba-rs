# Ada half-swizzle paired production census report

Date: 2026-09-06  
Task: `ada-half-swizzle-integration-plan.md`, Task 2  
Production checkpoint: `6570ce872aafe77bb94ecc5b6fee27fd48c054f7`

## Result

The required production census completed on the RTX 6000 Ada board for matching
CUDA 12.8, 13.0, and 13.2.  All six whole runs passed: 400 unique records in the
21-window screen and 400 unique records in the 101-window confirmation, with
exactly one successful completion per run and zero rejects, missing keys,
duplicates, or foreign schemas.

The confirmed internal result is:

| Toolkit | Forced candidate | Actual AUTO | Internal wins | Vendor wins | Direct-pair obligation |
|---|---|---|---:|---:|---|
| 12.8 | `Tc128Sm89Pipeline` | `Tc128` | 20/20 | 12/20 | all 20 cells |
| 12.8 | `Tc128Sm89Swizzle` | `Tc128` | 20/20 | 15/20 | all 20 cells |
| 13.0 | `Tc128Sm89Pipeline` | `Tc128` | 20/20 | 12/20 | all 20 cells |
| 13.0 | `Tc128Sm89Swizzle` | `Tc128` | 20/20 | 15/20 | all 20 cells |
| 13.2 | `Tc128Sm89Swizzle` | `Tc128Sm89Pipeline` | 10/20 | 13/20 | none |

For CUDA 12.8/13.0, both candidates beat the incumbent in every cell, but their
separately paired ratios against AUTO cannot rank them.  A same-run direct
pipeline-vs-swizzle comparison is mandatory for all 20 dtype/cell/bias cells.
No fastest-candidate claim is made here.

For CUDA 13.2, confirmed swizzle internal wins are BF16 B/D/E and F16 B/D,
each with bias off and on.  Confirmed internal losses are BF16 A/C, F16 A/C/E,
each with bias off and on.  The preliminary F16 E screen wins did not survive
101-window confirmation and are not promotion evidence.

The complete per-cell confirmation table is
`internal/perf/ada-half-swizzle-census-20260906/confirmation-cell-results.tsv`.
Each row contains the maximum paired p50 and p95 across both eager/graph paths
and both execution orders for the internal and vendor denominators.  A win is
true only when both maxima are below 1.0.  The verifier JSON files preserve the
same values without decimal truncation.

## Confirmed win/loss rosters

Cell notation below is `dtype:cell:bias`, where `b0` is no bias and `b1` is
bias enabled.  Both eager/graph paths and both execution orders are included in
every listed aggregate.

CUDA 12.8 and CUDA 13.0 have identical result rosters:

- Pipeline internal wins: all 20 cells; no internal losses.
- Swizzle internal wins: all 20 cells; no internal losses.
- Pipeline vendor losses: BF16 B0/B1/D0 and F16 B0/B1/D0/E0/E1.  Its other 12
  cells are vendor wins.
- Swizzle vendor losses: BF16 B0/D0 and F16 B0/D0/E0.  Its other 15 cells are
  vendor wins.

CUDA 13.2 swizzle:

- Internal wins: `bf16:B:b0`, `bf16:B:b1`, `bf16:D:b0`, `bf16:D:b1`,
  `bf16:E:b0`, `bf16:E:b1`, `f16:B:b0`, `f16:B:b1`, `f16:D:b0`, `f16:D:b1`.
- Internal losses: `bf16:A:b0`, `bf16:A:b1`, `bf16:C:b0`, `bf16:C:b1`,
  `f16:A:b0`, `f16:A:b1`, `f16:C:b0`, `f16:C:b1`, `f16:E:b0`, `f16:E:b1`.
- Vendor losses: `bf16:B:b0`, `bf16:B:b1`, `bf16:D:b0`, `f16:B:b0`,
  `f16:D:b0`, `f16:E:b0`, `f16:E:b1`.  Its other 13 cells are vendor wins.

Vendor is reported separately and is not an AUTO admission condition.  The
performance denominator is native-half CUBLAS with `CUBLAS_COMPUTE_32F`; the
PEDANTIC path is only the F32 numeric reference.  Bias-enabled vendor records
time the required FP32-to-half bias broadcast and GEMM beta work.  Consequently
vendor comparison is normalized-accuracy/performance evidence, not the custom
FP32-bias-preseed raw-bit oracle.

## Frozen identity and physical proof

No source, test, index, HEAD, production AUTO, or epoch changed.  No build ran.
Before timing and again after the last run:

- all 174 remote source/build/test inputs matched
  `ada-half-swizzle-force-20260906/source-manifest-final.sha256`;
- all 14 release executables and all 9 private cache blobs matched the hashes in
  `ada-half-swizzle-force-20260906/main-live-sha256.log`;
- the worktree remained at Task-1 checkpoint `6570ce872aafe77bb94ecc5b6fee27fd48c054f7`.

| Toolkit | CUDA root | Release performance binary | Binary SHA256 | Private cache |
|---|---|---|---|---|
| 12.8 | `/usr/local/cuda-12.8` | `/root/target-ada-half-swizzle-force-cuda128-20260906/release/deps/gemm_bi_fixed_performance-e01d837be76567b7` | `bc4d9eab239140b25f6bb68e425819d220b6c3179646a32e621ac9656b5906b1` | `/root/mamba-kcache-ada-half-swizzle-force-cuda128-20260906` |
| 13.0 | `/usr/local/cuda-13.0` | `/root/target-ada-half-swizzle-force-cuda130-20260906/release/deps/gemm_bi_fixed_performance-28fdf8ce2728ddd5` | `feacff36d23bbff9c2b73f2c9960ae571115033dec121e5888478178233b3fc6` | `/root/mamba-kcache-ada-half-swizzle-force-cuda130-20260906` |
| 13.2 | `/usr/local/cuda-13.2` | `/root/target-ada-half-swizzle-force-cuda132-20260906/release/deps/gemm_bi_fixed_performance-429f558dd590762a` | `68731c890219513407c9cd811f2cf3e5459849b407a08af6ae19d4a23b7264f4` | `/root/mamba-kcache-ada-half-swizzle-force-cuda132-20260906` |

The external controls are the committed
`ada-half-swizzle-force-20260906/identity-cuda{128,130,132}.json`, created before
this census.  The verifier matched every record and completion against its
toolkit control.  Every record binds CC 8.9, 142 SMs, known NVRTC library,
compiler target, Fixed source/invocation/artifact/header/library-domain digests,
and tuning epoch 41.  Each completion binds its emitted identity subset; the V2
completion schema omits `tuning_table_revision`, and the verifier intentionally
excludes that field for completion matching.  Nothing was inferred from timing
output.

For every record it also proved the selected physical graph: one kernel node,
no non-kernel nodes for custom arms, exact symbol, flat grid
`(ceil(M/128)*ceil(N/128),1,1)`, block `(256,1,1)`, and shared memory 71,680
bytes for `Tc128`/pipeline or 69,632 bytes for swizzle.  Raw storage, AUTO,
repeat, and graph-replay bit checks passed.  All normalized errors were finite
and within 0.01 BF16 / 0.0025 F16 tolerance.

## Commands and environments

Each raw log begins with its fully expanded command.  The common invocation was:

```text
env CUDA_HOME=/usr/local/cuda-X.Y \
    CUDA_PATH=/usr/local/cuda-X.Y \
    LD_LIBRARY_PATH=/usr/local/cuda-X.Y/lib64 \
    PATH=/usr/local/cuda-X.Y/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_TARGET_DIR=/root/target-ada-half-swizzle-force-cudaXYZ-20260906 \
    MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-half-swizzle-force-cudaXYZ-20260906 \
    MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9 \
    MAMBA_FIXED_ADA_ROWS=bf16,f16 \
    MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e \
    MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_VENDOR_PATHS=eager,graph \
    MAMBA_FIXED_ADA_WINDOWS=<21-or-101> \
    MAMBA_FIXED_VENDOR_TILES=<toolkit-candidates> \
    <frozen-release-performance-binary> \
    fixed_ada_forced_rungs_paired_precision_cublas \
    --ignored --exact --nocapture --test-threads=1
```

Candidate tiles were `Tc128Sm89Swizzle` on CUDA 13.2 and
`Tc128Sm89Pipeline,Tc128Sm89Swizzle` on CUDA 12.8/13.0.  Commands ran only via
`ssh -o BatchMode=yes -o ConnectTimeout=10 ada`; no Mac Cargo was invoked.

The verifier command for each whole run was:

```text
ruby .superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/verify-half-census.rb \
  internal/perf/ada-half-swizzle-force-20260906/identity-cudaXYZ.json \
  <21-or-101> internal/perf/ada-half-swizzle-census-20260906/<run>.log
```

Exit and duration summary:

| Stage | CUDA 12.8 | CUDA 13.0 | CUDA 13.2 |
|---|---:|---:|---:|
| 21-window screen | exit 0, 64.58 s, 160 records | exit 0, 63.14 s, 160 records | exit 0, 34.53 s, 80 records |
| 101-window confirmation | exit 0, 266.67 s, 160 records | exit 0, 265.42 s, 160 records | exit 0, 138.24 s, 80 records |

## Quiet-lane evidence and archive

Every immediate pre-run snapshot reported GPU UUID
`GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, CC 8.9, 142 SMs, 0% GPU and
memory utilization, P5, and no active compute application.  The harness's own
launch preflight also passed at 0% utilization with only its excluded 90 MiB
allocation.  Immediate post-run snapshots reported 0% utilization and no
compute application; residual P0 clocks, power, and temperatures up to 77 C are
preserved rather than misreported as foreign load.  No process matching,
signaling, or workload termination occurred.

The first exploratory telemetry query included unsupported `nvidia-smi` field
`multiprocessor_count`; that diagnostic is preserved in
`preflight-identities-telemetry.log`.  No timing had started.  The corrected
`preflight-telemetry-corrected.log` uses supported `nvidia-smi` fields plus a
read-only CUDA Driver attribute query and proves CC 8.9/142 SMs.

Raw logs, worker verification JSON, main's independent verifier/results,
per-cell TSV, identity transcripts, README, and hashes are under
`internal/perf/ada-half-swizzle-census-20260906/`.

## Boundary and lane release

This evidence authorizes no AUTO admission, epoch change, vendor-victory
requirement, Triad inference, global FAST claim, or candidate ranking on CUDA
12.8/13.0.  The required next step there is the separately bounded same-run
direct-pair qualification.

All measurement and postflight jobs finished, the final GPU state was idle, and
the exclusive Ada build/GPU lane was explicitly released to main on 2026-09-06
at 22:13 Europe/Berlin.  There are no active worker jobs and no further remote
commands are planned for this task.
