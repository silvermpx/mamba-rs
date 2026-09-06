# Ada RNA-wide CUDA 12.8 / 13.0 toolkit census report

Date: 2026-09-06

Frozen branch: `codex/gemm-bi-triad-sm80`

Frozen commit: `208c740c3f24c71f8bc090dbf597f7ee1a129fbd`

## Result

The existing forced production `FixedTile::Tf32RnaM128N128S3` route is
compatible with both CUDA 12.8 and CUDA 13.0 on the RTX 6000 Ada host. Both
matching builds succeeded, both cold/warm forced correctness pairs passed
exactly two tests with zero failures, and both 21- and 101-window timing runs
completed with 40 unique records, zero rejections, a known matching NVRTC
library domain, and the actual RNA physical symbol in every record.

At 101 windows, forced RNA is faster than production AUTO in all ten
shape/bias cells at the worst p50 and worst p95 across eager/graph and both
launch orders. It is faster than explicit cuBLAS FAST TF32 only for hot A with
bias. This is a compatibility and performance census, not an AUTO promotion.

No production, test, harness, dispatch, admission, revision, hash, or AUTO
source was changed. No commit or push was made.

## Scope and hard limit

The force correctness wrapper exercised the tail and hot-A cases, all five
incumbent Fixed TF32 rungs, finite and exceptional inputs, prefix boundaries,
aligned inputs and C4 views, eager repeats, graph replay, unsafe-input
rejection, and K=0. Its graph assertions check the exact RNA symbol, five
Driver parameters with offsets/sizes `(0,8),(8,8),(16,8),(24,8),(32,32)`, no
sixth parameter, 256 threads, and 98,304 bytes dynamic shared memory.

It did **not** run the full hot B-E exceptional/prefix/view matrix because that
coverage currently exists only behind actual-AUTO behavior. Therefore this
census does not close full B-E qualification, does not authorize widening the
CUDA 13.2-only AUTO gate, and does not claim that all supported CUDA toolkits
are fully qualified. Test-only parameterization plus observed RED/GREEN is a
remaining gate.

## Provenance and isolation

- The worktree was clean before acquisition and remained on the frozen commit.
- Remote host: `ada-6000`.
- GPU: NVIDIA RTX 6000 Ada Generation, CC 8.9, 142 SMs,
  UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.
- Driver: `595.45.04`.
- The global `/usr/local/cuda` symlink remained `/usr/local/cuda-13.2`; no
  global CUDA, driver, clock, or power setting was changed.
- Power limit was 300 W, equal to the default. Application clocks are reported
  deprecated by the driver. Every timed-run preflight reported 0% GPU and
  memory utilization, 90 MiB used, P5, and 1800 MHz idle SM clock. Temperatures
  were 42-56 C in the harness preflights.
- Remote source: `/root/mamba-ada-rna-toolkit-census-20260906`.
- CUDA 12.8 target/cache:
  `/root/target-ada-rna-cuda128-20260906` and
  `/root/mamba-kcache-ada-rna-cuda128-20260906`.
- CUDA 13.0 target/cache:
  `/root/target-ada-rna-cuda130-20260906` and
  `/root/mamba-kcache-ada-rna-cuda130-20260906`.
- Both new caches were root-owned mode 0700. Each went from zero files before
  cold correctness to three files, then remained at three on warm reuse.
- The frozen `/root/mamba-ada-rna-wide-auto-20260906` and
  `/root/target-ada-rna-wide-auto-20260906` were preserved.

The final build-input closure is 167 files: Cargo metadata, `.cargo` config,
the complete Rust/CUDA production sources, both named test targets, and the
single direct `#[path]` support module. Local and remote per-file SHA-256 lists
match exactly. The sorted manifest SHA-256 is
`d6d3467bd6a5d86f452433f158c728f6b48dd616650a4d38a6fcfe0e69de5b00`.
No old target, performance, kernel-cache, handoff, or internal-experiment data
was transferred as a build input.

The first CUDA 12.8 build attempt exited 101 because the initially minimal
sync omitted the directly imported
`tests/support/fixed_sm89_exact_n64_admission.rs`. The exact error is retained
in `cuda-12.8/build-no-run.log`. The dependency closure was audited, only that
required file was added, all hashes were rechecked, and the identical build
command then passed in `build-no-run-attempt2.log`. This was an acquisition
error, not a CUDA compiler or kernel failure; no gate was weakened.

## Matching builds and toolkit identity

| Toolkit | Feature | NVCC | runtime NVRTC | correctness executable SHA-256 | performance executable SHA-256 |
|---|---|---|---|---|---|
| 12.8 | `cuda,cudarc/cuda-12080` | 12.8.93 | `[12,8]`, known | `eb47dcdc1a38f6bb0faa81e4cd07650fe7791c9583000179c74d22e9b200b225` | `4c293e504d71242089adc382241254ed11df98b31a56d9ce24b4d4e71a10d87e` |
| 13.0 | `cuda,cudarc/cuda-13000` | 13.0.88 | `[13,0]`, known | `f5bfc7390d0f2632c79e573eaffdb7350391ef383db0fdb7db595b650216587f` | `fff28eb3aef0ba4375ac0583cde0927af734410623f51904941fc40bddb7413b` |

The real CUDA 12.8 NVRTC library
`libnvrtc.so.12.8.93` hashes to
`2bb82d1a34b9fefa46aca357299aed66763d4d6613d015d5a591224c00fa7e5a`;
its builtins library hashes to
`eccaa824230ee7858a94a3055cb01f1cb634df05d313880d0bdcf195161fcb4e`.
The real CUDA 13.0 NVRTC library
`libnvrtc.so.13.0.88` hashes to
`a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11`;
its builtins library hashes to
`91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2`.

The runtime records additionally bind these production identities:

| Identity | CUDA 12.8 | CUDA 13.0 |
|---|---|---|
| Fixed source digest | `7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301` | same |
| Fixed invocation digest | `ae8e2e3db0255db26419c8292c70ea945f46076770ddb3522d34443addeaf374` | `4d6815a9cdc06297113b72dc2d9ae9fac46b50a0583c561f358727d4a6f2cf49` |
| Fixed artifact digest | `c71288517eb76b839ce23b2f915b010ca73b817b08d5aa0eb2e88a7b2c9f2a3e` | `c90cd431d3c2df95e849b99f6fcef8e6a7e28f64d97e851ec39f2445a7dc5822` |
| Header manifest digest | `9924f331b7c7e70041f74e8a9b39d072930c493c921ddf19beb34b6263682fc8` | `7801fef3bdeb57597ff2028997aa9d6190fe63dec216b8bf0d1d5a07235f8685` |
| NVRTC library domain | `26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155` | `709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d` |

The source digest intentionally matches while invocation, header, library,
and artifact identities differ by toolkit. No CUDA 13 binary was run against
NVRTC 12.8.

## Build and correctness results

The required release build command succeeded for each matching feature. Only
`gemm_bi_fixed_correctness` and `gemm_bi_fixed_performance` were built.

| Toolkit | Build | Cold forced correctness | Warm forced correctness |
|---|---:|---:|---:|
| 12.8 | pass, exit 0 | 2 passed / 0 failed, 100.32 s | 2 passed / 0 failed, 8.98 s |
| 13.0 | pass, exit 0 | 2 passed / 0 failed, 97.51 s | 2 passed / 0 failed, 9.25 s |

The exact selected tests were
`fixed_tf32_rna_wide_matches_all_fixed_rungs_prefix_views_and_graph_bits` and
`fixed_tf32_rna_wide_rejects_unsafe_inputs_and_handles_k0`. Five unrelated
tests were filtered, and neither selected test was skipped or ignored after
selection.

## Timing validation

For each toolkit, both the 21-window screen and 101-window repeat produced:

- 40 records = 5 shapes x 2 bias states x 2 paths x 2 orders;
- 40 unique `(cell,bias,path,order)` cohorts and all ten shape/bias cells;
- zero rejection records;
- one completion record with `records=40`, `rejected=0`, `passed=true`;
- matching runtime NVRTC and `nvrtc_library_known=true`;
- forced tile `Tf32RnaM128N128S3` and physical symbol
  `gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3`;
- one forced graph kernel, zero non-kernel nodes, block `[256,1,1]`, and
  98,304 bytes shared memory in every record;
- all raw storage, AUTO identity, forced repeat, and vendor repeat flags true;
- graph replay true on graph records; eager `graph_replay_bits_equal=false`
  was checked as not applicable;
- explicit `CUBLAS_COMPUTE_32F_FAST_TF32` vendor comparison and independent
  `CUBLAS_COMPUTE_32F_PEDANTIC` numerical reference;
- required bias epilogue timed exactly when bias was present.

The 21-window screens took 21.35 s (12.8) and 21.38 s (13.0). Since every
shape/bias cell had an eligible forced-over-AUTO win, the required 101-window
repeats were run with the same frozen source, per-toolkit executable, target,
and private cache. They took 71.28 s and 71.82 s respectively.

### 101-window paired ratios

Each entry is `forced / denominator`; values below 1 favor RNA. Quantiles are
samplewise ratios using nearest index `round((n-1)*q)`. The table reports the
worst value across the four eager/graph x launch-order cohorts for each
shape/bias cell. FAST includes the timed bias broadcast when bias is present;
PEDANTIC is only the numerical reference.

| Toolkit | Cell | Bias | RNA/AUTO p50 | RNA/AUTO p95 | RNA/FAST p50 | RNA/FAST p95 |
|---|---|---:|---:|---:|---:|---:|
| 12.8 | hot A | no | 0.8833 | 0.9028 | 1.1638 | 1.1804 |
| 12.8 | hot A | yes | 0.8918 | 0.9127 | 0.9355 | 0.9555 |
| 12.8 | hot B | no | 0.7847 | 0.7953 | 1.1957 | 1.2167 |
| 12.8 | hot B | yes | 0.7884 | 0.7948 | 1.0485 | 1.0630 |
| 12.8 | hot C | no | 0.6469 | 0.6479 | 1.4383 | 1.4388 |
| 12.8 | hot C | yes | 0.6484 | 0.6488 | 1.3170 | 1.3175 |
| 12.8 | hot D | no | 0.9662 | 0.9844 | 1.3717 | 1.3975 |
| 12.8 | hot D | yes | 0.9693 | 0.9717 | 1.1895 | 1.2018 |
| 12.8 | hot E | no | 0.9470 | 0.9690 | 1.4371 | 1.4969 |
| 12.8 | hot E | yes | 0.9496 | 0.9717 | 1.3464 | 1.3977 |
| 13.0 | hot A | no | 0.8836 | 0.9070 | 1.1648 | 1.1946 |
| 13.0 | hot A | yes | 0.8923 | 0.9197 | 0.9425 | 0.9730 |
| 13.0 | hot B | no | 0.7846 | 0.7914 | 1.1966 | 1.2186 |
| 13.0 | hot B | yes | 0.7884 | 0.7954 | 1.0529 | 1.0639 |
| 13.0 | hot C | no | 0.6472 | 0.6568 | 1.4383 | 1.4588 |
| 13.0 | hot C | yes | 0.6483 | 0.6498 | 1.3170 | 1.3188 |
| 13.0 | hot D | no | 0.9663 | 0.9701 | 1.3717 | 1.3833 |
| 13.0 | hot D | yes | 0.9693 | 0.9774 | 1.1930 | 1.2228 |
| 13.0 | hot E | no | 0.9470 | 0.9646 | 1.4374 | 1.4988 |
| 13.0 | hot E | yes | 0.9497 | 0.9708 | 1.3473 | 1.3964 |

The toolkit results are closely aligned. The candidate's AUTO advantage is
largest on hot C and B and narrowest on hot D/E. Against FAST, only hot A with
bias is a complete worst-p95 win; hot B with bias is close but remains slower,
and the other cells clearly favor FAST.

## Evidence

Evidence is under
`internal/perf/ada-rna-toolkit-census-20260906/`. Important files are:

- `source-manifest.sha256` and `provenance.log`;
- per-toolkit `build-no-run*.log`, `binary-identities-final.log`,
  `toolkit-library-identities.log`, `correctness-cold.log`, and
  `correctness-warm.log`;
- per-toolkit `timing21-preflight.log`, `timing21.log`, and
  `timing21-validation.json`;
- per-toolkit `timing101-preflight.log`, `timing101.log`, and
  `timing101-validation.json`.

The raw 101-window logs hash to:

- CUDA 12.8:
  `955f252909c94607e359e2a2eeb47e9744f76c41a8f2fac128ef338a80e15dcb`;
- CUDA 13.0:
  `d687caefea0b6c85a378f99230b4ee7f44c55a40e7ac6bfe1980f9967d3b0a37`.

## Remaining gates

1. Decouple the full forced correctness corpus from the13.2 actual-AUTO
   assertions and observe the B-E exceptional/prefix/view qualification on
   all three toolkits. Add an observed corpus-regression RED/GREEN without
   changing production behavior; do not run13.2 AUTO expectations on12.8/13.0.
2. Review whether the CUDA 12.8/13.0 compiler, library, artifact, and device
   identities should become explicit admission cohorts.
3. Only after full qualification, separately decide whether to widen AUTO and
   update dispatch/graph policy revision and frozen evidence.
4. Preserve the current CUDA 13.2 AUTO qualification as a separate cohort;
   this census neither replaces nor mutates it.
