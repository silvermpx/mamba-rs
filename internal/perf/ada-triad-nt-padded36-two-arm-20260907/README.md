# Ada Triad NT: padded36 two-arm discovery, 2026-09-07

Both test-only candidates are **STOP_NO_RETRY**. Neither is promoted to
production AUTO. This is a bounded candidate screen, not cuBLAS qualification.

## Measured result

One CUDA13.2 / RTX6000 Ada CC8.9 binary ran the two exact tests sequentially.
Each test executed once and passed resources, finite full-mantissa target/tail
bit checks, eager/graph repeats, input/guard checks and post-timing graph replay.

Ratios below are candidate / actual production AUTO, lower is better.
The retention rule was fixed before measurement: p50 and p95 below0.99 in
each of eager ABBA, eager BAAB, graph ABBA and graph BAAB.

- Precomputed copy plan:185 registers,0 local,0 static shared,82,944 dynamic
  shared bytes,1 resident CTA. Ratios p50/p95:
  eager ABBA1.094053331/1.094196726;
  eager BAAB1.093788837/1.094325914;
  graph ABBA1.095567654/1.095749274;
  graph BAAB1.095482243/1.096079088.
  All strata lose. Median times approximately300.6–300.7us eager and299.9us
  graph versus274.8–274.9us and273.7us for AUTO.
- Padded ldmatrix loads:160 registers,0 local,0 static shared,82,944 dynamic
  shared bytes,1 resident CTA. Ratios p50/p95:
  eager ABBA0.999331027/0.999987081;
  eager BAAB0.999443687/1.000177957;
  graph ABBA1.001662696/1.002078802;
  graph BAAB1.001887241/1.002072281.
  Eager is effectively parity; graph is slightly slower. No retention.

The production profile recorded154 registers. The register increases above
are observations, not proof of the sole performance cause. No cuBLAS run,
21/101-window admission, new profile, whole matrix rerun or losing-arm retry
was performed. The existing Ada60 baseline remains unchanged.

## Numerical and experiment scope

Logical target M/K/N=(2048,768,3072), NT output2048x768, reduction3072.
Reference is the actual public AUTO route:
gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3, grid192, block256,
82,944 dynamic shared bytes. Forced same-tile tail is(129,65,36), not tail AUTO.
Alpha1, beta0, no bias. Both candidates start independently from the original
padded36 source; neither incorporates the rejected compact-XOR layout.

Copy-plan caches per-thread source row bases, validity and shared destination
bases, preserving six vector assignments and the original prefill/wait/refill
schedule. Ldmatrix replaces only target NT fragment reads, preserving RNA
conversion and ascending K8 MMA association. Its non-transposed x4/x2 mapping
was checked against the [NVIDIA PTX ldmatrix specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix)
and [m16n8k8 TF32 fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k8).

The finite bits screen does not establish NaN/Inf, every stride/subview, K0,
all epilogues, all architectures/toolkits or the complete prefix contract.
Graph checks cover symbol, grid/block/shared and five-argument ABI offsets/sizes,
not an independent decode of every captured argument value. No numerical ABI,
tuning revision, dispatcher or production CUDA source changes are made here.

## Evidence and corrected setup

- Original Rust build/list succeeds at16:24:28–16:25:08 UTC. Its e92648d...
  source retained a stale signature-assertion symbol. This was found before
  either GPU test; this binary has no GPU measurement. Exact old source is
  archived in build-cuda132/gemm_bi_tf32_nt_compact_xor.pre-repair.rs.
- Narrow repair renames both matching signature assertions and adds regression
  checks. Genuine host RED: both boundary tests fail with the missing new
  assertions; GREEN: native10/10. Root's extra actual-generated-source check
  verifies all18 exports equal the18 assertion targets for both candidates.
- Repaired source SHA37dec61a49190a8bb5c98cc59d57dfb0746a6e58f5b46febe4893fb18d0ee975.
  Repaired binary SHA34a67271406e49a62a1baba8765a640bb26b2c0777b339f4545ece40cd25713e.
  Full372-input manifest SHAdb00c029ed47bd6a41a3a3a096f0a101c41d4128dcbb8d2773ae3b8982ba22dc.
- Copy-plan actual test44.43s, raw SHA84d8b5a3b5b01a1e596b534f49928983180b9cc85553ce4f019fa7566fe6f618.
  Ldmatrix raw SHA448835ee03f5d89a3922fdec75c0c998474f8f82bfca8f7b4152502de3e1a9da.
- First build wrapper had a Python boolean typo before PRE/cargo/GPU, after
  copying the three production cache files. Its invalid log is preserved.
  The corrected runner verifies rather than overwrites those exact destinations.
- Cache is private0700, preseeded with three SHA-verified unchanged production
  artifacts from the compact1 cache. This is intentionally NOT cold-cache data.
  Old cache was not mutated. Candidates are compiled separately by NVRTC.
- Raw immediate busy RELEASE samples are retained. Distinct five-second drains
  are quiet, as are the internal pre/timed/post gates. No process was killed
  and no clocks were changed. Candidate PTX was not persisted; do not treat
  these negative screens as artifact-complete production qualification.

## Local replay and next bounded step

From this checkpoint's worktree revision:

    ruby internal/perf/ada-triad-nt-padded36-two-arm-20260907/replay.rb
    ruby internal/perf/ada-triad-nt-padded36-two-arm-20260907/host-copy-plan-replay.rb
    ruby internal/perf/ada-triad-nt-padded36-two-arm-20260907/host-ldmatrix-replay.rb
    ruby internal/perf/ada-triad-nt-padded36-two-arm-20260907/host-generated-source-replay.rb

Replay verifies both372-input source maps (with the explicit archived old
source override), binary/source/raw bindings, two actual exact tests, all56
raw four-leg brackets, eight paired strata and decisions. Native scripts use
macOS clang/rustc without CUDA. The copy-plan address-only shim exercises the
actual header prefix for2,764,800 tuples and catches A/B stage-stride and
tail-byte mutations; it does not emulate CUDA execution. The ldmatrix proof
uses the actual coordinate helpers and catches both wrong-K-half mutations.

Next independent hypothesis: keep original padded36/M128N64/BK32/S3 but use
all eight warps for compute, MAtoms2 instead of4, preserving per-output K order.
This changes warp ownership rather than the failed copy-plan/XOR mechanisms.
It needs its own coverage proof, resource/bits screen and one bounded timing
attempt. It is not implemented or measured by this checkpoint.
