# Ada Triad compact8/S2 integration checkpoint

The previously measured NT compact8/S2 candidate is now qualified on CUDA13.2
and its three measured cells are wired into production AUTO. The post-admission
actual-AUTO runtime check passes on all three; CUDA12.8/13.0 remain unadmitted
until their own runs. This is not a cuBLAS Fast win.

The new route keeps the portable RNA numeric contract and one NT
M128N64/BK32/S2 kernel: 256 threads, 49,152 dynamic shared bytes, no tensor maps.
Its physical tuning revision is 1; the shared tuning revision remains 45,
numeric ABI 5 and schedule revision 8. Existing CUDA sources and portable
cohort literals are unchanged. Optional rejection retains the old routes.

## CUDA13.2 paired qualification — 2026-09-07T20:50Z

The accepted harness completed exactly one test,24 cohorts and504 paired
observations, with21 samples per eager/graph and AB/BA stratum. Root's
`replay-pairs.rb` independently reproduces all ratios, nearest-rank p50/p95,
cell decisions, manifests, binding consistency and completion counts.

| NT logical shape | Case | Current AUTO → finalist, graph AB p50 us | Paired finalist/current p50 across four strata | Paired finalist/Fast p50 |
|---|---|---:|---:|---:|
|(2048,768,3072)|d768-in|274.324 →226.077|.8240–.8242|1.9321–1.9343|
|(2048,1536,768)|d768-out|119.954 →105.791|.8756–.8821|1.5884–1.5946|
|(4621,384,1928)|Prism|192.385 →160.867|.8369–.8375|2.0958–2.1076|

All three pass current-ratio p50 **and** p95 below1 in all four strata.
Fast ratios come from separate directly paired Fast cohorts, not from dividing
unrelated table medians. In particular Prism finalist timing in the Fast cohort
was173.221us versus82.215us; its earlier current cohort was160.867us. Do not
substitute one cohort's numerator into the other. cuBLAS is native
`CUBLAS_COMPUTE_32F_FAST_TF32`/`CUBLAS_GEMM_DEFAULT`, not PEDANTIC or a bit oracle.

Raw: `task3-once21-cuda132/test.log`, SHA
`30041ac542f0128e2ec6f4da793eb42a9d29f93d06d8a24d71006d514ac117bb`.
Binary SHA`02014b3d1a8771abc120fc49c8742e9d32b103c13ce84f123f5c162debd3676b`;
source map`08b5d723232e93a8dc8acc7a252aa7bad2d3c6cde4a3d25c946756e6c62be1b3`.
All377 source rows accounted for:374 local matches, three committed baseline
versions (pre-admission dispatch/cohort and unrelated SM120 WIP).
Strict quiet PRE20:49:13Z, raw nonquiet RELEASE20:50:01Z, separate DRAIN with
six quiet observations. Reused0700 private cache; all four artifacts unchanged.

Actual resource/K0/live-revision gate also passed1/1 in3.97s:
121 registers,0 local,0 static,49152 dynamic shared,256 threads,occupancy2.
K0 normal/exceptional-old-C eager+graph positive-zero checks and actual
finalist revision1 versus portable/shared45 accept/reject checks pass.
Raw `task3-resource-k0-cuda132/test.log` SHA
`c8a2bea018b09dfb0dc6188a0626f7ff447ee26739d91cd43065e8739c04833a`.
Its one quiet drain sample is not the timing run's six-sample quiet window.

The retained failed builds diagnose test-only imports and a JSON-helper type
annotation; neither ran timing. Final2 release build/list passed in4.13s
without warnings (library709 tests, performance63). Review approved the repaired
harness. The optional post-AUTO expected-cell filter must come from the measured
admission manifest, never from observing which cells AUTO happens to choose.

## Actual AUTO closure — CUDA13.2

`task3-auto-cuda132-final2/` passes the release build and all four exact tests:
table-driven admissions/29 identity mutations/neighbors/operands/old-route
declines, the retained unmeasured forced-only holder test, strict expected-cell
parsing, and actual-AUTO versus forced-finalist bits. The GPU test executes
all three expected cells, first asserts the real one-node finalist manifest,
then runs eager twice and graph twice with resets, input and guard checks.
It passed in9.29s; there is no observational skipping of disconnected cells.

Source map`4ea35e39b4bbb54f4317d0e67f0cc1ffbedca8b4a984d1854597c8f235a67796`:
377 rows,376 match local source and the unrelated SM120 WIP uses its committed
baseline. Library binary`cbe3905735c237bb34febd65b90a5d6b41b3397c066a76e99d6ef747e54e9cf5`;
cohort binary`d018f5274e4f5c9b544a5d73295707d7b4cc4764c63ce4a3dd99f3ea28402295`.
All four private cache files stayed unchanged, and the separate drain completed.
Retained attempts record an evidence-directory guard and a cfg(test) namespace
repair; neither launched GPU work. Production identity/cells were unchanged
by that test-import repair. No empty-table RED run is claimed for the new
host admission assertion; its final actual GREEN is recorded here.

## Verified scope

- CUDA 13.2 release build of the library and the cohort-binding, performance
  matrix and selector test targets: exit 0, no compiler warnings/errors.
- Actual test lists: library 708, performance matrix 58, cohort binding 3,
  selector 1. Eight selected CPU-only tests each executed once and passed.
- Final evidence is in `task2-build-cuda132-final2/`. The source manifest SHA
  is `89435a7c60598982131e6fa4be6f0bc62b56ddabc5d14f37212bcb7b82b73425`.
  All 377 remote source rows are accounted for: 376 match local files and one
  matches the committed SM120-test baseline; unrelated local SM120 WIP was
  deliberately not synced. All 13 pinned source/helper hashes match.
- PRE 2026-09-07T19:47:53Z, RELEASE 19:48:45Z. The reused private cache was
  mode 0700 and its original three artifacts stayed byte-identical. This was
  Rust build/list/host testing only: no NVRTC module load or GPU runtime, so
  no fourth cache artifact was expected.
- Root replay: 28 native source tests plus 5 CUDA-cfg structural Rust tests,
  eight frozen reference hashes, rustfmt and diff-check pass. CUDA-cfg native
  tests are not CUDA compilation. The complete generated source still equals
  the measured candidate after the allowed naming/comment normalization:
  plain SHA `d20cb39f390ef45181be94e4bd7baeb51548c0e6efac64bad72aae71a889e3ce`.

## Retained diagnostic attempts

Checkpoint1 failed on two missing Rust enum match arms. Checkpoint2 compiled
and passed the initial three host tests. Checkpoint3-invalid stopped at source
hash preflight, before any build. The first final batch passed 7/8 tests: its
synthetic PTX fixture incorrectly used the TMA/40-byte ABI for this pointer/
32-byte route. Only the fixture was corrected; the production ABI validator
was not relaxed. Final2 passed 8/8. These are integration findings, not
candidate performance losses or proof of full behavioral TDD coverage.

## Earlier forced smoke and remaining toolkit work

First real CUDA13.2 smoke is now complete in `task3-early-bits-cuda132/`:
one exact ignored test passed in58.10s. Seven finite full-mantissa cases cover
the three target shapes, simultaneous tails and M1 prefixes at alpha1/-0.75.
Each arm ran eager twice and graph twice with fresh identical A/B/C; the test
checks input immutability, three allocation guards, repeat/path equality and
finalist-versus-forced-portable-RNA bit equality. NT beta/bias negatives reject
through actual qualification; presizing alone does not reject forced NT beta.
This smoke does not provide new timing or AUTO admission evidence.

The377-row smoke source map is
`d4961b0ee8505fc128fb73f22e5d58f28c302665a70636edf25e0eb03665c9fe`:
374 files match local state at replay, while the incomplete resource/timing
harness and unrelated SM120 WIP used their committed baseline versions.
Binary SHA `6a2689f1c171a4a9ddc01da99492650f53c89259fd7e1d30283abf43ae7f3b38`;
raw test SHA `8ed8db2ab710742a3c9635d7da9976af6db559228f1160ee1a50615f2f50e7ff`.
The real load adds exactly one cache artifact, key
`e61681f002f8ecdd61677fc89a8f0faeac5ce494db503fcc97fd344633ec2d65`,
SHA `1da42f2fbc44dfd9093959831296d86de2414a6d9bb042bc1640730c68d36ac5`;
the original three remain unchanged. PRE20:16:05Z was quiet, immediate
RELEASE20:17:03Z was not quiet, and a separate20:17:04Z DRAIN sample was quiet.
That is one quiet drain observation, not a claimed five-second quiet window.

CUDA12.8/13.0 still require the same scoped build/load, actual Driver resources,
forced portable-RNA bits/repeats/eager/graph/tails/prefixes/K0/input/guards and
paired current/Fast checks before their literal admissions and AUTO postchecks.
CUDA13.2 completed those requirements above, including live stale-revision
rejection. No cross-GPU guarantee or new RTX5090 observation is established.
