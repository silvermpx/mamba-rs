# Ada Triad compact8/S2 integration checkpoint

This checkpoint wires the previously measured NT compact8/S2 candidate as an
optional, forced-only `TriadSm89Finalist` module. AUTO admission is still empty.
It is not a new speed measurement or a cuBLAS Fast win.

The new route keeps the portable RNA numeric contract and one NT
M128N64/BK32/S2 kernel: 256 threads, 49,152 dynamic shared bytes, no tensor maps.
Its physical tuning revision is 1; the shared tuning revision remains 45,
numeric ABI 5 and schedule revision 8. Existing CUDA sources and portable
cohort literals are unchanged. Optional rejection retains the old routes.

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

## Still required before AUTO admission

The frozen Task3 batch must compile/load on CUDA 12.8, 13.0 and 13.2, record
actual Driver resources, pass forced portable-RNA bit equality, repeats,
eager/graph, tails/prefixes, K0 and input/guard checks, and compare paired
candidate/current and candidate/cuBLAS Fast timing. Only passing literal
toolkit/cell identities may then enter AUTO. Current checkpoint does not
establish live-context stale-revision rejection or a cross-GPU guarantee.
