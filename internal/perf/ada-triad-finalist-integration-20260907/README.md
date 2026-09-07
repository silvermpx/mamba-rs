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

The frozen Task3 batch must compile/load on CUDA 12.8, 13.0 and 13.2, record
actual Driver resources, pass forced portable-RNA bit equality, repeats,
eager/graph, tails/prefixes, K0 and input/guard checks, and compare paired
candidate/current and candidate/cuBLAS Fast timing. Only passing literal
toolkit/cell identities may then enter AUTO. Current checkpoint does not
establish live-context stale-revision rejection or a cross-GPU guarantee.
