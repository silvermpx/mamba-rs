# Ada half-TN five-warp producer/consumer screen — 2026-09-09

## Decision

Valid stop. The SM89 TC64/BK64/S2 five-warp producer/consumer candidate is
bit-exact on F16 d768-in and satisfies its hard resource contract, but loses
the retained regpipe+`float2` candidate by 15.4–17.3% and native-half cuBLAS
Fast by 12.0–16.8% across paired eager/graph strata. Do not run BF16 or sibling
shapes and do not retry the mechanism unchanged.

This is test-only discovery. Production routes and the dispatcher were not
changed.

## Mechanism and compile proof

Physical warp0 performs the compact XOR-layout 16-byte `cp.async` copies into
the existing two 32 KiB shared buffers. Physical warps1–4 retain the old
logical warp0–3 `ldmatrix`, register-pipeline MMA order, accumulator ownership
and `float2` epilogue. Four shared `mbarrier` objects coordinate two ready and
two filled phases.

The first compile-only implementation exposed an Ada-specific API trap:
CUDA13.2 `__mbarrier_try_wait` is SM90+, so SM89 lowered it to a trap and then
deleted the producer body. Replacing it with the SM80+ supported
`__mbarrier_test_wait` produced the intended PTX. The admitted PTX contains:

- `cp.async.cg.shared.global`: 16 occurrences across BF16+F16 entries;
- `cp.async.mbarrier.arrive.shared.b64`: 2;
- `mbarrier.test_wait.shared.b64`: 4;
- A/B `ldmatrix`: 16/32;
- `mma.sync`: 64;
- `trap`: 0.

Frozen artifacts:

- Rust harness SHA-256:
  `568dccc24144e5826b785684fb6269a91dc75fe848af48fe4cf6854ecdac0d0b`
- Source adapter SHA-256:
  `52d7752c227feef3a70e2697c5f36dc020a0890e5c431c41a4a65e5f41f4bd02`
- Composed CUDA source SHA-256:
  `780c37b162e8986f6ff568f70d5a13c4754e168dccf6f74932daaad2fd1a919c`
- NVRTC PTX SHA-256 before evidence newline normalization:
  `495653a7629ed3244cd228afb61d07598dc4d275f465e87e244d8b4572ef7fc5`
- Archived PTX SHA-256:
  `d7e51de6c27afe0d3a2af94c430e6a88c31b2472c2676fdd9d5cb6f4455b46a6`

## Correctness and resources

- Candidate, retained regpipe+`float2`, and current TC64 F16 d768-in outputs
  are bit-identical for two eager and two captured-graph repeats.
- Input and output guards remain unchanged; all timed pointers are 256-byte
  aligned.
- Candidate: 160 threads, 92 registers/thread, local0, 32,800 B static shared,
  dynamic0, max threads160, occupancy3. Hard gates were <=128 registers,
  local0 and occupancy>=3.
- Retained: 128 threads, 125 registers/thread, local0, 32,768 B static shared,
  occupancy3.

## Paired once7 timing

Ratios are candidate / comparator on F16 d768-in `(2048, 768, 3072)`, 20
logical GEMMs per observation.

| Path / order | vs retained p50 | vs retained p95 | vs Fast p50 | vs Fast p95 |
| --- | ---: | ---: | ---: | ---: |
| eager / ABBA | 1.164768 | 1.167748 | 1.122459 | 1.129636 |
| eager / BAAB | 1.164154 | 1.173150 | 1.119561 | 1.130413 |
| graph / ABBA | 1.158430 | 1.165108 | 1.160248 | 1.167656 |
| graph / BAAB | 1.154350 | 1.167458 | 1.160549 | 1.164541 |

The dedicated producer reduces register pressure but costs more in barrier
traffic and/or reduced useful-warp issue than it recovers from overlap on this
cell. The measured mechanism cannot close the remaining half-TN Fast gap.

## GPU preflight

The immediate external preflight and the in-harness five-sample gate both
reported 0% compute and 0% memory utilization. The card had 48,463 MiB free
before the run. The post gate again reported 0%/0%; the test context used
488 MiB. No external process was stopped or unloaded.

Evidence: [raw.log](raw.log), [candidate.cu](evidence/candidate.cu),
[candidate.ptx](evidence/candidate.ptx).
