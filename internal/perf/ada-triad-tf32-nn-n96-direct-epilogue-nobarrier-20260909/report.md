# Ada TF32 NN N96 direct-epilogue no-barrier screen — 2026-09-09

## Decision

Valid stop. Removing the final CTA barrier from the full-tile direct-`float2`
epilogue is bit-exact, but it does not clear the strict `<0.99` retained-best
gate on Prism. Keep the preceding direct-epilogue N96 candidate and do not
retry this no-barrier variant unchanged. cuBLAS Fast was intentionally not
screened after the retained gate failed.

This is a test-only discovery result. No production route or dispatcher was
changed.

## Frozen source

- Harness SHA-256:
  `4c8d98c330544f27064544b54877917be4ab184b10ac47f63887905b7aea7454`
- Source adapter SHA-256:
  `a1878809146e065fadd19401eb9a9e47aae0a20840a16568b66de4ef5face83c`
- Candidate change: skip only the final CTA barrier for the full-tile direct
  `float2` epilogue.

## Correctness and resources

- Full tile, tail, exceptional payload, K0, input/guard checks: PASS.
- Eager and captured-graph exact comparison with the measured direct-epilogue
  N96 candidate: PASS.
- Candidate and retained: 124 registers/thread, local0, 86,016 B dynamic
  shared memory, occupancy1, 256 threads.

## Paired once7 timing

Ratios are candidate / measured direct-epilogue N96 on Prism
`(4621, 384, 1928)`, 20 logical GEMMs per observation.

| Path / order | p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 0.998005 | 0.999885 |
| eager / BAAB | 0.999625 | 1.000053 |
| graph / ABBA | 0.996140 | 0.997803 |
| graph / BAAB | 0.997785 | 1.000023 |

The apparent gain is only 0.0–0.4%, and two p95 strata cross 1.0. It is not a
robust retained-best improvement.

## GPU isolation

The user-owned models stayed resident. Every phase used the idle-resident
preflight: five consecutive samples at no more than 1% compute and memory
utilization plus an explicit free-VRAM floor. Pre/post utilization was 0%/0%;
free memory was 2669 MiB before setup and 2056 MiB after the test.

Raw output: [raw.log](raw.log).
