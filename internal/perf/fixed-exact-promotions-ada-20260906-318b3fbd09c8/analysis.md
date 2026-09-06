# Ada Fixed exact promotion qualification

Frozen benchmark source: commit `318b3fbd09c8fe0bfac7eaabefdec2bba457f212`.

## A/D — `F32Sm89N64CopyPlan`, 101 windows

- Harness PASS: 32 records, 0 rejected; exact custom tolerance, repeat bits, raw-storage identity, forced symbol, and eager/graph replay identity all passed.
- Strict promotion gate PASS: all 32 balanced cohorts beat incumbent AUTO at both paired p50 and paired p95.
- Forced/AUTO p50 range: 0.830110–0.846557. Forced/AUTO p95 range: 0.832349–0.854944.
- Against cuBLAS PEDANTIC: 16/16 p50 and p95 wins; p50 range 0.793988–0.906183, worst p95 0.914202.
- Against cuBLAS FAST_TF32: 0/16 wins; p50 range 1.489599–2.064311, worst p95 2.118521.
- Log SHA-256: `9224161eec3a578f472c07810647952c56604a980d699876b187887a929c575a`.

This is a strong internal and PEDANTIC promotion, but it does not satisfy the final exact-vs-FAST objective.

## C — `F32N128S2`, 21-window screen

- Harness PASS: 16 records, 0 rejected; all numeric and graph identity gates passed.
- Strict promotion gate FAIL: all 16 paired p50 ratios win (0.955184–0.961124), but only 14/16 p95 ratios win. The failures are exact/no-bias/auto-first eager (p95 1.005504) and graph (p95 1.022126).
- Against cuBLAS PEDANTIC: 0/8 wins; p50 range 1.400257–1.445304, worst p95 1.477921.
- Against cuBLAS FAST_TF32: 0/8 wins; p50 range 3.158872–3.471219, worst p95 3.502356.
- Per the predeclared gate, the 101-window confirmation was skipped.
- Log SHA-256: `9318f4cca804d6585f61438da07abe4e0eed3a930b1fc27c8bedc5911522865d`.

The C candidate is a modest median internal improvement but neither p95-qualified nor vendor-competitive in this run.
