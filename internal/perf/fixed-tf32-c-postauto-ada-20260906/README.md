# Ada Fixed TF32 C0/C1: actual AUTO qualification

2026-09-06, base `6e7ed213` plus the two source files hashed in `preflight.log`.
Isolated source `/root/mamba-ada-dispatch-review-gxJWgz`, target
`/root/target-ada-dispatch-review-gxJWgz`. RTX6000 Ada, CC8.9,142SM,
NVRTC13.2, GPU UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.

Only the selector changes: exact `(M,K,N)=(4621,1928,384)`, bothbias, known
NVRTC13.2 on this142-SM Ada cohort. AUTO changes from M128S2 to M64S2. Thin,
neighboring dimensions, othertoolkits/devices and otherprecisions retain
their prior routes. No CUDA source, module composition or numerical ABI change.

## Results

- `c-postauto-old-m128-control101.log`:8/8records,0rejected,completion PASS;
  bias0/1 × eager/graph × auto-first/vendor-first,101windows each.
- Every AUTO graph contains the actual
  `gemm_bi_nn_tf32_v1_m64n64_bk32_s2`; the forced control is prior M128S2.
  All rawstorage/AUTO/repeat bits pass. Graph replay bits pass on graph rows;
  the eager rows' graph flag is false/notapplicable, not a replay failure.
- Old/new paired p50=1.036022--1.042052 and p95=1.038318--1.045031.
  New time is approximately3.5--4.0% lower at the median. All8 own-win cohorts
  pass, but AUTO/explicit cuBLAS FAST_TF32 median remains1.912826--2.121482.
- `prefix-special-bias-graph.log`:1test passed,0failed,64.38s. Real AUTO
  C0/C1 output equals forced prior M128 bit-for-bit; M1 thin AUTO equals the
  first row; specialbias patterns and actual M64 graph replay also pass.
- Main independently verified SHA256SUMS, parsed all8records and completion,
  checked actualsymbols, applicablebitflags and ratios; static scoped review
  found no issue. Rustfmt and code-only diffcheck pass.

Prepromotion screen/101 evidence is in sibling
`../fixed-tf32-all6-ada-20260906/`. This directory is postpromotion proof, not
an extrapolation of those forced-candidate measurements. Source and binary
hashes are in `preflight.log`; Fixed source/artifact/invocation/library hashes
are embedded in each performance record. SM120 compositions are untouched.
