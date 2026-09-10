# RTX 5090 current TF32 cohorts: production AUTO integration

Current hardware: RTX5090, CC12.0,170 SMs, driver595.84,
UUID `GPU-a10ad830-a2cf-054d-e0fa-54add30a2cf8`.

Initial three-toolkit dispatch SHA-256:
`e86bc0d2b120a5638190fab246e19b02953b770d299395feaf09ef4a19b6d8ca`.
Initial three-toolkit binding test SHA-256:
`e4922a1a2659b8856079174636eebfe5ee8d2f76395e627dc35613ffbdeaf1f8`.
Every packet includes its own source, executable and runner manifest.

Final reviewed dispatch SHA-256:
`c4ef0dbe74b0d20a19888a84d8c137fd4da8c7839606bc542a917b947979f90e`.
Final reviewed binding test SHA-256:
`e79597052dfa926b9f28f94c0a80a9e129f8ff43c2e184201140b1db0e03f6ee`.
The final review delta changes only test imports/assertions; production
cohort tables and every CUDA implementation remain identical.

## Change

The current CUDA12.8 and13.0 specialized/portable identity pairs are now
represented in production AUTO. Allthree current toolkits share exactly23
independently accepted operation/shape keys, including six portable paths.
True G10, public TN(M,K,N)=(8192,128,128), independently lost on allthree
toolkits and therefore stays on the exact F32 path. Its kernel remains
available for other legitimate selections; no CUDA implementation was deleted.
G11 is admitted. Existing historical/other-driver cohorts remain intact.

The CUDA source, arithmetic, source-composition rules, tuning revision45,
Inference selectors and Ada production decisions are unchanged. Every one
of the17 fields in each of thefour fresh identities matches its accepted
top-level qualification receipt:68/68 checked independently by root.
Retained qualification receipts live in
`../sm120-tf32-retained-20260910/`.

## Actual production AUTO checks

The same ignored `tf32_cohort_binds_on_this_board` test ran the exact24
public cases, not forced candidates. It requires the expected physical
symbol/module/numeric contract for each admitted key, exact fallback for
G10, full-output repeated eager/graph bits, unchanged A/B operands, intact
allocation red zones, and frozen request/launch metadata.

| Toolkit | Actual cases | TF32 selections | Exact fallback | Result | Test duration |
|---|---:|---:|---:|---|---:|
| CUDA12.8 |24|23|1|PASS|16.94s|
| CUDA13.0 |24|23|1|PASS|17.66s|
| CUDA13.2 |24|23|1|PASS|17.33s|

Raw directories: `cuda128/`, `cuda130/`, `cuda132/`. All24 case/output-digest
records match across these three runs. This confirms bit-identical full
outputs across the tested toolkit versions on this fixed GPU/driver/source
and corpus; it is not a proof for arbitrary hardware or untested inputs.
The first printed case shares a Cargo test-prefix line, so parsers must
match `SM120 current cohort` anywhere on the line, not only at its start.

CUDA13.0's14 previously unexecuted mixed-precision matrix cells also pass:
28 eager/graph records in `cuda130-missing14/`,9.57s. Those complete the
earlier50-cell packet and the two-cell missing-symbol repair. This is
combined66-cell coverage, not a claim that all66 were rerun in one packet.

## Host/source verification

- Tests-first RED ecd03e74: CUDA13.0 compilation fails with E0432 for exactly
  thefive absent new constants. This is compile-time RED, not a behavioral
  test failure. Original log: `red/`.
- First dispatch GREEN24f34e12:20 SM120 tests pass, plus129 Ada preservation
  tests pass with one ignored. Logs: `host-green/`.
- Final dispatch e86bc0d2, expanded literal assertions:20 SM120 tests pass,
  release compilation51.44s, test execution0.01s. Logs: `host-fix1/`.
- The first binding draft incorrectly expected the portable numeric contract
  for specialized TMA nodes. It was corrected before GPU execution; the
  final test distinguishes portable, direct TMA and stream-K contracts.
- Independent review requested a nonempty G10 launch inventory and portable
  identity-mismatch tests for all three current toolkit pairs. Both test-only
  fixes are accepted by `source-rereview.md`; the original findings remain in
  `source-review.md`.
- Final reviewed dispatch c4ef0dbe: all20 focused SM120 host tests pass after
  a42.90s release build (`host-reviewfix/`). This includes the expanded
  three-toolkit portable-twin mutation test.
- Final reviewed binding e7959705: actual AUTO rechecks all24 cases on RTX5090 /
  CUDA13.2, PASS17.05s (`cuda132-reviewfix/`). Its24 full-output digest records
  are identical to the initial three-toolkit runs. No production change
  requires repeating the lower-toolkit qualification/performance sweeps.

## Preservation of full benchmark images

The four full paired AUTO benchmark packets retain their actual production
source revision `732c11462dd9a47376e6d15b36040ca69bde3e34`. Comparing that
revision with this assembly changes only `gemm_bi_triad/dispatch.rs` under
`src/` and `kernels/`: the lower-toolkit SM120 TF32 identities/manifest and
their tests. The only current CUDA13.2 route removal is true G10, which is
absent from the full66-cell paired Triad inventory (verified from its JSONL).
Other current13.2 routes, all Ada routes, Inference production code, CUDA
bytes and compiler composition are unchanged. Consequently those full
packets remain measurements of the same covered routes; their recorded SHA
must not be relabelled as the later integration commit.

Independent source review is recorded separately. These checks are binding,
correctness and determinism evidence, not a new cuBLAS performance win.
Full measured AUTO/cuBLAS comparisons for both cards are in
`../final-auto-benchmarks-20260910/`.
