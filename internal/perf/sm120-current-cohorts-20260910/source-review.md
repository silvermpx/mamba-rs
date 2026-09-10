# SM120 current-cohort integration review

Review scope: frozen `sm120-current-cohorts-review.diff` only, against base
`73248b4a`, requirements in `sm120-current-cohorts-brief.md`, and controller
evidence in `sm120-current-cohorts-controller-report.md`.

Frozen files independently matched the supplied hashes:

- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs`:
  `e86bc0d2b120a5638190fab246e19b02953b770d299395feaf09ef4a19b6d8ca`
- `tests/gemm_bi_tf32_cohort_binding.rs`:
  `e4922a1a2659b8856079174636eebfe5ee8d2f76395e627dc35613ffbdeaf1f8`

## Findings

### Important: rejected G10's exact-route assertion is vacuous for empty evidence

At `tests/gemm_bi_tf32_cohort_binding.rs:598-617`, the rejected G10 arm uses
`nodes.iter().all(...)`. An empty `nodes` slice therefore satisfies the assertion,
so this test can pass without proving the requirement that G10 actually executed
an exact-F32 route. `validate_timed_request` proves request identity after the
launch, but it does not make this physical-node assertion non-vacuous.

Add a non-empty assertion before the `all` check (or destructure/assert the exact
allowed node inventory). Keep the existing exact-contract and no-`tf32` checks.

### Important: portable-twin mismatch isolation does not cover the new lower-toolkit identities

The requested portable-twin mismatch coverage remains specialized to CUDA 13.2:
`sm120_tf32_live_portable_tn_cell_is_exact_and_fail_closed` obtains only
`sm120_cohort((13, 2))` at
`src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:13655-13657`, then applies portable
identity mutations at lines 13799-13813. The table-driven all-live-cohort test
later mutates only `availability.specialized` at lines 14713-14725. Thus the fresh
CUDA 12.8 and 13.0 portable twins are proven literal and live, but not proven to
fail closed under a portable identity mismatch as explicitly required.

Make the portable mutation check table-driven over the three current 595.84
cohorts (or add the portable mutations to the existing all-live loop). This is a
host-only coverage correction and does not require changing production tables or
rerunning qualification.

## Verified requirements

- I independently compared every accepted receipt's 17 specialized and 17
  portable identity fields with the four new production constants. CUDA 12.8
  `internal/perf/sm120-tf32-retained-20260910/cuda128/all24-v2/G*.jsonl`
  contributes 23 accepted records; CUDA 13.0 contributes 21 accepted V1 records
  from `cuda130/{G01,rest}` plus accepted V2 G06/G09 from `warmup-v2`. All fields
  match within each toolkit. The failed V1 G06/G09 copies were not used.
- Each current driver-595.84 cohort has the same 23 unique typed keys, exact
  routes and operand gates; six use `TriadSm80`, five use stream-K. The production
  manifest matches both accepted lower-toolkit receipt sets. True G10 is the sole
  excluded key and the binding fixture uses exactly TN `(8192,128,128)` with
  `alpha=1`, `beta=1`, no bias.
- Fresh CUDA 12.8/13.0 specialized and portable identities are live. Both existing
  CUDA 13.2 cohorts and all retired cohorts remain. Tuning revision remains 45.
- Cross-version specialized-body splices are rejected. Literal identity tests
  cover the new compiler, artifact, source, invocation, header, NVRTC-domain,
  driver, device and resource fields.
- The 24-case binding fixture's 23 accepted symbols, modules and numeric contracts
  match the accepted receipts. It checks actual AUTO, eager metadata/graph
  metadata equality, eager repeat and graph output-bit equality, A/B immutability,
  allocation red zones and the frozen request. TN resets nonzero C on every run.
- The frozen diff changes only dispatch and the binding test; it contains no CUDA
  source or Inference change and does not enlarge the canonical benchmark. The
  SM89 portion of `tf32_cohort_binds_on_this_board` is byte-identical to base from
  its original `shapes` declaration onward.

Root-provided execution evidence: final host 20 SM120 tests pass; Ada preservation
129 pass/1 ignored; the live 24-case post-AUTO test passes on CUDA 12.8, 13.0 and
13.2, with identical output hashes across toolkits. I did not rerun builds or GPU
tests, per scope.

## Verdicts

- **Specification compliance: NEEDS CHANGES.** Production admission and live
  behavior match all accepted receipts, but the two explicit fail-closed proof
  requirements above are incomplete.
- **Code quality: NEEDS CHANGES.** The implementation is otherwise narrowly
  scoped, typed and internally consistent; both findings are small test-only
  corrections.

## Implementer resolution

Both findings were addressed without changing production constants, cohort
selection, CUDA source, arithmetic, or tuning revision:

- `tests/gemm_bi_tf32_cohort_binding.rs` now requires the rejected G10
  evidence-node list to be nonempty before checking that every node carries an
  allowed exact-F32 numeric contract and a non-TF32 symbol.
- `sm120_tf32_live_portable_tn_cell_is_exact_and_fail_closed` now constructs
  the three exact driver-595.84 specialized/portable identity pairs for CUDA
  12.8, 13.0, and 13.2. Each baseline pair must select the portable winner;
  every existing portable and coupled-identity mutation must then fail closed
  to scalar FMA. The separate CUDA 13.2 / driver-595.58.03 cohort is not
  unwrapped or assumed to have a portable twin.

Final source hashes:

- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs`:
  `c4ef0dbe74b0d20a19888a84d8c137fd4da8c7839606bc542a917b947979f90e`
- `tests/gemm_bi_tf32_cohort_binding.rs`:
  `e79597052dfa926b9f28f94c0a80a9e129f8ff43c2e184201140b1db0e03f6ee`

The two files pass `rustfmt --check` and `git diff --check`. CUDA-host and
hardware reruns, plus the scoped re-review, remain root-owned and were pending
when this resolution was recorded.
