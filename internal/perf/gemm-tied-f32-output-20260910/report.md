# Deterministic tied half-input / F32-output verification

Task903; base `779ef943dfa785eec0d07d37e042be065fa18724`.
Source: `21a3f0e52690c6d2613a039174427424148f5d32`.
Status: root verification passed; independent spec/quality review approved.
RTX6000Ada, CC8.9, CUDA13.2; root owns execution. No performance measurements.

## Tests-first RED

Frozen test SHA256:
`7bb6ecfd9f51f134ee4267025c94d8657cb3422209bc0335b3a58a0310fea517`.
Preserved at `source-snapshots/gemm_tied_f32_output_red.rs`.

Both actual GPU tests fail at the expected explicit boundary:
`context-aware GemmEx: deterministic GEMM mode reached a cuBLAS dispatch boundary`.

| Exact test | Exit | Runtime |
|---|---|---|
| `deterministic_tied_bf16_inputs_preserve_true_f32_product_bits` |101|4.91s|
| `deterministic_tied_f16_inputs_preserve_true_f32_product_bits` |101|4.88s|

Compilation passed41.13s. Each test ran one case with zero ignored; the runner
requires the expected failing-test status. Runner exit0, source-after manifest
PASS, completed `2026-09-10T12:35:45Z`. Raw receipts are in `red/`.

Immutable source: `/root/mamba-tied-head-red.JpIMJc`.
Remote packet: `/root/tied-head-red-evidence-20260910`.
Production src/kernels were unchanged from the base at this checkpoint.
Root authorized the same implementer to proceed only after observing both REDs.

## GREEN: exact source verification

Immutable source: `/root/mamba-tied-head-green.qhBBIV`.
Final six source hashes are in `implementation-report.md`; the full source
manifest and Cargo inputs are in `green/`. Remote before/after and local
manifest checks pass. All CUDA/header bytes are unchanged from the base;
scoped rustfmt and Git whitespace checks pass.

| Check | Actual result |
|---|---|
| CUDA+HF release all-target compilation | PASS,10.00s |
| CUDA-only release library compilation | PASS,6.51s |
| Integration bits, irregular inputs, guards, null/overflow, K0 |6 passed,0 failed,0 ignored;28.03s |
| Two upcasts, all NT launches, no output downcast |1 passed;9.57s |
| K0 observer, F32 epilogue only |1 passed;5.00s |
| Frozen scratch reuse and rejected growth |1 passed;4.91s |
| Unsupported/mismatched private input types |1 passed;4.98s |
| BF16 M1 actual capture then first tied head |1 passed;5.02s |
| F16 M3 actual capture then first tied head |1 passed;5.21s |
| CUDA+HF Rustdoc, broken links denied |PASS,2.51s |

Twelve real CUDA test cases passed with none ignored. The integration target
build took54.42s and the colocated library test build57.13s. The first runner
stopped with exit1 before its second lib case because a preflight sample was
9% GPU utilization despite48463MiB free. No test failed. A following telemetry
check found0% and no compute process. `green-resume/` ran only the remaining
five cases plus Rustdoc, verifying the same source and executable hashes
before and after. Its bounded idle retries retain every telemetry sample;
every test launch still required<=1% utilization and>=2048MiB free.
Continuation exit0, completed `2026-09-10T13:19:57Z`.

Existing warnings remain visible in raw logs: deprecated compatibility setters
in old tests, a CUDA-only M3 `ctx` accessor unused without HF, and the known
public-to-private Rustdoc link in `gemm_bi_triad/contract.rs:2834`. They are
queued for the API/docs/test-cleanup phase; no lint suppression was added.

This establishes the focused tied-half/F32-output and workspace behavior on
Ada/CUDA13.2. It is not a speed measurement or fresh5090 result. Complete
Inference physical inventory, M3 graph-plan guards and high-level no-vendor
tripwire acceptance are subsequent release tasks.

## Independent review

`review.md` approves spec compliance and task quality, with no Critical or
Important findings. The one Minor is the pre-existing warning debt recorded
above and retained for release cleanup. Root resolved the review's external-
evidence items against its actual12-case receipts, full562-file source manifests,
unchanged CUDA/header diff, scoped formatting checks and Git inspection of the
single source commit on the existing worktree branch. No push, merge, new branch
or release publication occurred in this task.
