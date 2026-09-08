# Agent Operational Rules

## Active user override: discovery first (2026-09-08)

- Coverage is the WHOLE Triad: NN, NT and TN, all five measured shape families
  and all four precisions. TF32/F16/BF16 target cuBLAS Fast wins; exact F32
  targets the closest achievable Fast performance without weakening its bits.
  A retained candidate or one near-parity cell does not close a precision/op.
- Current task is to FIND faster Triad candidates, not repeatedly confirm the
  existing three scalar NN wins. Stop pending integration/once21/all-toolkit
  runs until a batch of performance finalists is assembled.
- Search on Ada / CUDA13.2 only. Reuse existing harnesses: focused output-bit,
  repeat/graph and guard checks, then short once7 paired speed comparisons.
  Check resource viability; do not run full suites/sanitizers/toolkit matrices
  for every candidate. A failed bit check blocks that candidate's speed claim.
- Keep candidate arithmetic and the relevant existing bit contract unchanged.
  Compare against the real selected route and explicit cuBLAS Fast. Fix only
  harness defects that would invalidate the immediate comparison.
- Preserve256-byte base alignment in timed guarded allocations:128 half words
  or64 F32 words, and check actual pointer alignment. A16-byte-offset half NN
  guard doubled cuBLAS time in the 2026-09-08 diagnostic. Misalignment belongs
  in correctness probes; do not use that cohort as the sole production baseline.
- cuBLAS graphs may contain multiple/opaque/non-kernel nodes. Require nonempty
  capture, check vendor eager/graph output and time the entire graph; do not
  impose our candidate node count or kernel-parameter ABI on vendor internals.
- Test a small batch of distinct, source/profile-grounded hypotheses. Preserve
  losers and their results without repeatedly rerunning the same losing case.
  Keep experiments out of production; preserve all existing WIP and SM120 routes.
- After collecting finalists, integrate together and run one combined
  correctness/dispatcher/CUDA12.8+13.0+13.2 confirmation batch.
  User clarified: reaching the end of a small discovery batch is NOT the
  trigger. First assemble the replacement shortlist across the whole Triad
  performance matrix; do not start full gates/integration merely because a
  few cells won. Keep unresolved Fast gaps explicit.
- Report NEW measured speed wins separately from wiring/validation progress.
  Do not present the old three F32 NN cases or old N96 result as new wins.
- Keep coordination short: no repeated large source snapshots, review loops,
  report rewrites or full-matrix reruns without a concrete new need.
- Archive the next frozen test snapshot outside the shared remote source tree.
  Install it only AFTER the preceding build and test process exit; otherwise
  Cargo may consider an older executable newer than the newly installed source.
  Exact-test list gates must reject stale artifacts before GPU work. For a
  timestamp-only repair, preserve source hashes and rebuild the focused test;
  do not rerun valid preceding measurements or regenerate broad manifests.
- Resolve exact test names from each binary's --list: typed parity names are
  bare, while TF32 compact names are nested under cuda_suite::. Do not reuse
  one harness's module prefix for another; a wrapper-only name repair does
  not require rebuilding the unchanged binary.

## CUDA execution lanes

- Before launching any build or test, classify it by feature set and target architecture. Route SM89 work to `ada` and SM120 work to the dedicated RTX 5090 lane.
- Any command using `--features cuda` must run on a CUDA host, even when the selected test is logically CPU-only. The CUDA dependency build requires the CUDA toolkit.
- NVRTC, NVCC, PTX, SASS, `ptxas`, `nvdisasm`, `cuobjdump`, compute-sanitizer, CUDA runtime tests, and GPU tests must run on the matching CUDA host after syncing the exact files under test.
- Flotilla and local Mac workers are only for Rust work that does not enable the CUDA feature.
- Functional CUDA compilation and GPU correctness tests may run while another non-exclusive workload is present on `ada`.
- Timed performance benchmarks must wait until the user's current GPU benchmark finishes and the user confirms the performance lane is available.
- Never stop or kill another GPU process unless the user explicitly requests that exact action.

## SM120 environment

- Host: `ssh -p 16152 root@ssh1.vast.ai` (vast proxy; the direct 42.113.51.8:52796 port died on 2026-09-02, same container). Long runs only inside tmux (`tmux new-session -d -s triad ...`): plain setsid/nohup chains were killed when the session dropped.
- GPU: RTX 5090, CC 12.0, 170 SMs, 32 GiB, driver 595.91.07 (driver API 13.2).
- Toolkits: CUDA 13.0.88 at `/usr/local/cuda-13.0` (image default) and CUDA 13.2 at `/usr/local/cuda-13.2` (installed side by side; production exact-shape admissions require NVRTC 13.2).
- Synced source: `/root/mamba-rs-triad`.
- Check compute processes, utilization, clocks, and temperature immediately before timed work.

```sh
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
```

## Ada environment

```sh
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
```

Use an isolated remote source directory and an isolated `CARGO_TARGET_DIR` for each OLD/NEW qualification build. Disable shared CUDA/kernel caches for evidence-producing qualification runs.
