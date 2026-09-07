# Task6B — production S3/AUTO/Fast paired qualification

## Measured outcome (timing and fix1 verification complete)

Only CUDA13.2 B0/no-bias BF16 and F16 confirm strict own improvement over actual
AUTO42. CUDA12.8/13.0 BF16 fail screen21 p95; F16 passes21 but fails fresh101
p95. All six cells are valid evidence; no screening or confirmation was repeated.
Fast remains faster in all six aggregate vendor comparisons. Vendor victory was
not an own-improvement admission condition. No AUTO/production change is made.

| Toolkit | Dtype | Screen21 worst p50/p95 | Confirm101 worst p50/p95 | Admission |
|---|---|---|---|---|
|12.8|BF16|0.983487847 / 1.006867571|not eligible|false: mixed screen|
|12.8|F16|0.981792374 / 0.997414724|0.981555906 / 1.006347845|false: mixed confirm|
|13.0|BF16|0.983097932 / 1.011073358|not eligible|false: mixed screen|
|13.0|F16|0.981527365 / 0.994274018|0.981151367 / 1.015680726|false: mixed confirm|
|13.2|BF16|0.930532163 / 0.951183161|0.929639720 / 0.945012679|true|
|13.2|F16|0.929007877 / 0.955216865|0.927334868 / 0.958864521|true|

`winner-matrix.json` is the exact six-cell machine-readable decision with full
constituents, independently recomputed Fast comparisons, failing configurations
and original raw SHA256s. Root independently recomputed all3 smokes/screens/
confirmations and reached the same result. Review identified Important I1
closure and Minor M1/M2 preflight gaps, with the disposition recorded below.
Original timing artifacts are frozen and archived before the host-only fix;
timing was not repeated.

## Every own-comparison constituent (S3/AUTO)

| Toolkit | Dtype | Path/start | Screen21 p50 / p95 | Confirm101 p50 / p95 |
|---|---|---|---|---|
| 12.8 | bf16 | eager/0 | 0.980059020 / 1.004408698 | not eligible |
| 12.8 | bf16 | eager/1 | 0.980753849 / 1.005192299 | not eligible |
| 12.8 | bf16 | graph/0 | 0.983154210 / 1.006867571 | not eligible |
| 12.8 | bf16 | graph/1 | 0.983487847 / 1.006778259 | not eligible |
| 12.8 | f16 | eager/0 | 0.979498476 / 0.989047192 | 0.978215226 / 0.990204779 |
| 12.8 | f16 | eager/1 | 0.979027683 / 0.986139948 | 0.978199220 / 0.993396337 |
| 12.8 | f16 | graph/0 | 0.981792374 / 0.996801803 | 0.981555906 / 1.006347845 |
| 12.8 | f16 | graph/1 | 0.981133898 / 0.997414724 | 0.980788957 / 1.004545489 |
| 13.0 | bf16 | eager/0 | 0.980479667 / 0.989670476 | not eligible |
| 13.0 | bf16 | eager/1 | 0.980295185 / 1.000049305 | not eligible |
| 13.0 | bf16 | graph/0 | 0.983097932 / 1.011073358 | not eligible |
| 13.0 | bf16 | graph/1 | 0.981726185 / 0.992311614 | not eligible |
| 13.0 | f16 | eager/0 | 0.978637834 / 0.987726169 | 0.978067484 / 1.003248287 |
| 13.0 | f16 | eager/1 | 0.978043904 / 0.993010146 | 0.978144891 / 1.003723710 |
| 13.0 | f16 | graph/0 | 0.981527365 / 0.994274018 | 0.981151367 / 1.015680726 |
| 13.0 | f16 | graph/1 | 0.980976022 / 0.993440358 | 0.980401767 / 1.015307813 |
| 13.2 | bf16 | eager/0 | 0.927468784 / 0.948558058 | 0.925848068 / 0.940969108 |
| 13.2 | bf16 | eager/1 | 0.926271991 / 0.949484031 | 0.924775696 / 0.939621291 |
| 13.2 | bf16 | graph/0 | 0.930532163 / 0.948956784 | 0.929524753 / 0.942012124 |
| 13.2 | bf16 | graph/1 | 0.929174429 / 0.951183161 | 0.929639720 / 0.945012679 |
| 13.2 | f16 | eager/0 | 0.924619200 / 0.946433951 | 0.923088768 / 0.958864521 |
| 13.2 | f16 | eager/1 | 0.923707167 / 0.940796552 | 0.923783276 / 0.948764578 |
| 13.2 | f16 | graph/0 | 0.929007877 / 0.955216865 | 0.926550193 / 0.952259713 |
| 13.2 | f16 | graph/1 | 0.928364602 / 0.950737322 | 0.927334868 / 0.957009663 |

## Fast context (worst across paths/parities at the last eligible stage)

|Toolkit|Dtype|Stage|AUTO/Fast p50/p95|S3/Fast p50/p95|
|---|---|---|---|---|
|12.8|BF16|21|1.243442247 / 1.276196594|1.218393753 / 1.239650838|
|12.8|F16|101|1.188320313 / 1.219178081|1.147593570 / 1.168427719|
|13.0|BF16|21|1.243533701 / 1.299975061|1.221118769 / 1.233769374|
|13.0|F16|101|1.188759046 / 1.217104266|1.145417047 / 1.168284489|
|13.2|BF16|101|1.311991314 / 1.352387974|1.213927713 / 1.232313497|
|13.2|F16|101|1.243756245 / 1.271113505|1.131168506 / 1.167174487|

## Ownership and completed initial milestones

Base0892986d6382fc00f5012e3f58b750bdb37383fb. Only the authorized performance
test and new evidence/report paths are changed. No production or AUTO/revision42
changes. This task exclusively owned the Ada lane through the final release
recorded below; no competing GPU process was present.

Evidence root: `internal/perf/ada-half-s3-paired-20260907/`.
Remote source `/root/mamba-ada-half-s3-paired-20260907`; remote evidence
`/root/evidence-ada-half-s3-paired-20260907`. Isolated per-toolkit targets
`/root/target-ada-half-s3-paired-cuda{128,130,132}-20260907`; private0700 caches
`/root/mamba-kcache-ada-half-s3-paired-cuda{128,130,132}-20260907`.

Live lane precheck passed02:57:03Z: exact UUID
GPU-d1edd7be-e88d-aed6-047d-622163306f0e, RTX6000 Ada, CC8.9,0%GPU/0%memory,
noapps. Initial query included unsupported nvidia-smi field
`multiprocessor_count`; corrected query used supported fields and runtime
asserted142SM. An optional process-inspection command also found remote `rg`
unavailable. Neither failure launched GPU work or supplied acceptance evidence.

## Completed RED/GREEN and first smoke

Matching CUDA12.8 command (matching CUDA_HOME/PATH/LD_LIBRARY_PATH and isolated
CARGO_TARGET_DIR):

`cargo test --release --features cuda,cudarc/cuda-12080 --test gemm_bi_fixed_performance ada_s3_pair::mirrored_brackets -- --nocapture`

`red.log`: deliberately unmirrored ABAB/incomplete schedule rejected against
literal ABBA/BAAB/traversal expectations,0passed/1failed, exit101. Initial
source preserved in source-red.tar.gz (includes harmless AppleDouble/xattr
metadata, excluded from compiler-input manifests). tar emitted xattr warnings.

`green-build-attempt1.log`: focused command with filter `ada_s3_pair::` passes
3/3,exit0. Initial source SHA256
b57b825a8a3bdfc7548f55c7837244f9b02ecd7139c2d88749f5baab49ac7a1d.
CUDA12.8 binary SHA256
574c43f5cd1de38151a09d7604848924537c597e94e57c130e531aa7ebd7c138.

`python3 /root/evidence-ada-half-s3-paired-20260907/run.py build 12.8 build1`
passes matching focused3/3 and full nonignored51passed/64ignored; complete
compiler/library/source/binary binding is cuda128-binding.json. Initial build
SSH tool invocation exits0; later run logs additionally retain explicit outer
SSH closure markers.

`python3 /root/evidence-ada-half-s3-paired-20260907/run.py run 12.8 smoke1 1 bf16,f16`
passes one ignored test, all8 configurations,96raw observations,24pairs,
24summaries. test/postcheck/wrapper/outerSSH exits0. PRE03:06:04Z exact
UUID/idle/noapps; POST03:07:44Z same identity/noapps with residual9%/1%
utilization correctly accepted. `cuda128-smoke1/test.log` and matching SSH
log retain actual captured arguments, native-half Fast modes, guarded
allocation offsets, independent one-op/20-op graphs and complement poison,
repeat, numerical, no-op and immutable-input gates. Artifact identity exactly
matches approved Task6A CUDA12.8 identity.

`python3 internal/perf/ada-half-s3-paired-20260907/analyze.py <smoke-directory> <binding> <Task6A-identity> <SSH-log>`
produces cuda128-smoke1-analysis.json, valid8/96/24. This one-window functional
smoke is not screening/admission evidence. Its worst S3/AUTO ratios are
BF16.983800 and F16.982296, each descriptive only.

`python3 internal/perf/ada-half-s3-paired-20260907/test_validation.py`
passes9 host test groups including actual smoke closure and many adversarial
subcases: chronology, exact cohorts, missing/duplicate/foreign samples/pairs,
wrong arm/position/parity/traversal, invalid timing, K256, wrong dtype/toolkit/
revision, captured arguments, cuBLAS modes, source/binary mismatch, forged
summaries, true recomputed loss/mixed-p95, telemetry and all exit layers.
See analyzer-wrapper-green1.log. Genuine losses remain valid with admissionfalse.

## Final measured source, commands and identities

Direct rustfmt produced a pure692-line addition; no historical Rust line was
removed or changed. Frozen measured source SHA256:
f43a2a22dea7435716ff0c99d721e1d255ad9311f18e9f0b53a15280eafb582b.
All subsequent source/binary bindings use this hash; formatting-era initial
smoke/bindings and both initial binaries remain separately preserved.

`python3 /root/evidence-ada-half-s3-paired-20260907/run-final.py build <toolkit> buildfinal`
ran separately for12.8/13.0/13.2. Each passes focused3 and nonignored51/64.
Matching cargo features are exactly cuda,cudarc/cuda-12080; cuda,cudarc/cuda-13000;
cuda,cudarc/cuda-13020. The wrapper exports matching CUDA_HOME,CUDA_PATH,PATH,
LD_LIBRARY_PATH, isolated CARGO_TARGET_DIR and private0700 MAMBA_RS_KERNEL_CACHE.
`all-buildfinal-ssh.log` holds command/full wrapper/outerSSH closure; per-toolkit
buildfinal folders hold compiler and suite outputs. Full356-file source/build
input, actual NVRTC/cuBLAS library and nvcc/ptxas hashes are in each finalbinding.

|Toolkit|Actual measured test binary SHA256|
|---|---|
|12.8|618a31de2bf2936ce1e1dfbab8dd3bbeca35db35b06650bfc2e346cce90233b3|
|13.0|57deaa66bff293b0570f65c07b12fb1e31bfc4c54320f9f8fb7376e850b63062|
|13.2|8edb3d6f0a7e8e8b12da2ede6c0dfee9ce3c2e1b074878c6fb0dc888dde7f354|

Literal run command template:
`python3 /root/evidence-ada-half-s3-paired-20260907/run-final.py run <toolkit> <attempt> <windows> <dtypes>`.

The complete allowed run list was:

- Eachtoolkit: `smokefinal 1 bf16,f16`, then `screen21 21 bf16,f16`.
- CUDA12.8/13.0: `confirm101 101 f16` only.
- CUDA13.2: `confirm101 101 bf16,f16`.

Each actual command, binary, PRE/POST raw JSON, RUN_RESULT, test exit, wrapper
exit and outer SSH exit is recorded in its matching `cuda*-<attempt>-ssh.log`.
Every actual run has all exits0. Initial12.8 smoke is an additional explicitly
historical functional run, not another screen. Finalsmokes:8config/96sample/
24pair/24summary each. Screens:8/2016/504/24 each. Confirm12.8/13.0:4/4848/
1212/12 each;13.2:8/9696/2424/24. No omitted or duplicate cohorts. The last
actual GPU benchmark completed03:21:02Z. POST accepts residual utilization
while still requiring exact identity and noapps; every PRE requires0%/0%.

All three graph inventories retain actual AUTO Swizzle69632B and S3 98304B,
grid666x1x1,block256x1x1, exact five argument layout and decoded pointers/
bundle `[1065353216,0,4621,2304,768,768,2304,2304]`, including20 matching nodes
per custom timed graph. Vendor inventories retain actual cuBLAS kernel nodes;
Fast explicitly pins/queries DEFAULT_MATH, HOST pointer and ATOMICS_NOT_ALLOWED,
with COMPUTE_32F and GEMM_DEFAULT_TENSOR_OP. PEDANTIC_F32 is reference only.
The algorithm enum is a literal Task5B compatibility pin: cuBLAS section2.2.9
states legacy GemmAlgo values have no effect on SM80+, so this is not a claim
that the historical GEMM_DEFAULT comparator was non-Fast on Ada. See the
[CUDA12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasgemmalgo-t)
and [CUDA13.2](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#cublasgemmalgo-t)
references already supplied in the brief. Storage/compute/math and actual
physical kernels determine the comparator contract.

Physical/compiler identity matches Task6A exactly on all3. Fixed source digest
205f58b56429b8e74f3ac1e7ab9f0cf6a9bf193ffeb87b254bd3ab00e6b02ca5.
Fixed PTX artifact hashes12.8/13.0/13.2 respectively:
60977db33de28d807ac7dd3dafe2d176914f7b7eab589917a47988050712352c;
d1aa6e33a612d99cd44a1e9c7eebe495c05eeef2212bf2cf4296db55eedb86a9;
8b89aadf7d456b78ab249ff614568730946006eefd7143e35bce1e158741aff1.
Because these are byte-identical production payloads, approved Task6A resource
evidence remains applicable: BF16/F16 registers182/182 on12.8/13.0 and188/188
on13.2, zero local/stack/spills/static shared, dynamic shared98304,occupancy1.
This task does not claim a fresh resource measurement or standalone NVCC timing.

## Review fix1 — exact same-attempt closure

Root ruled Important I1 in scope and Minor M1/M2 deferred explicitly to immediate
Task6C. M1 omits several unused legacy environment controls from preflight;
they cannot filter or redirect this literal experiment. M2 first compares A/B
immutability after timing; that postcheck still prevents corrupted input from
becoming accepted evidence. Neither deferral alters frozen source or results.
No Rust/run.py/binary/CUDA change or GPU/timing rerun occurred for fix1.

RED command:
`PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-half-s3-paired-20260907/test_validation.py Wrapper.test_verify_run_rejects_conflicting_or_foreign_attempt_transcripts Wrapper.test_verify_run_binds_saved_telemetry_phase_and_result_to_transcript`

`fix1-red.log`:2tests,19 meaningful failures,exit1 against old validator.
Conflicting/suffixed exits, unrelated attempts, wrong command/binary, mismatched
RUN_RESULT, telemetry and phase swaps were incorrectly accepted before the fix.

The corrected analyzer parses exactly eight ordered, anchored records, checks
actual test/command/wrapper/SSH exits, matches RUN_RESULT to result.json and
PRE/POST JSON to saved files and phase labels, and binds the precise executed
command to the supplied binary. Extra failure/conflict/duplicate/missing/
suffix records fail. Allowed POST residual utilization remains valid.

GREEN command:
`PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-half-s3-paired-20260907/test_validation.py`

`fix1-green.log`:11testgroups pass,exit0, including true recomputed21/101
win/loss/mixed-p95 cases. Corrected analyzer SHA256
0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30;
tests SHA2564461fe0d3562e27be393743c97fcd47016a9d75fbfa3cdffc69a430c3ecbe131.
Unchanged wrapper SHA256ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713.

Revalidation command for each of all10 actual records (initialsmoke plus final
3smokes/3screens/3confirms):
`python3 internal/perf/ada-half-s3-paired-20260907/analyze.py <run-directory> <matching-binding.json> <Task6A-identity.json> <same-attempt-ssh.log>`.
`fix1-revalidation.log` records10valid results. Regenerated derived JSONs bind
validator/source/SSH/binding/qualification SHA256s; raw logs/bindings are unchanged.
Pre-fix derived JSONs are preserved in pre-fix1-derived.tar.gz.
`python3 internal/perf/ada-half-s3-paired-20260907/matrix.py` regenerates the same
six decisions. Winner matrix SHA256
f95624e34a5646ccee75647a4bd515d2def73f4e0bd60d4b471bdce5a0cf6eb4.

## Archive, failure history and handoff

`source-final.tar.gz` retains all356 measured inputs. Per-toolkit
`cuda*-final-binary-cache.tar.gz` retains actual measured executable plus3cache
envelopes. `verify_artifacts.py` verified all356inputs against each binding and
approved Task6A archive: only the performance test differs. All3binaries and
9cache envelopes reconcile; every Fixed payload exactly matches Task6A.
`archive-verification.json` records PASS, independently reproduced by root.
`root-recomputed.json` retains root's separate all9 final run arithmetic,
chronology and raw-hash checks. Source/resource/identity evidence stays at
revision42 throughout. No old Task6A evidence was mutated.

Failures were limited to intended host REDs/negative controls and preparation:
unsupported nvidia-smi field, unavailable remote rg, harmless initial tar xattr
warnings, and an initially guessed source-review filename which did not exist
(immediately read the named source-review file from fix1 brief). Initial
disabled/stale-control negative output omitted stderr in its tee log; a second
host-only invocation preserved full diagnostics in disabled-stale-negative-full.log.
Both deliberate cases exit101 before GPU creation; neither is performance data.
No functional GPU run,screen,confirm or production artifact gate failed.
The four rejected decision cells are valid performance losses/mixed p95, not
invalid runs, and were not repeated.

GPU lane released2026-09-07T03:28:29Z: exact UUID
GPU-d1edd7be-e88d-aed6-047d-622163306f0e, RTX6000 Ada CC8.9,0%GPU/0%memory,
noapps. `lane-release.log` records identity/idle,noapps,outerSSH exits0. No more
Task6B GPU work. Root owns the next lane and all stage/commit/push operations.

`SHA256SUMS` is rooted at the worktree and covers this evidence directory,
including the final-report.md copy, actual raw logs/bindings, archives, scripts,
derived analyses, matrix and root recomputation. It excludes itself, its
verification output and incidental Python bytecode. `manifest-verification.log`
records complete verification. Root fix1 source re-review is the remaining
integration review, separate from the completed measurement/verification work.
