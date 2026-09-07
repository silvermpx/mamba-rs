# Task7 Ada F32/TF32 cross-toolkit qualification report

Status: **QUALIFICATION COMPLETE / ROOT EVIDENCE ACCEPTANCE PENDING**, 2026-09-07.
The accepted final5b source, two binaries, four smoke1 runs, four screen21 runs,
and two exact-F32 eligible101 runs are frozen. Exact CopyPlan is a confirmed
owner win against actual Legacy AUTO for all 16 toolkit-literals, but loses to
Fast in every constituent. TF32 M64S2 loses to actual RNA AUTO for all four
toolkit-literals at screen21, so its eligible subset is empty and no TF32 101
was run. No selector, kernel, loader, holder, epoch, or routing change is made.

## Final5 fix1 checkpoint

The independent final4 review found three Important harness gaps and no
Critical/Minor findings. Fix round 1 now binds/emits/analyzes actual numeric ABI
5 and schedule 8; validates the confirm screen artifact before the first kernel
workflow; and checks saved A/B/bias plus all input/output guards immediately
after timed observations before any later read/replay, while retaining the
later check. Focused mutation tests cover each revision, prelaunch artifact
failure with zero launches, and A/B/bias corruption before a hypothetical
restoring replay. Full history and commands are in
`ada-f32-tf32-toolkit-task7-fix1-report.md`.

Formatted final5b file hashes are performance
`317cb44c60926bca32da6b3f7828187be7ab61b4173cf6b07297a86961698419`,
support `cb20805c079b200f1bb77f02912c16fb56113d26cf3204c04d046cc872cee73f`,
runner `24275ce7ed506336f64297a0eb29afaefe12d366ca4ed47860a0a16031dbd1aa`,
analyzer `e8bebda48987a699392f7c8bdbdf8f74b61349a2d3aebc20a36bf1e222026182`,
remote wrapper `cb91f07492a9e9b661d98e6be58d7b71f48511eae082ffbeca133d50e7bfdf0e`,
and Python tests
`43afdd008451cd89195200e687b3fe8142072969fbb701b78d5f3c33e5ab4608`.
Composite runtime SHA is
`97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7`.
Python RED exposed four missing revision rejects; GREEN is 9/9. Rust RED is
four absent-gate compile errors; GREEN is 12/12. The formatted final5b CUDA
12.8 build passes focused 12/12 and nonignored 65 pass + 65 ignored; binary SHA
is `0f3f2e617797fcce4e70789cc507cd25704429f98663812013d6df9851321bb5`.
CUDA 13.0 also passes focused/nonignored with binary
`cf90b10de3cb8afb0a4a68575a1c9db9597750ff3b929dfff85864a423009154`.
All four fresh final5b smokes pass full independent closure; JSONL SHAs are
12.8 exact `dc87b9e2...`, 12.8 TF32 `875155cb...`, 13.0 exact `7b66bcd8...`,
and 13.0 TF32 `ca7854c0...`. Build output contains disclosed unused-import
warnings for the two revision constants. The fix-only re-review accepted all
three Important fixes with no new Critical/Important finding, and root accepted
the source+smoke gate. The source remained byte-identical through every timing
run. Pre-format final5 identities remain intermediate history and are not
timing evidence.

## Controller correction and final4 checkpoint

The adjacent binding ruling corrects the original audit's historical TF32
incumbent: current aligned production AUTO is `Tf32RnaM128N128S3`, not
`Tf32M128S2`. The original final3 source checkpoint, exact smoke evidence, and
failed TF32 runtime RED below remain immutable history. No operand/alignment/
holder bypass or selector edit was made.

The corrected final4 runtime expects RNA grid111/block256/dynamic98304,
five-argument ABI ending `(32,32)`, and eight words
`[1.0bits,0,4621,1928,384,1928,384,384]`. Every one/20 node still checks the
full pointer/ABI/bundle/terminal contract, reusing the existing RNA-wide graph
contract. The candidate remains M64S2 with its distinct 24-byte bundle. The
TF32 single-active-term probe is now 2x4x4 (K and N multiples of four), with one
nonzero reduction term. Host regression explicitly rejects the old M128S2
incumbent and old portable 24-byte bundle.

Corrected frozen hashes:

| File | final4 SHA-256 |
|---|---|
| `tests/gemm_bi_fixed_performance.rs` | `7b698acbe5702482fe75a5c3e169e4b4d06a6b32a7c41e0e551a910cede20615` |
| `tests/support/fixed_sm89_toolkit_admission.rs` | `12cecc95a431bca91a6187e1a5df3202a4c07247ecca596e7d1c938fed8fa021` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/run.py` | `97bb4d0f7263a6d93d1b765ee905ac5d2bb9e347d2f47a35ec83215f1e5af3ba` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py` | `d9dc5303ec1b8af14f1b7e2b4d46d5d5894e62907d61df83b6ef3e7b760345af` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/remote.py` | `cb91f07492a9e9b661d98e6be58d7b71f48511eae082ffbeca133d50e7bfdf0e` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/test_validation.py` | `808e9726ca1196dab074cfe5ff573cec2d84d300e6c5173496710d7dec908d4f` |

Final4 composite SHA is
`e1b7175c93554e45f99f5b0a2ba999509655aed493d8d5d52737a4cbe392a64a`.
Matching final4 builds passed focused 10/10 and the full nonignored target on
both toolkits. CUDA 12.8 binary SHA is
`60000e18c652dbee7b651c6c07a5efacb77fd6a72237e08d20c25476daf6b699`;
CUDA 13.0 binary SHA is
`b7938c5a4dff5af74a32f03277d410d78f47dd490db37a150f6148e878d956d9`.
Targets and private caches use distinct `final4` paths, preserving final3.
`host-validation-rna-green.log` is 9/9, exit 0, SHA
`0e3d197c437efb4e5695f62d5e6e976a1ed446d3ab87bd2e6eb14a752dd9ecbb`.
All four matching final4 functional smokes passed test/POST/wrapper/outer SSH
and independent analyzer closure. The corrected source review is pending;
screen21 remains blocked on it.

## Scope and frozen checkpoint

- Worktree: `/Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80`
- Base: `e1d503c78694936b46ee79ca6b1a320a494ce7f5`
- Runtime source ownership is limited to
  `tests/gemm_bi_fixed_performance.rs` and
  `tests/support/fixed_sm89_toolkit_admission.rs`.
- Historical entry behavior is unchanged when
  `MAMBA_FIXED_ADA_TOOLKIT_ADMISSION` is absent. Present values other than
  exactly `1` reject before allocation.
- The focused support module owns strict controls, canonical literal inventory,
  public AUTO/forced/vendor replays, physical/bit/guard/input gates, scheduler,
  JSONL emission, and completion. `run.py` owns a single per-toolkit environment
  constructor for every identity/build/host/functional/timing command.
- Preserved original final3 runtime composite SHA:
  `4936366b8cdd9e9fa02669b0a329b6e563e0407bb99b4db59268a6e01c13bda5`.

Frozen file hashes:

| File | SHA-256 |
|---|---|
| `tests/gemm_bi_fixed_performance.rs` | `7b698acbe5702482fe75a5c3e169e4b4d06a6b32a7c41e0e551a910cede20615` |
| `tests/support/fixed_sm89_toolkit_admission.rs` | `d10e8d9e172ba93cf969ec18b90c33f5f3e8224498f79aeb972fe7a712fdae67` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/run.py` | `56d8ab73e955ae1a168853d27c61be9a1d07068d409822cd613c87767f618d48` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py` | `7c6e6baf0d6f15761ba9a5ffcf327bee5a93c5bdc15316893ffa34187934ec14` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/remote.py` | `cb91f07492a9e9b661d98e6be58d7b71f48511eae082ffbeca133d50e7bfdf0e` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/test_validation.py` | `71638e8e27e396b305e1430079caa8cd6298f325994a0ba21da0d213af58fb06` |

Direct `rustfmt --edition 2021` was run. Its incidental formatting of an
unowned child module and unrelated regions of the parent was restored with
`apply_patch`; the intended source above is `git diff --check` clean and is the
source used by both final3 binaries.

## Exact controls and environment

Every toolkit command is constructed by `env_for(toolkit)`. CUDA 12.8 uses
`CUDA_HOME=CUDA_PATH=/usr/local/cuda-12.8`, feature
`cuda,cudarc/cuda-12080`, target
`/root/target-ada-f32-tf32-toolkit-final5b-cuda128-20260907`, and private-0700 cache
`/root/mamba-kcache-ada-f32-tf32-toolkit-final5b-cuda128-20260907`. CUDA 13.0 uses
the corresponding `cuda-13.0`, `cudarc/cuda-13000`, `cuda130` target/cache
paths. `PATH`, `LD_LIBRARY_PATH`, Cargo target, and kernel cache are assigned
there; all stale `MAMBA_FIXED_*` and `NVIDIA_TF32_OVERRIDE` values are removed.

Functional/timing controls are:

```text
MAMBA_FIXED_ADA_TOOLKIT_ADMISSION=1
MAMBA_FIXED_ADA_VENDOR=1
MAMBA_FIXED_ADA_ROWS=f32_exact_fast|tf32
MAMBA_FIXED_ADA_LITERALS=<strict comma-separated cell:bias list>
MAMBA_FIXED_ADA_WINDOWS=1|21|101
MAMBA_FIXED_ADA_TOOLKIT=12.8|13.0
MAMBA_FIXED_ADA_TUNING_REVISION=43
MAMBA_FIXED_ADA_STAGE=smoke1|screen21|confirm101
MAMBA_FIXED_ADA_SOURCE_SHA=<frozen composite SHA>
MAMBA_FIXED_ADA_BINARY_SHA=<bound executable SHA>
MAMBA_FIXED_ADA_JSONL=<create-new path>
MAMBA_FIXED_VENDOR_TILES=F32Sm89N64CopyPlan|Tf32M64S2
MAMBA_FIXED_VENDOR_PATHS=eager,graph
MAMBA_FIXED_VENDOR_EXACT_CC=8.9
MAMBA_FIXED_ADA_SCREEN_SHA=<confirm only>
MAMBA_FIXED_ADA_SCREEN_ARTIFACT_SHA=<confirm only>
```

The runner fully parses and independently recomputes the stored screen before
launching confirm. The analyzer independently repeats source/binary/toolkit/
artifact, exact eligible-subset, raw schedule/arithmetic, physical, telemetry,
result, wrapper, and outer-SSH closure checks. A screen digest by itself is not
admission. Any malformed/incomplete/foreign shared screen rejects the whole
confirm; a valid own-loss removes only that literal.

## PRE, TDD, and host validation history

Fresh task PRE was taken before GPU work at `2026-09-07T05:09:43Z`:
exact UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, RTX 6000 Ada,
CC 8.9, 0% GPU, 0% memory, no compute apps, exits 0. Raw:
`internal/perf/ada-f32-tf32-toolkit-20260907/pre-task-strict-pre.log`.

Preserved development history (none is timing evidence):

| Raw path | Outcome |
|---|---|
| `cuda128-red1/host.log` | RED: seven expected failures, one negative pass; exit 101. |
| `cuda128-green1/host.log` | 7/8; host test compared entire schedules for different windows; preserved failure. |
| `cuda128-green2/host.log` | 8/8 pass. |
| `cuda128-red-jsonl/host.log` | RED for create-new JSONL control; exit 101. |
| `cuda128-impl1/host.log` | compile failures in struct literal/format capture; preserved. |
| remote `cuda128-red-artifact/host.log` | RED for confirm artifact binding. |
| `cuda128-green-artifact-physical/host.log` | compile failure from optional `serde_json`; preserved; dependency/features were not widened. |
| remote `cuda128-green-artifact-physical2/host.log` | 9/9 pass after exact native encoding assertion. |
| remote `cuda128-green-target/host.log` | 9/9 pass after stable `sm_89` target identity check. |
| `host-validation-green.log` | all 8 Python tests pass, but command wrapper exits 1 because zsh `status` is read-only; preserved. |
| `host-validation-green2.log` | corrected wrapper, 8/8 pass, exit 0. |

The Python suite consumes and parses actual Rust-emitted JSONL, checks ABI is
JSON arrays, and rejects missing/duplicate/foreign literals/records; malformed
stage/revision/source/binary/device/toolkit/vendor modes; wrong ABI, captured
args/pointers, terminal status, shape, physical flags; wrong schedule fields;
nonpositive/nonfinite raw times; forged pair/summary/config/global completion;
failed inner/POST/wrapper/outer closure; and foreign source/binary/toolkit
bindings. It proves a fully recomputed valid literal loss preserves eligible
siblings and that a mixed p95 is valid evidence with admission false. It also
tests the exact recomputed confirm subset and foreign-screen identity rejects.

Build command, for each toolkit:

```text
python3 /root/evidence-ada-f32-tf32-toolkit-20260907/run.py build <12.8|13.0> final3
```

Both final3 builds passed focused tests 9/9 and the full nonignored performance
test binary, with exit 0. Frozen binaries:

| Toolkit | Executable SHA-256 |
|---|---|
| 12.8 | `e8c8197e911cc191e565298a9798ea9726ac4843863c591a865d369604bd1cc1` |
| 13.0 | `f89af87854f6a993dd29ccbd00e78d55f4b0b5c7a6d09c20043e6d11656fe65f` |

## Functional smoke attempts

The immutable outer wrapper command form is:

```text
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py TOOLKIT ATTEMPT FAMILY smoke1 LITERALS
```

CUDA 12.8 exact F32 initial smoke (`cuda128-exact-smoke1`) had a quiet PRE,
then exit 101 before identity because a debug rendering of the compiler target
was compared to `Sm89`. Actual debug output was an internal `CudaTarget`
byte representation. Raw output is preserved; this was fixed to the stable
public `as_str()=="sm_89"` identity, then both toolkits were rebuilt as final3.

CUDA 12.8 exact F32 final3 smoke (`cuda128-exact-smoke2`) passed test, POST,
wrapper, and outer SSH with exit 0. JSONL SHA is
`aeb401cb872f4f42934d8e1825f0f9cb68d01671b9c116a9d59cfb4643938af1`:
626 records, 8 physical, 32 configurations, 384 raw, 96 pairs, 96 summaries.
The analyzer and an independent root replay recomputed all schedules,
chronology, bracket arithmetic and directions. Source, binary, Task6A fixed
source/invocation/artifact, telemetry, copied raw hashes, wrapper and outer
receipt all match. Smoke1 is functional evidence only; its one-window values
are not admission.

CUDA 12.8 TF32 final3 smoke (`cuda128-tf32-smoke1`) is a preserved **real
functional failure caused by a brief/audit conflict with current production**.
PRE at `2026-09-07T05:48:20Z` was strict quiet. Public
AUTO returned `Tf32RnaM128N128S3`, not the brief-required incumbent
`Tf32M128S2`; the test exited 101 immediately after its identity record. POST
had the same device and no compute apps; outer exit is 1. Raw JSONL SHA:
`9789a911a25bbd746142854a0522122f07c48fce81c0ca3152f642be128acd3a`.
No retry was made. Root traced this to `fixed_sm89_rna_wide_auto_eligible`:
current revision-43 production deliberately admits aligned A--E, including C,
on CUDA 12.8/13.0/13.2 and `fixed_forward` selects RNA before
`fixed_pick_tf32`. This is not a runtime regression. Per root ruling, the
harness will not disable RNA or alter operands to evade it. TF32 is held pending
corrected scope; no TF32 retry or screen is authorized.

CUDA 13.0 exact F32 final3 smoke (`cuda130-exact-smoke1`) passed test, POST,
wrapper and outer SSH with exit 0 after strict quiet PRE at
`2026-09-07T05:51:21Z`. Independent analyzer result is valid. JSONL SHA is
`1f483baafe5cba52da5279f8af1a0d96b6a29c19afbe98d02aa806c553a5e70a`;
test-log SHA is
`62f9009d04ad8ced7b46892e9e4f90684af7cb91694270f9717f377a64b95f08`;
outer transcript SHA is
`090d642731ed5d2aff50556cfa1078f59cca4837047db67ef8d9cbea032e9843`.
Its one-window values are functional only. CUDA 13.0 TF32 smoke is held and will
not conceal or supersede the 12.8 TF32 failure.

Corrected final4 functional smokes all used fresh strict quiet PRE and passed
test, POST identity/no-apps, wrapper, outer SSH, full record closure and the
independent analyzer. Exact artifacts each contain 626 records (8 physical,
32 configs, 384 raw, 96 pairs, 96 summaries); TF32 artifacts each contain 158
records (2 physical, 8 configs, 96 raw, 24 pairs, 24 summaries). One-window
results are not used for admission.

| Toolkit/family | Attempt | JSONL SHA-256 | SSH SHA-256 | Analyzer SHA-256 |
|---|---|---|---|---|
| 12.8 exact | `cuda128-final4-exact-smoke` | `386d20d1abcad6809557e519e7efdd4552d6a725fda45ad1ddbfe3583bb47a1d` | `ccbcc119c064cd5b94ca11978ee7ef36d75c67d851fd33604c35821e4960264a` | `52e1a8ba471d9c7fa5a4d29a81e441a4f6d787f68654cf7ed4237471f2ed5878` |
| 12.8 TF32 | `cuda128-final4-tf32-smoke` | `d6df0cac45a54c2cb17202b7b999cad43e82ea9008f09cbd5f3a278e53aa2d4c` | `599909f0a286ff641ef286e7bd8e7190959b04c9dc3615763c11270fd5de8925` | `f0b272efed4b0fd958056d3dc22da5e7cdff75235417c5dfa1fc02cc08735f9c` |
| 13.0 exact | `cuda130-final4-exact-smoke` | `3b364cdb6403737e212b384d5c244236f5b0835fdfa68bb08e8ae856622e9282` | `64565c71b4ea2753717101786d35c32d56be2807893b923b48064d866b60d9cb` | `1e7afa23227a2bd147c9226e3fd2ded1deb197898ec9e4f29e502bab0e823d65` |
| 13.0 TF32 | `cuda130-final4-tf32-smoke` | `c26440cbf45653350f88785345656469c1fcff8529a853cb3bb8a2f581ae6ac8` | `b1e524d1799c2dabe4be613b9663f1d9b2154c9367643073be96f52f9f321948` | `126f919d24617c0dc6afd43de2a78dfc84ce671591db558acc929efdbd486fe7` |

Final4 binding-file SHAs are
`ef42381452117527de8352c7a5102b8a807db737c61b8f06b2e954459b0699f7`
(12.8) and
`b6a94cc7c3479ff644becc25b864c97e844872051c6fd14e8c0ebf6738eacd72`
(13.0).

## Physical-proof boundary

Known custom symbols use the exact pointer/bundle/terminal ABI contracts in the
brief: Legacy 12 args, CopyPlan compact 5 args/eight words, and TF32 compact 5
args/24-byte bundle. ABI widths/counts are checked before captured-value
dereferences. Every one-node and every node of each 20-node capture checks
symbol, geometry, shared memory, ABI, pointers, bundles, and terminal rejection;
each 20-node pointer vector equals its own one-node capture, including C, while
cross-arm A/B/bias inputs match.

For cuBLAS, the implementation preserves actual captured kernel-node inventory
and geometry, bias-work inclusion, requested/queried compute/math/pointer/
atomics modes, and numerical/repeat/overwrite gates through the existing vendor
graph helper. It deliberately does not invent private cuBLAS argument layouts.

Guarded A/B/bias/C, complete input snapshots, poison upload/readback, every-word
poison difference, repeated eager/graph, single-term bias orientation, finite
ordering, exact bits (custom) and repeat bits (Fast) precede timing. The empty
graph is actually launched and must fail specifically because output remains
unchanged poison; arbitrary launch/driver errors are not accepted.

## Accepted final5b functional gate

The fix-only re-review accepted all three Important fixes and reported no new
Critical/Important finding. Root independently matched both 357-input source
bindings, tuning/numeric/schedule revisions 43/5/8, physical contracts, raw
arithmetic, and wrapper/outer closure for all four final5b smokes. Each exact
smoke has 626 records; each TF32 smoke has 158. All test, POST, wrapper, outer
SSH, and independent analyzer exits are 0.

| Toolkit/family | Final5b smoke JSONL SHA-256 |
|---|---|
| 12.8 exact | `dc87b9e2f2a9e41ec7923cf37100fe641fc6a50f5aa288fb481ba46b91174261` |
| 12.8 TF32 | `875155cbf0a8cf1387223473f9dc04cf7fb25195ba1d903917aa3761aabf623d` |
| 13.0 exact | `7b66bcd85135f74219ad96363f6ab89cf4e307b34787e8a873f870b20cb89aa7` |
| 13.0 TF32 | `ca7854c0a041fa682cbb3ad5a140cc3547c6142813f2fdf378ccfcd2c6b8a15b` |

## Timing commands and record closure

The exact literal controls were:

```text
EXACT=hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1
TF32=hot_c:0,hot_c:1
```

The six acquisition commands below were run once, in order, with no failed
timing attempt, rescreen, or retry. `remote.py` invoked the sole per-toolkit
environment constructor in `run.py`; every command ended with test/POST/
wrapper/outer SSH exit 0.

```text
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 12.8 final5b-exact-screen21 f32_exact_fast screen21 hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 12.8 final5b-tf32-screen21 tf32 screen21 hot_c:0,hot_c:1
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 13.0 final5b-exact-screen21 f32_exact_fast screen21 hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 13.0 final5b-tf32-screen21 tf32 screen21 hot_c:0,hot_c:1
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 12.8 final5b-exact-confirm101 f32_exact_fast confirm101 hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1 /root/evidence-ada-f32-tf32-toolkit-20260907/cuda128-final5b-exact-screen21/records.jsonl
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/remote.py 13.0 final5b-exact-confirm101 f32_exact_fast confirm101 hot_a:0,hot_a:1,hot_b:0,hot_b:1,hot_d:0,hot_d:1,hot_e:0,hot_e:1 /root/evidence-ada-f32-tf32-toolkit-20260907/cuda130-final5b-exact-screen21/records.jsonl
```

Every mirrored run was independently analyzed with:

```text
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py RUN_DIR TOOLKIT_BINDING [SCREEN_RECORDS_JSONL]
```

Confirm analysis supplied the matching local screen JSONL. Thus runner and
analyzer each parsed the whole shared screen, recomputed its eligible subset,
and checked source/binary/toolkit/artifact identity before accepting 101.

| Toolkit/family/stage | Lines | Physical | Raw | Pairs | Summaries | Configs | JSONL SHA-256 | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 12.8 exact screen21 | 10226 | 8 | 8064 | 2016 | 96 | 32 | `d43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf` | all 8 |
| 12.8 TF32 screen21 | 2558 | 2 | 2016 | 504 | 24 | 8 | `ce6385b523633841925c14471725d2cebd2a9ded13e54bb778f13d53e135a973` | empty |
| 13.0 exact screen21 | 10226 | 8 | 8064 | 2016 | 96 | 32 | `cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682` | all 8 |
| 13.0 TF32 screen21 | 2558 | 2 | 2016 | 504 | 24 | 8 | `0c639d97640af56e1826e3752b11ced0fb107ac6592a75faad6fadcc7855425a` | empty |
| 12.8 exact confirm101 | 48626 | 8 | 38784 | 9696 | 96 | 32 | `eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96` | all 8 |
| 13.0 exact confirm101 | 48626 | 8 | 38784 | 9696 | 96 | 32 | `0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711` | all 8 |

TF32's valid screen losses made both eligible subsets empty, so the binding
protocol forbade TF32 confirm101. The two exact confirms use the same frozen
source, per-toolkit binary, cache, Fixed artifact, and toolkit21 screen as the
corresponding screen.

## Complete 20-literal decision matrix

Ratios below are candidate/reference and show the worst p50 and p95 among the
four required strata (`eager/start0`, `eager/start1`, `graph/start0`,
`graph/start1`) at the decisive stage. Owner admission is solely against the
actual AUTO incumbent. The separate Fast column is never used to relabel an
owner win.

| Toolkit | Family/literal | Decisive stage | Owner verdict | worst candidate/AUTO p50/p95 | worst candidate/Fast p50/p95 |
|---|---|---|---|---:|---:|
| 12.8 | exact `hot_a:0` | confirm101 | CONFIRMED WIN | 0.843007/0.879217 | 1.834091/1.875126 |
| 12.8 | exact `hot_a:1` | confirm101 | CONFIRMED WIN | 0.842494/0.891455 | 1.504367/1.543210 |
| 12.8 | exact `hot_b:0` | confirm101 | CONFIRMED WIN | 0.817959/0.848895 | 2.198212/2.239683 |
| 12.8 | exact `hot_b:1` | confirm101 | CONFIRMED WIN | 0.817342/0.840282 | 1.929185/1.961281 |
| 12.8 | exact `hot_d:0` | confirm101 | CONFIRMED WIN | 0.827367/0.846293 | 2.053673/2.079278 |
| 12.8 | exact `hot_d:1` | confirm101 | CONFIRMED WIN | 0.828157/0.854705 | 1.807777/1.843069 |
| 12.8 | exact `hot_e:0` | confirm101 | CONFIRMED WIN | 0.814615/0.828924 | 2.193915/2.223561 |
| 12.8 | exact `hot_e:1` | confirm101 | CONFIRMED WIN | 0.815332/0.839916 | 2.069556/2.111143 |
| 12.8 | TF32 `hot_c:0` | screen21 | LOSS / no 101 | 1.514233/1.518228 | 2.182562/2.188560 |
| 12.8 | TF32 `hot_c:1` | screen21 | LOSS / no 101 | 1.515603/1.518609 | 1.995957/2.001284 |
| 13.0 | exact `hot_a:0` | confirm101 | CONFIRMED WIN | 0.844589/0.877076 | 1.837724/1.880154 |
| 13.0 | exact `hot_a:1` | confirm101 | CONFIRMED WIN | 0.842835/0.888977 | 1.505815/1.545571 |
| 13.0 | exact `hot_b:0` | confirm101 | CONFIRMED WIN | 0.826929/0.845739 | 2.197361/2.239997 |
| 13.0 | exact `hot_b:1` | confirm101 | CONFIRMED WIN | 0.815471/0.843010 | 1.926423/1.960697 |
| 13.0 | exact `hot_d:0` | confirm101 | CONFIRMED WIN | 0.826978/0.845673 | 2.052521/2.078723 |
| 13.0 | exact `hot_d:1` | confirm101 | CONFIRMED WIN | 0.829139/0.856866 | 1.804529/1.832843 |
| 13.0 | exact `hot_e:0` | confirm101 | CONFIRMED WIN | 0.814324/0.830633 | 2.197623/2.222198 |
| 13.0 | exact `hot_e:1` | confirm101 | CONFIRMED WIN | 0.815227/0.836546 | 2.069773/2.104876 |
| 13.0 | TF32 `hot_c:0` | screen21 | LOSS / no 101 | 1.513534/1.518339 | 2.183314/2.189621 |
| 13.0 | TF32 `hot_c:1` | screen21 | LOSS / no 101 | 1.515374/1.518566 | 1.995163/2.000520 |

The machine-readable 20-row matrix is
`internal/perf/ada-f32-tf32-toolkit-20260907/final-matrix.json`, SHA-256
`353f15d37d7e39f9720b612ea87d152b55b26f93521e29cacb15e701c847d53b`.
Every p50/p95 constituent for all three directions and all four strata from
both exact screen21 and confirm101 stages and both TF32 screen21 stages is in
`final-constituents.json`, SHA-256
`d0bef5e7779b1f1f0e649d817f04ca1ee5e8e40aeee1b19baea695c0ce87596a`.
That file declares the stratum order explicitly; it contains 36 literal-stage
records and 432 `(p50,p95)` constituent pairs. No constituent is omitted,
discarded, or replaced. Raw JSONL remains the authoritative source.

## Artifacts, archives, and release

Final5b source/binary bindings are:

| Toolkit | Binding SHA-256 | Binary SHA-256 |
|---|---|---|
| 12.8 | `243778b249c06534329f2bba18775adc3ef661d0996aef87ae070fbd4f61e2fd` | `0f3f2e617797fcce4e70789cc507cd25704429f98663812013d6df9851321bb5` |
| 13.0 | `eae1bd397e66a01fd5433b8e74163c56f7542af503aec3e6a514326260ff6f4b` | `cf90b10de3cb8afb0a4a68575a1c9db9597750ff3b929dfff85864a423009154` |

The source archive contains exactly all 357 regular inputs and every hash
matches both bindings. Each runtime archive contains the exact bound binary
and the three used cache blobs; the cache directory records mode 0700.
`archive-verification.log` records the complete check and cache hashes with
`ARCHIVE_VERIFICATION=PASS`; `remote-archive-sha256.log` proves the remote and
local archive digests agree.

| Archive | SHA-256 |
|---|---|
| `archives/source-final5b.tar.gz` | `3fb01ada00920caff7a40e1209f34258ca8574f75e2beca36c48df03cd3f47c7` |
| `archives/cuda128-final5b-binary-cache.tar.gz` | `f4a2cd502b93974771c13e3ff6d09d77e4aea4f7a1b453d8f43446e0d818cc54` |
| `archives/cuda130-final5b-binary-cache.tar.gz` | `70a71408f7ed6fc20dd6bb43f7a135160d46e006614885c3d2af999a0c05cd02` |

Final release telemetry is
`internal/perf/ada-f32-tf32-toolkit-20260907/final-release.log`, SHA-256
`306502eb09a995c5eb7fe143bd9e70f901b2fadadeade19e7f22a83041795a37`.
At `2026-09-07T06:56:16Z` it records the exact RTX 6000 Ada UUID, CC 8.9,
0% GPU, 0% memory, no compute apps, and both query exits 0. The immutable
rooted file manifest is stored in the same evidence directory.

## Release recommendation

Task7 qualifies existing exact-F32 CopyPlan as a stable performance owner win
against the real Legacy AUTO incumbent for A/B/D/E on CUDA 12.8 and 13.0.
It does **not** establish a Fast win: candidate/Fast exceeds 1 in every
reported exact constituent. Existing TF32 M64S2 is decisively slower than the
real aligned RNA AUTO incumbent for C on both toolkits, so it is not eligible.
The evidence therefore supports retaining the existing production routing,
making no selector promotion in this task, and carrying the exact owner/Fast
distinction into any later release decision.
