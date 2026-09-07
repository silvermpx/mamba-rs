# Task6B final evidence gate

## Spec compliance

**PASS for the completed evidence gate.** The six-cell result supports only CUDA 13.2 homogeneous BF16 and F16 for literal B0 `(M,K,N)=(4621,768,2304)`, bias=false. CUDA 12.8/13.0 retain their existing choices. This is an own-improvement result against actual AUTO42 Swizzle, not a vendor win or an AUTO implementation review.

**Separate source gate resolved:** the completed sibling `ada-half-s3-paired-task6b-fix1-review.md` marks I1 ADDRESSED with no new breakage. I read that verdict and the resulting revalidation evidence, without duplicating the I1 implementation review. There is no remaining source-fix condition on this evidence approval. Prior Minor M1/M2 remain explicitly deferred by root to immediate Task6C/final branch review (`final-report.md:203`); this evidence verdict does not claim they were fixed.

All paths below are under `internal/perf/ada-half-s3-paired-20260907/` unless otherwise qualified.

## Decisions and all-stratum closure

| Toolkit | Dtype | Screen worst p50 / p95 | Fresh confirm worst p50 / p95 | Evidence decision |
|---|---|---|---|---|
| 12.8 | BF16 | 0.983487847 / 1.006867571 | Ineligible | Retain incumbent |
| 12.8 | F16 | 0.981792374 / 0.997414724 | 0.981555906 / 1.006347845 | Retain incumbent |
| 13.0 | BF16 | 0.983097932 / 1.011073358 | Ineligible | Retain incumbent |
| 13.0 | F16 | 0.981527365 / 0.994274018 | 0.981151367 / 1.015680726 | Retain incumbent |
| 13.2 | BF16 | 0.930532163 / 0.951183161 | 0.929639720 / 0.945012679 | Confirmed own win |
| 13.2 | F16 | 0.929007877 / 0.955216865 | 0.927334868 / 0.958864521 | Confirmed own win |

- `winner-matrix.json:7`, `:196`, `:487`, `:658`, `:967`, `:1239` and `final-report.md:28`: every cell retains all eager/graph × starting-parity 0/1 constituents. Both own p50 and p95 must be below one in every stratum. The BF16 12.8/13.0 screen losses are retained; only their F16 cells advanced, and both failed fresh confirm p95. No threshold relaxation, omitted adverse stratum, or reciprocal-p95 shortcut appears in the decision.
- `fix1-revalidation.log:5` through its final record and `matrix.py:14`: screens cover both dtypes on each toolkit; confirms contain exactly F16 for 12.8/13.0 and BF16/F16 for 13.2. All three screens finished by 03:18:30Z; the first fresh confirm began 03:19:33Z (`cuda128-confirm101/pre.json:1`). Screens each close 8 configurations/2,016 samples/504 pairs; confirms close 4/4,848/1,212 for 12.8 and 13.0, and 8/9,696/2,424 for 13.2.
- `root-recomputed.json:1`: root's separate arithmetic/chronology output covers all nine final smokes/screens/confirms. I compared every stored recomputed summary, including both Fast directions, to its derived analysis and checked all nine recorded raw SHA256s against the actual logs; all matched. I did not rerun the arithmetic checker or analyzer.
- `final-report.md:57` and each matrix `vendor_context`: S3/Fast and AUTO/Fast remain independently reported from their own brackets. Fast is faster in all six final aggregate comparisons; this does not veto the two confirmed own wins.

## Identity, physical and functional evidence

- `cuda128-binding-final.json:1`, `cuda130-binding-final.json:1`, `cuda132-binding-final.json:1`: all bind 356 inputs, measured Rust SHA256 `f43a2a22dea7435716ff0c99d721e1d255ad9311f18e9f0b53a15280eafb582b`, matching toolkit features/compiler/library paths and hashes, distinct per-toolkit binaries, and isolated 0700 caches. Actual binaries are `618a31de…90233b3`, `57deaa66…50b63062`, and `8edb3d6f…dde7f354`, respectively.
- `archive-verification.json:1` and `verify_artifacts.py:17`: the archived 356-file source inventory equals every measured binding; only the performance test differs from the approved Task6A source archive. Each archived executable matches its binary hash, and the three Fixed PTX payload hashes match Task6A: `60977db3…12352c`, `d1aa6e33…edb86a9`, `8b89aadf…41aff1`. Root independently verified the archive payloads; I reviewed that output and the complete new utility source, and verified the archive file hashes through the rooted manifest rather than repeating root's payload scan.
- For all nine final raw logs, I directly compared the identity record's source/binary hash to its final binding and the six production compiler/artifact identity fields to the corresponding Task6A identity JSON. Every comparison matched. Thus the production-resource reuse at `final-report.md:188` is explicitly tied to byte-identical Task6A artifacts, not a fresh resource measurement or a standalone NVCC experiment.
- Named physical-evidence checks: `cuda128-smokefinal/test.log:4`, `cuda130-smokefinal/test.log:4`, and `cuda132-confirm101/test.log:4` through their physical/configuration records. These retain distinct guarded C pointers with shared A/B, 256-byte guards and aligned interiors, actual Swizzle/S3 symbols, grid 666×1×1, block 256×1×1, dynamic shared 69,632/98,304 bytes, exact five-parameter ABI and decoded bundle `[1065353216,0,4621,2304,768,768,2304,2304]`. Each custom 20-op graph inventory has 20 copies matching its inspected one-op node.
- The same raw records show complement-upload verification, stable eager/graph repeats, intact guards, immutable-input/post gates, and no-op rejection. BF16 normalized error is approximately 0.0024206022 against tolerance 0.01; F16 is approximately 0.0003037254 against 0.0025. Vendor output has its separately recorded numerical/repeat gate. Actual vendor nodes are native BF16/FP16 Ampere GEMMs, with compute-32F, DEFAULT_MATH, HOST pointer mode, and ATOMICS_NOT_ALLOWED; PEDANTIC_F32 is labeled only as reference (`cuda132-confirm101/test.log:4`, `:7`, `:6086`).

## Exit, history, release and archive closure

- `all-buildfinal-ssh.log:2` through `:37` closes all three builds and outer SSH with zero exits. Per-toolkit focused outputs report 3 passed; all nonignored outputs report 51 passed/64 ignored at line 121. No warning/error matches were found in those focused/nonignored outputs.
- `fix1-green.log:1` records 11 passing host groups; `fix1-revalidation.log:1` records all ten actual runs valid under the exact-same-attempt analyzer, including the separately preserved historical smoke. I inspected the complete actual 13.2 confirm SSH transcript (`cuda132-confirm101-ssh.log:1`), which contains matching PRE, executed binary, command exit, POST, RUN_RESULT and zero wrapper/outer exits. The separate source review owns proof that the corrected parser enforces this for all inputs.
- The saved PRE/POST JSON for every screen and confirm shows exact UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, correct identity, no applications and successful telemetry queries. PRE is always 0%/0%; POST residual utilization is correctly retained as valid. `lane-release.log:1` records 2026-09-07T03:28:29Z, the same UUID/CC8.9, 0% GPU/memory, no apps and zero identity/no-apps/outer-SSH exits.
- `SHA256SUMS:1` inventory, `final-report.md:242`, `red.log:71`, `disabled-stale-negative-full.log:4`, and `fix1-red.log:1`: original source/binaries, prior derived results, intentional negative tests and preparation failures remain preserved. There is one screen and one eligible confirm per toolkit, with the four adverse decision cells retained as valid performance evidence. No repeated screen/confirm is present in the archived run inventory. Initial smoke and stderr-preserving host-negative repetition are explicitly distinguished from performance screening.
- Reviewed final report hash: `a27dec8e58f16a4720525b70fe74cd2df4fd49d6437130e81409eb11a7c4b1c2`. Reviewed rooted manifest hash: `710391bba6854434ea2340332413edcd57f98e8218321a17a208553fe2d976a5`. These supersede the initial dispatch snapshot after the controller-confirmed prose refresh. Local `shasum -a 256 -c` completed with **123 entries, zero failures**. Matrix remains `f95624e34a5646ccee75647a4bd515d2def73f4e0bd60d4b471bdce5a0cf6eb4`; root recomputation remains `fb3f5716b5792012ebe71cef42433701946629283589293a9146979e5917c7eb`.

## Quality verdict and findings

**Evidence quality: APPROVED. The separate I1 source-fix gate is also approved. No new Critical, Important or Minor findings in this evidence scope.**

The new `matrix.py` cleanly consumes validated screen/confirm outputs, restricts confirms to eligible dtypes, preserves failed strata/raw hashes, and keeps vendor context independent. `verify_artifacts.py` checks the exact archived input inventory, actual executables and Task6A Fixed payload linkage without extracting files into the checkout. Both complete utility diffs and their package hashes were reviewed.

Review was local/read-only except for this report. No SSH/GPU/build/test-suite reruns, source/index/branch changes, or subagents. One ad-hoc read-only JSON comparison initially used `filter_map`, unavailable in the installed older Ruby; replacing it with `map.compact` completed all nine comparisons. This was a reviewer tooling compatibility issue, not an evidence or implementation failure.
