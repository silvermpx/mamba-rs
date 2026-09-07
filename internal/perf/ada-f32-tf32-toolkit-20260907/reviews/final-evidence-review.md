# Task7 final evidence acceptance — independent bounded review

Date: 2026-09-07

Scope: only the final5b evidence accumulated after the accepted fix1 source
review. I did not rerun a GPU measurement, SSH command, build, Cargo command,
or owner test suite, and I did not repeat the full source review. I performed
local read-only hashing, archive parsing, receipt/record inspection, and an
independent raw-arithmetic recomputation. The implementation, index, branch,
and HEAD were not changed.

## Verdicts

- **Spec compliance: PASS.** The frozen package supports all 16 exact-F32
  toolkit-literal owner wins and all four TF32 owner losses under the binding
  admission rule. Only the exact literals may proceed to a separately
  authorized promotion task; TF32 remains on the existing RNA AUTO route.
- **Task quality: PASS WITH MINOR.** The evidence is complete, internally
  consistent, reproducible from the preserved raw records, and narrowly
  reported. The already-disclosed unused-import warning remains the only
  quality defect I found.
- **Acceptance recommendation: ACCEPT.** I found no Critical or Important
  defect in the evidence added after the accepted final5b source gate.

## Findings

### Critical

None.

### Important

None.

### Minor

1. **Pre-existing source-gate warning, not a new evidence defect:**
   `NUMERIC_ABI_REVISION` and `SCHEDULE_REVISION` are imported but unused in
   `tests/gemm_bi_fixed_performance.rs`. Both final5b focused and nonignored
   build logs record the warning, for example
   `internal/perf/ada-f32-tf32-toolkit-20260907/cuda128-final5b-build/focused.log:57`
   and the corresponding CUDA 13.0 log. The accepted fix1 review already
   classified this below Important. The runtime identity records and analyzer
   use the actual compiler revision fields and require literal values 5 and 8,
   so the warning does not invalidate this package.

There are no new findings attributable to the post-source-review evidence.

## Frozen review inputs

- Main report:
  `.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-f32-tf32-toolkit-task7-report.md`,
  SHA-256
  `b15ac946a4171b656605eb4aebd78569da8b144ea5ce727000f34bea240a4892`.
- Fix1 report:
  `.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-f32-tf32-toolkit-task7-fix1-report.md`,
  SHA-256
  `996d254fb5b9118132db1b551f177b01e17593c107db77acc7bf24e06f22a374`.
- Accepted fix1 review: adjacent
  `ada-f32-tf32-toolkit-task7-fix1-review.md`, SHA-256
  `57607305bea1f22e79794af490be407f79fe98578cfaf4671f32144f95206545`.
- Evidence root: `internal/perf/ada-f32-tf32-toolkit-20260907/`.
- Composite timing source:
  `97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7`.
- Final5b bindings:
  `cuda128-binding-final5b.json` SHA-256
  `243778b249c06534329f2bba18775adc3ef661d0996aef87ae070fbd4f61e2fd`
  and `cuda130-binding-final5b.json` SHA-256
  `eae1bd397e66a01fd5433b8e74163c56f7542af503aec3e6a514326260ff6f4b`.
- Bound executables: CUDA 12.8
  `0f3f2e617797fcce4e70789cc507cd25704429f98663812013d6df9851321bb5`
  and CUDA 13.0
  `cf90b10de3cb8afb0a4a68575a1c9db9597750ff3b929dfff85864a423009154`.

The supplied report and fix1-report hashes match the frozen READY message.

## Independent timing recomputation

I parsed all six timing JSONLs independently of the saved analyzer output. For
each file I checked newline termination, the terminal prefix digest, the exact
record-kind/cardinality closure, positive finite samples, every raw chronology
field, the alternating comparison traversal, ABBA/BAAB arm order, four-sample
bracket indices, and all three B/A directions. I recomputed each pair from the
two sums and selected p50/p95 at
`round((W - 1) * 0.50)` and `round((W - 1) * 0.95)` after sorting. Every stored
pair and summary matched within `1e-12`; every literal decision then matched
the rule `candidate/AUTO p50 AND p95 < 1` in all four ordered strata
`eager/start0`, `eager/start1`, `graph/start0`, `graph/start1`.

Authoritative raw inputs and verified SHA-256 values:

| Attempt | Records path | SHA-256 | Recomputed eligible subset |
| --- | --- | --- | --- |
| CUDA 12.8 exact screen21 | `cuda128-final5b-exact-screen21/records.jsonl` | `d43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf` | all eight exact literals |
| CUDA 12.8 TF32 screen21 | `cuda128-final5b-tf32-screen21/records.jsonl` | `ce6385b523633841925c14471725d2cebd2a9ded13e54bb778f13d53e135a973` | empty |
| CUDA 13.0 exact screen21 | `cuda130-final5b-exact-screen21/records.jsonl` | `cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682` | all eight exact literals |
| CUDA 13.0 TF32 screen21 | `cuda130-final5b-tf32-screen21/records.jsonl` | `0c639d97640af56e1826e3752b11ced0fb107ac6592a75faad6fadcc7855425a` | empty |
| CUDA 12.8 exact confirm101 | `cuda128-final5b-exact-confirm101/records.jsonl` | `eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96` | all eight requested literals |
| CUDA 13.0 exact confirm101 | `cuda130-final5b-exact-confirm101/records.jsonl` | `0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711` | all eight requested literals |

The two confirms request exactly their recomputed same-toolkit screen subset.
Each confirm identity binds the whole matching screen digest and the matching
Fixed artifact digest, and source, binary, compiler target, toolkit, device,
numeric/schedule/tuning revisions, Fixed source/invocation/artifact, header
manifest, and NVRTC library domain are unchanged from its screen.

The independent recomputation also matches every one of the 432 constituent
`(p50,p95)` pairs in `final-constituents.json` (SHA-256
`d0bef5e7779b1f1f0e649d817f04ca1ee5e8e40aeee1b19baea695c0ce87596a`)
and every derived field in the 20 rows of `final-matrix.json` (SHA-256
`353f15d37d7e39f9720b612ea87d152b55b26f93521e29cacb15e701c847d53b`).
At confirm101, exact candidate/AUTO p50 ranges are
`0.812591..0.843007` (12.8) and `0.812153..0.844589` (13.0), while p95 ranges
are `0.824435..0.891455` and `0.827479..0.888977`; all required constituents
are therefore below one. Every exact candidate/Fast constituent remains above
one (minimum p50/p95 `1.488896/1.523547` on 12.8 and
`1.491581/1.526982` on 13.0). At screen21 every TF32 candidate/AUTO
constituent is above one (minimum p50/p95 `1.504418/1.510040` on 12.8 and
`1.499889/1.502600` on 13.0), so no TF32 confirm was permitted or present.

This establishes 16 exact toolkit-literal owner wins, not Fast wins, and four
TF32 owner losses.

## Functional, physical, and receipt gates

I checked all ten final5b attempt directories: four smoke1, four screen21, and
two exact confirm101. The four smoke JSONLs have the reported hashes:

- CUDA 12.8 exact `dc87b9e2f2a9e41ec7923cf37100fe641fc6a50f5aa288fb481ba46b91174261`;
- CUDA 12.8 TF32 `875155cbf0a8cf1387223473f9dc04cf7fb25195ba1d903917aa3761aabf623d`;
- CUDA 13.0 exact `7b66bcd85135f74219ad96363f6ab89cf4e307b34787e8a873f870b20cb89aa7`;
- CUDA 13.0 TF32 `ca7854c0a041fa682cbb3ad5a140cc3547c6142813f2fdf378ccfcd2c6b8a15b`.

For every final5b attempt, the result's JSONL and test-log digests match the
files, the test and POST exits are zero, the wrapper has one successful command
receipt and one completion marker, the outer transcript digest matches, and
the outer SSH exit is zero. The mirrored top-level and attempt-local SSH/outer
receipts are byte-identical. Each saved analyzer result is `valid: true` and
matches the attempt identity. Every PRE is the exact Ada UUID, CC 8.9/142 SM,
0% GPU, 0% memory, and no compute apps. Every POST has the same device and no
compute apps; the recorded residual utilization is allowed by the brief. The
final release is quiet/no-apps in `final-release.log`, SHA-256
`306502eb09a995c5eb7fe143bd9e70f901b2fadadeade19e7f22a83041795a37`.

Every physical record sets the custom-bit, Fast-repeat-bit, poison upload and
readback, empty-graph overwrite rejection, guard, immutable-input,
bias-orientation, and finite-ordering gates true. Direct inspection also
confirmed:

- exact public AUTO is `Legacy` and the candidate is
  `F32Sm89N64CopyPlan`; Legacy uses the expected 12-argument ABI and CopyPlan
  the compact five-argument/eight-word ABI;
- TF32 public AUTO is `Tf32RnaM128N128S3`, not the superseded portable
  incumbent, with grid 111/block 256/dynamic 98304 and the 32-byte bundle;
  `Tf32M64S2` uses grid 438/block 128/dynamic 32768 and the distinct 24-byte
  bundle;
- every known custom one-node contract equals every corresponding node in its
  20-operation graph, A/B/bias pointers agree across custom arms, both bias
  pointer states are present, and C is arm-local;
- Fast bias-false graphs contain one observable vendor kernel per logical
  operation; bias-true graphs include `bias_broadcast` plus the GEMM per
  logical operation. Only observable graph inventory/geometry and the public
  compute, math, algorithm, host-pointer, and atomics modes are claimed. No
  private cuBLAS parameter ABI is inferred.

The final5b build logs record 12/12 focused and 65 passed + 65 ignored
nonignored-target results on each toolkit, apart from the disclosed unused
imports. Numeric ABI 5, schedule 8, tuning 43, exact device/compiler/library
identities, and the per-toolkit binary are present in every final5b identity.
The Fixed source digest is common across toolkits:
`205f58b56429b8e74f3ac1e7ab9f0cf6a9bf193ffeb87b254bd3ab00e6b02ca5`.
Within each toolkit, smoke/screen/confirm retain the same Fixed invocation and
artifact: 12.8 uses invocation
`57c3e4f2177104f137d432f19398a4a36799aca4c264634a1ab25d17dbf422ba`
and artifact
`60977db33de28d807ac7dd3dafe2d176914f7b7eab589917a47988050712352c`;
13.0 uses invocation
`adee1f8b255bacfce8921820d9397a0a22a0dbaac735759fdcd222a453176155`
and artifact
`d1aa6e33a612d99cd44a1e9c7eebe495c05eeef2212bf2cf4296db55eedb86a9`.
The corresponding cache payload digests match those Fixed artifacts.

## Manifest and archive closure

`manifest-root.json`, SHA-256
`d3af59584bcf0fc6746ef79c9b8fc5451ce85a4aaa94be6c8a7df1ba346744e4`,
binds the 243-entry `manifest-files.sha256`, SHA-256
`95c8356fa8546ce343c3946e9ab9fd1f4432ca6999cc18ab027362803d69c537`.
An independent `sha256sum -c` of all 243 entries passed, including the frozen
main/fix/review ledgers, raw records and receipts, matrix/constituents, bindings,
archives, final verification, and release log.

I parsed the archives without extraction:

- `archives/source-final5b.tar.gz`, SHA-256
  `3fb01ada00920caff7a40e1209f34258ca8574f75e2beca36c48df03cd3f47c7`,
  contains exactly the 357 regular paths in both binding input maps. The two
  maps are identical, and every archive payload and current working-copy file
  matches its bound digest.
- `archives/cuda128-final5b-binary-cache.tar.gz`, SHA-256
  `f4a2cd502b93974771c13e3ff6d09d77e4aea4f7a1b453d8f43446e0d818cc54`,
  contains the exact 12.8 binary and three cache files.
- `archives/cuda130-final5b-binary-cache.tar.gz`, SHA-256
  `70a71408f7ed6fc20dd6bb43f7a135160d46e006614885c3d2af999a0c05cd02`,
  contains the exact 13.0 binary and three cache files.

Both archived cache directories retain mode 0700. For all six cache envelopes,
the GNU long filename's key equals the embedded compile key, magic/version/kind
are valid, payload length is exact, and the embedded payload SHA-256 matches
the payload. The Fixed-key payloads match the Fixed artifact identity recorded
by every attempt. `archive-verification.log` also reports
`ARCHIVE_VERIFICATION=PASS`; `remote-archive-sha256.log` records equal remote
and local tar digests.

`final-local-verification.log`, SHA-256
`8a65ddfaa87b8a3e78548e350cad17fa1714bb23acc76fdef9be3c1cc1c81e59`,
records successful formatting/diff checks, the 9/9 validation suite, and saved
analyzer runs for all four screens and both confirms, with every command and
the final wrapper at exit zero. These saved owner/root checks are corroboration;
the raw recomputation above is this review's independent evidence judgment.

## Scope of the recommendation

The package recommends no Task7 production, selector, numeric, schedule,
kernel, loader, or holder change. Its only positive release candidate set is
the 16 exact A/B/D/E x bias0/1 x CUDA12.8/13.0 toolkit-literals, for a separate
reviewed promotion task with the required new epoch and post-AUTO evidence.
The four TF32 cells remain on existing RNA AUTO. The evidence does not support
an all-inference, all-precision, all-architecture, or Fast-victory claim.

The main report preserves the original failed TF32/audit assumption, the
controller's RNA correction before any TF32 screen, the final3/final4 and
pre-format final5 source identities as non-timing history, the accepted final5b
source gate, and all genuine losses. I found no suppression or relabeling of a
valid loss.

## Cannot independently verify

- This local read-only review cannot prove the negative external facts that no
  unmirrored Ada attempt, concurrent user, or byte-identical rebuild existed.
  The local evidence contains exactly the four screen21 and two confirm101
  timing directories in the reported chronological order, with no failed
  timing directory, and every source/binary/cache/compiled-artifact identity
  is stable across each relevant screen/confirm pair. The exact PRE/POST/outer
  receipts and final quiet release support the owner's exclusivity statement.
- I did not independently execute the GPU, builds, SSH wrappers, or owner
  suites because the review brief forbids those actions. Their preserved logs
  and receipts were checked, and the raw timing arithmetic was recomputed
  locally from the authoritative JSONLs.

These limits do not create a finding against the frozen package.
