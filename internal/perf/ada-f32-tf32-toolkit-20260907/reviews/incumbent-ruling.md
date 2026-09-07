# Task7 incumbent correction — controller ruling

2026-09-07. This corrects a source-audit/brief defect before any21 or101
screen. It is not a new kernel task and does not authorize disabling any
existing production route or changing operands to avoid an admitted route.

## Evidence and cause

The original CUDA12.8 TF32 smoke correctly failed before timing: public AUTO
returned `Tf32RnaM128N128S3`, while the brief assumed `Tf32M128S2`.
Original raw SHA9789a911a25bbd746142854a0522122f07c48fce81c0ca3152f642be128acd3a
and source checkpoint/report/package remain preserved; no performance claim
is taken from that incomplete run.

`fixed_forward` at gemm_bi_fixed.rs:3510 tests
`fixed_sm89_rna_wide_auto_eligible` before the ordinary TF32 picker. That
predicate at:3285 admits all A-E/no-bias+bias hot rows for known NVRTC12.8,
13.0 and13.2 with C/A/B16, an admitted holder and CC8.9/142SM. Therefore
the old picker's C M128S2 return is a fallback, not current aligned AUTO.
The earlier loader-gap note and Task7 audit missed that dispatch precedence.

Commit86fd7f56 already promoted all30 aligned toolkit TF32 cells. Root read
internal/perf/ada-rna-toolkit-auto-20260906/{README,integration-report}.md and
freshly verified its49-entry manifest, allPASS. The historical proof contains
actualAUTO101 C against oldM128S2 on12.8/13.0, not the M64 candidate proposed
by Task7. These are preserved historical results, not new current43 timing.

## Binding ruling

Ruling: keep Task7's20 literal toolkit cells, but compare TF32 M64S2 against
actual AUTO/RNA-wide on both12.8 and13.0. The public production launch is the
authority; bypassing RNA would measure a fabricated incumbent. This preserves
the user's best-kernel goal and records any genuine candidate loss. If the
candidate loses21, stop that literal without101. Never weaken the own-win rule
or call an already installed RNA route a new promotion. Cost: corrected host
tests, a new frozen source/binary and matching functional smoke; no21/101 is
discarded because none started. ExactF32 route/arithmetic/scope stays unchanged.

Correct TF32 C physical contracts:

| Arm | Actual tile/symbol | Grid/block/shared | ABI and bundle |
| --- | --- | --- | --- |
| AUTO | Tf32RnaM128N128S3 / gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3 | (111,1,1)/(256,1,1)/98304dynamic |5args: (0,8),(8,8),(16,8),(24,8),(32,32); words[1.0bits,0,4621,1928,384,1928,384,384]; sixth absent |
| candidate | Tf32M64S2 / gemm_bi_nn_tf32_v1_m64n64_bk32_s2 | (438,1,1)/(128,1,1)/32768dynamic |5args: (0,8),(8,8),(16,8),(24,8),(32,24); words[4621,1928,384,1928,384,384]; sixth absent |

Reuse the existing `fixed_explicit_vendor_rna_wide_graph_contract` and its
graph-inspection path where directly applicable; do not invent a second
numeric layout or infer that all five-argument kernels use the same bundle.
Inspect all1/20 nodes and full pointers/bundle/ABI as before. The TF32
single-active-term probe must be valid for RNA's force contract (K/N multiples
of4, e.g.2x4x4 with one nonzero reduction term), not the earlier2x1x3 probe.
Keep the full hot C16 input/output class, no alignment or holder bypass.

Implement the minimal test/runtime/analyzer expected-incumbent/ABI updates,
with meaningful old-incumbent and old-bundle RED negatives then GREEN. Root
reviews the corrected frozen package before screen21. Source kernels,
selectors, holders, epoch43/numeric5/schedule8 remain unchanged. Retain prior
exact smokes as valid historical source-checkpoint evidence; final smokes and
21/101 bind the new final source/binaries, with no rebuild midway between21/101.

Fix the report's current claim of TF32 missing aligned AUTO wiring; the actual
remaining fallback-view eligibility is a different domain, not an aligned-hot
gap. Root will correct the tracked handoff/gap-note interpretation at commit.
