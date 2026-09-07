# Bounded TN compact8/S2 discovery screen

Base:2fed713b. This is test-only discovery after the three-toolkit NT AUTO
integration, not a new production admission. Existing worktree/branch only.

## Probe

TN Prism logical(m,k,n)=(4621,384,1928), physical C384x1928, reduction4621.
Current actualAUTO is gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3,
grid93,256threads,79872shared. Historical plainM64S2 and paddedM128S2 lose;
do not resweep them. NCU attributes most excess shared transactions to wide
LDGSTS copy sites; compact packed staging is a hypothesis, not a proven win.

One candidate: M128N64/BK32/S2 packed TN A[2][32][128],B[2][32][64],
49152dynamicbytes,8compute warps,MAtoms2,NAtoms4,
warp_m=(warp>>1)*32,warp_n=(warp&1)*32. Exact template gate
Op==SgbTf32Tn && BM==128 && BN==64 && Stages==2.
Slot axis XOR ((reduction&3)<<3), preserving contiguous16B copy chunks.
Preserve RNA cvt, all per-output ascending K8 and outer reduction order,
cp.async commit/wait/barriers, zero-fill/tails and TN __fmaf_rn(alpha,acc,oldC).
Candidate owns a distinct test-only symbol; no production/kernel edits,
dispatch changes, source identity changes, new module holder or copied full
SM80 CUDA body. Reuse count-checked transformations and existing raw harness.

## Execution

GPU owner implement_ada_f32_tf32_toolkits owns edits to
tests/gemm_bi_tf32_nt_compact_xor.rs, all runtime harness and CUDA execution.
Source owner implement_ada_half_s3_auto owns ONLY
tests/support/triad_tn_compact_source.rs and
tests/gemm_bi_tf32_tn_compact_xor.cuh. Interface:
`pub fn candidate_source(production: &str) -> Result<String,String>` and
`pub const SYMBOL: &str`, with symbol
gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2.
CUDA helper: gemm_bi_tf32_tn_test_compact_xor_axis.
Neither writer edits the other's files. No subagents/commits/push.
Root owns this brief/overall report/commit; auditor reviews read-only.

1. Add focused native tests first: XOR bijection/alignment, tile ownership,
   exact-gated source changes, missing/duplicate anchors fail closed.
2. Extend existing raw NVRTC/ABI/guards/graph/QuietGpu harness minimally.
   Source/candidate/ABI/resource identity must be emitted.
3. Resource stop gate:0local/static,49152dynamic,256threads,occ>=2;
   record actualregs. No correctness/timing claim after failedresource gate.
4. Bits: fullmantissa target and tail(129,65,36), K0 TN oldC, alpha1 and
   alpha-0.75 on a smalltail; eager2+graph2, inputs/guards; exact portableRNA
   M128N64S3 reference for positive reduction (actualAUTO asserted for target).
   K0 uses the direct test-only candidate and an independent finite old-C
   oracle: __fmaf_rn(1,+0,oldC) preserves nonzero oldC bits and maps either
   signed zero to positive zero. The public physical
   qualification API deliberately rejects logical m=0; do not weaken it.
5. Paired once7 only on CUDA13.2: real AUTO vs candidate, eager and graph,
   both orders. TN beta1 means every one-GEMM observation must freshly
   restore identical C/A/B outside events. Repeated beta1 launches without
   reset are invalid. Reuse existing reseeded ABBA/BAAB sample mechanics;
   do not touch unrelated SM120 WIP or its untracked helper.
6. Build/list once, focused tests only, PRE/raw RELEASE/separatequietDRAIN,
   exactsource/binary hashes, privatecache reused honestly. No fullmatrix,
   no clocks/workload manipulation. Retain all raw records includingloss.
7. Report measured outcome and stop this candidate on robustloss/resource/
   correctness failure. No production admission or Fastwin from this screen.

Primary references consulted: NVIDIA PTX cp.async/wait_group semantics,
https://docs.nvidia.com/cuda/parallel-thread-execution/ and NVIDIA CUTLASS
https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/swizzle.hpp.
Only SM80-supported operations; no Ada TMA or SM90+ instructions.

Ruling: user explicitly requested autonomous fast candidate search, so this
bounded pre-presented experiment proceeds without another approval round.
Cost if hypothesis is wrong: one isolated short probe; production unaffected.

Ruling21:43Z: split source transformation and runtime harness into disjoint
files after the initial sole implementer estimated20–30min. User expressly
requested adaptive parallel work and fast iteration; this narrow split
overrides skill default of one implementation worker. Exact API/symbol above
bind both sides. Cost if coordination fails: a compile/test repair, no GPU
concurrency or production mutation. Source reviewer checks the combined freeze.

Ruling21:57Z: run the isolated resource/bits/once7 test concurrently with
independent read-only review after root inspected the complete frozen source/
runtime and independently passed28native tests; CUDA build/list/host4 alsoPASS.
Independent review still gates acceptance/public performance claims, not
execution of this guarded test-only probe. A measurement-relevant finding
invalidates the affected run. This avoids idle GPU time without admitting code.

First runtime: resource104regs/local0/static0/dynamic49152/256/occ2 PASS,
target and both positive-reduction tail bit cases passed, then the public
physical qualifier rejected logicalm=0 before timing. Preserve this failedraw.
Ruling22:03Z: repair only K0 test via direct candidate eager2+graph2, finite
fixture precondition and expectedoldC bits with signed zeros normalized to+0,
plus existing input/guard
checks. This is an analytical K0 oracle, not a forced-portable comparison.
No CUDA source/production API/resource gate changes. Cost if wrong: K0 test
could miss a special-value case; this finite-only screen makes no such claim.
Root initially suggested nonzero oldC, then immediately read the actual corpus:
it includes signed zeros and subnormals. Keep that corpus; canonicalize zero in
the independent expected result instead of deleting edge coverage. This
correction was communicated before the repaired GPU run.
