# Ada Triad: reuse winners, then qualify one assembled batch

Continue branch `codex/gemm-bi-triad-sm80`. User authorizes autonomous kernel
work, adaptive agents, all installed CUDA toolkits and human-only commits.
Discovery checkpoint `8931e8a1` freezes the raw results and rejected variants.
No new branch, no deletions, no Fixed or RTX5090 retuning.

## Research-informed choices

NVIDIA CUTLASS overlaps global/shared loads with computation, including
register fragments, and moves address setup outside the mainloop. Its
multistage implementation distributes asynchronous copies across warp-MMA
iterations. This is the source-level recipe, not proof of a speedup here:
[efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[actual multistage implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_multistage.h).

Ada has 100 KiB shared memory/SM, 99 KiB maximum per block, and 64K 32-bit
registers/SM. A deeper pipeline trades residency for overlap; occupancy alone
is not a performance verdict. Compile explicitly for SM89 for its FP32
throughput. [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

Our evidence: TC64 improves residency but loses 45–50%; TC128 SASS already
interleaves some LDSM/HMMA, with no excessive shared wavefronts. Therefore
do not repeat a source-only "missing ping-pong" claim or another tile-only
screen. Test the complete existing Fixed S3 package (copy plan, compact
layout, distributed copies, fragment schedule, vector epilogue) before
writing another half kernel. This choice is an inference from sources/profile.

For TF32 NN d768-out, the current wide M128N128 grid has only96 CTAs for142
SMs. N96 increases it to128 CTAs with unchanged ascending K8 accumulation.
This is a concrete underfilled-wave hypothesis, not an assumed win. NVIDIA
describes the tradeoff between larger-tile reuse and fewer parallel blocks:
[matrix-multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
Use the existing Fixed N96 geometry but preserve current Triad AddHalfUlp
conversion by an exact count-checked source adapter; explicit RNA differs for
some exceptional payloads and cannot silently replace the old numeric route.

## Bounded concurrent work

1. Root: actual loaded Fixed CopyPlan NN, three shapes, versus independently
   qualified public AUTO and explicit cuBLAS `32F_FAST_TF32`, same full-mantissa
   input words, guarded output/input reset before each one-GEMM observation.
   Keep tail/K0/null and eager/graph repeat bit checks before target timing.
   Run first13.2, then each retained candidate on12.8/13.0; no Pedantic swap.
2. Carver: existing transpose16 + actual Fixed CopyPlan inner GEMM, NT d768-out,
   versus public AUTO, whole two-node pipeline, once7. Keep the measured M64
   pipeline unchanged as an available fallback; no inner-only timing claim.
3. Beauvoir: actual Fixed half S3 holder versus existing TC128, F16/BF16 NN
   d768-in, exact-bit screen then once7. No new CUDA source or arithmetic.
4. Avicenna alone owns GPU execution; all builds/list/raw/source freezes and
   failures are preserved. Source reviews and analysis can run concurrently.

## Integration once retained live artifacts are measured

Prefer reusing the already loaded optional Fixed holders. Do not duplicate
their CUDA or change the Fixed/portable module bytes. F32 NN CopyPlan needs
a distinct scalar plan tag, a bounded three-shape/overwrite/no-bias selector,
holder-present and real Fixed compiler/artifact facts, and a per-node module
identity in physical/prepared/eager routes. Never label a Fixed function as
TriadScalar. Unknown artifact/toolkit, absent holder, unsupported epilogue,
misaligned pointer or unqualified shape retains its prior route.

For NT CopyPlan, node0 is TriadScalar transpose and node1 is Fixed GEMM. Keep
the existing scratch allocation/guard/graph lifetime contract. Its speed and
bit result are not implied by the NN win; screen the whole pipeline first.

Global tuning45, numeric ABI and schedule remain unchanged. New route tags
and scoped identities distinguish new plans without invalidating unrelated
portable, Fixed and SM120 evidence. Add focused negative selector, identity,
missing-holder and graph tests before implementation; one final all-toolkit
assembled AUTO/Fast batch follows, not a full matrix per small patch.

TF32 TN dense d768-in is another retained candidate. Adding it to the existing
fourth Ada finalist artifact changes that artifact identity, including NT.
Use a two-symbol finalist module and private revision2; requalify NT3 + TN1
together on all3 toolkits. Keep NT resource cap/occupancy separate from TN,
and never carry old NT admission literals onto the new artifact or add a
fifth artifact to sidestep identity validation.

After assembly, report exact remaining Fast gaps. Only then remove clearly
superseded experimental variants, with their results preserved in evidence.
This plan does not claim every workload can beat Fast under the stricter
exact-F32 contract; no unmeasured loss is declared a hardware limit.
