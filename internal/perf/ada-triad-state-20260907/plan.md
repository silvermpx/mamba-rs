# Current Ada Triad state snapshot requested by the user

User addition after approving Triad work: report where the assembled Triad is
ahead of cuBLAS Fast and where it loses, and assess reuse of Fixed improvements.
This is a read-only measurement follow-up after the two current-AUTO profiles,
not a replacement for them and not a new candidate or production qualification.

## Bound acquisition

Reuse the exact CUDA13.2 `gemm_bi_performance_matrix` binary already built for
`ada-triad-two-cell-ncu-20260907`, with its unchanged source/library bindings.
No rebuild, no test/CUDA/dispatcher edits, no new GPU process owner. Keep the
two-cell profile evidence immutable and use a distinct snapshot directory.

Run two existing exact ignored tests, once each with21 windows and order `ab`:

1. `gemm_bi_deterministic_performance_matrix`: select exactly60 public policy
   cells via `GEMM_BI_QUAL_CELL_IDS`, Cartesian product of
   `f32_policy_exact,f32_policy_allow_tf32,bf16_policy_tc,f16_policy_tc`,
   `nn,tn,nt`, and
   `d128_in_proj,d128_out_proj,d768_in_proj,d768_out_proj,prism_in_proj`,
   each ending `/contiguous`. Require120 unique eager/graph records.
2. `gemm_bi_cublas_performance_denominators`: select exactly45 cells via
   `GEMM_BI_CUBLAS_CELL_IDS`, product `cublas/{f32,bf16,f16}/{nn,tn,nt}/`
   with the same five shape names. The unchanged harness emits90 records
   (Fast and Pedantic). Retain both as raw evidence, but use **only Fast**
   for the requested report. Pedantic is not a substitute Fast result.

Use the existing environment constructor, private cache, exact GPU UUID and
quiet PRE/RELEASE records; preserve any distinct transient busy-release/drain
history. Require one executed test, correct expected record counts, positive
finite21-element samples, matching source/binary/library/device/route identity
and exit0. Do not run a duplicate full-matrix smoke before the21-window run.
No repeated valid workload run,101-window expansion, or other CUDA/device lane
is part of this snapshot. The prior two-cell profile wrapper checks are reused.

## Interpretation and report

This existing harness times custom eager/graph separately from vendor eager.
Ratios are **independent quantile ratios**, not paired-window admission.
Do not compute samplewise custom/vendor ratios or call graph/eager a measured
graph/graph win. The main winner/loss table is eager versus eager; show graph
custom timing separately, with no vendor graph claim. Both allocators start
active operands at zero, and the matrix does not restore TN beta1 C before
every operation. This is a synthetic GEMM state snapshot, not end-to-end
training latency, a correctness/bit proof, or release qualification.

For each of60 rows, report exact policy/op/shape, selected physical symbol(s),
custom median/p95, matching Fast median/p95 and their ratios. Aggregate counts
by precision and operation: both ratios<1 is observed faster, both>1 observed
slower, otherwise mixed/equal. Mark results within3% of parity as marginal
diagnostics, not robust promotion evidence. Rank the worst ratios and the
largest absolute median gaps separately; avoid unweighted averages being
presented as a real workload speedup. ExactF32 and TF32 share the explicit
FAST_TF32 denominator but preserve their different numerical contracts.

Root independently replays records, joins and quantiles, and the reviewer checks
the result interpretation. Commit raw evidence and a concise state report.
The snapshot informs the next bounded candidate; unchanged full baselines
must not run again after every prototype.
