# Current-body RTX5090 TF32 retained-route qualification

This directory contains pre-admission candidate measurements, not a claim
that every route is already selected by production AUTO. The comparator is
the toolkit's current exact-F32 AUTO floor, not cuBLAS Fast.

## CUDA13.0 existing23-key packet

All23 existing shape keys completed their numerical, repeat/eager-graph,
guard and paired-timing checks. Exactly21 candidates passed the complete
21/101-window AB/BA performance admission rule. Two candidates remain
unadmitted in this original V1 packet:

| Case | Final101 eager speedup / p05 | Final101 graph speedup / p05 | Failed check |
|---|---|---|---|
| NN d128-in | 1.138 / 1.125 | 1.666 / 1.665 | Discovery21 eager p05, both orders |
| TN d128-out | 1.609 / 1.607 | 1.599 / 1.598 | Discovery21 eager p05, both orders |

Every sub-unit ratio in those cases is at the start of discovery: first3
samples per order for NN, first2 for TN. No recorded sample was dropped and
neither failed candidate is admitted on the strength of its final median.
The source warms/calibrates before a subsequent quiet-GPU wait. The audit
confirmed this ordering defect; prospective V2 results are recorded below.

The other21 cases are valid retained winners under this packet's full rule.
Their raw records preserve all observations, exact chosen symbols, numeric
families and specialized/portable compiler/artifact/device identities.
`verify.jq` independently replays nearest-rank statistics, sample ratios,
winner selection, identity consistency and completion counts for all23.

`cuda130/G01/` contains the first two cases. Its GPU test passed in81.51s;
the outer wrapper then stopped because it counted the JSONL completion row
as a cell. `cuda130/rest/` resumes at G02 without rerunning those cases and
contains the remaining21. Both packets use the same frozen production source
and private toolkit-specific caches. The wrapper correction changes no test
or admission threshold.

## Prospective CUDA13.0 V2 confirmation

The V2 harness runs exactly four calibrated paired warmup cycles after the
final quiet gate and before recorded discovery. Each cycle is candidate,
scalar, scalar, candidate. Both paths use the same fixed procedure; recorded
21/101 windows, numerical checks and admission thresholds are unchanged.
Completion metadata names the V2 schema, count and order.

The predeclared one-shot sequence G06 then G09 completed in38.88s and38.40s.
Both candidates now pass every existing admission gate:

| Case | Final101 eager speedup / p05 | Final101 graph speedup / p05 |
|---|---|---|
| NN d128-in | 1.137 / 1.123 | 1.666 / 1.665 |
| TN d128-out | 1.609 / 1.607 | 1.599 / 1.598 |

Discovery eager p05 is above1 in both orders for both cases. The original
V1 failures remain unchanged. V2 raw receipts, all four focused host-test
passes and the frozen source manifest are in `cuda130/warmup-v2/`.
Completion hashes and all sample statistics independently replay. Combining
the21 V1 winners with these two V2 winners yields23 unique accepted keys,
with one identical specialized/portable identity pair across all23.
The corrected fixture and warmup patch received an independent static
specification/code-quality review with no findings. Runtime evidence proves
the new protocol's result; no retrospective TDD RED is claimed for this patch.

## G10 actual-shape result

The corrected fixture uses public TN dimensions `(8192,128,128)`, yielding
output shape `(128,128)` with reduction length8192. The mapping test passed
on both toolkits. The retained `m64n128_bk32_s4_pair` candidate passed its
numerical checks but lost decisively against the exact-F32 comparator:

| Toolkit | Eager speedup | Graph speedup | Admission |
|---|---:|---:|---|
| CUDA13.2 | 0.20328 | 0.20776 | Rejected |
| CUDA13.0 | 0.18564 | 0.19019 | Rejected |

These are conservative AB/BA final101 medians; discovery and final p05
ratios agree with the loss. Neither result is explained by the front-only
transient above. Raw V1 receipts are in `cuda132/g10/` and `cuda130/g10/`;
their completion SHA-256 values and all recorded statistics replay exactly.
The existing CUDA13.2 G10 AUTO admission must be removed during assembly.
The kernel implementation and its other keys remain available.

## Still required

- The current-body CUDA12.8 retained-route packet, including G10.
- Resolve or explicitly omit each toolkit's performance failures.
- Add fresh live cohorts only from accepted receipts, then prove exact
  selected AUTO symbols and repeated output bits on all24 candidate keys.

No old CUDA13.2 compiler identity is copied to a lower-toolkit cohort.
Matrix evidence's `eager_graph_equal` flag denotes ordered physical-launch
equality; the selector's explicit output comparisons supply the bit checks.
