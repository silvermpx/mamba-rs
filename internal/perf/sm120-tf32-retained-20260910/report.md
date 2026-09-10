# Current-body RTX5090 TF32 retained-route qualification

This directory contains pre-admission candidate measurements, not a claim
that every route is already selected by production AUTO. The comparator is
the toolkit's current exact-F32 AUTO floor, not cuBLAS Fast.

## CUDA13.0 existing23-key packet

All23 existing shape keys completed their numerical, repeat/eager-graph,
guard and paired-timing checks. Exactly21 candidates passed the complete
21/101-window AB/BA performance admission rule. Two candidates remain
unadmitted pending a bounded timing-start investigation:

| Case | Final101 eager speedup / p05 | Final101 graph speedup / p05 | Failed check |
|---|---|---|---|
| NN d128-in | 1.138 / 1.125 | 1.666 / 1.665 | Discovery21 eager p05, both orders |
| TN d128-out | 1.609 / 1.607 | 1.599 / 1.598 | Discovery21 eager p05, both orders |

Every sub-unit ratio in those cases is at the start of discovery: first3
samples per order for NN, first2 for TN. No recorded sample was dropped and
neither failed candidate is admitted on the strength of its final median.
The source warms/calibrates before a subsequent quiet-GPU wait; whether
that cold restart accounts for the discrepancy is being checked separately.

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

## Still required

- True G10 public dimensions(8192,128,128), whose former fixture was
  transposed, on CUDA13.2 and13.0.
- The current-body CUDA12.8 retained-route packet, including G10.
- Resolve or explicitly omit each toolkit's performance failures.
- Add fresh live cohorts only from accepted receipts, then prove exact
  selected AUTO symbols and repeated output bits on all24 candidate keys.

No old CUDA13.2 compiler identity is copied to a lower-toolkit cohort.
Matrix evidence's `eager_graph_equal` flag denotes ordered physical-launch
equality; the selector's explicit output comparisons supply the bit checks.
