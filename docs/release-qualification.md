# Release qualification

CI compiles and lints everything, but it owns no GPU: a fully green
pipeline proves nothing about kernel behavior. A release is therefore
qualified on a GPU box against the lane manifest, and the evidence it
produces is data, not scrollback.

## Lanes

`qual/lanes.toml` assigns every test suite to a lane; the census suite
(`qual_lane_census`, host-side, runs in CI) keeps the manifest and the
tree in lockstep.

| lane | when it runs | how |
|---|---|---|
| gate | every GPU-box run | `qual/run.sh gate` (the plain battery) |
| contract | before a release, after a kernel edit, after a merge | `qual/run.sh contract` (the ignored bit gates, first red stops) |
| record | manually, one instrument at a time | `qual/run.sh record` lists the commands |
| host | every CI push | part of the featureless `cargo test` |

## Evidence

Bit-gate printers record their cells when `MAMBA_RS_ACCEPTANCE_TSV`
names a file: boundary sets of the invariance matrix, the tensor-core
goldens, training run digests, the decode chain, the serve hash table,
the scan ledger. Two captures compare with:

```
cargo run --release --example acceptance_diff -- base.tsv candidate.tsv
```

which prints every moved or one-sided cell and exits non-zero on any
difference. A release candidate's capture must be byte-identical to the
last accepted one for every cell whose kernels the release does not
intentionally change; an intentional change re-records its cells in the
same commit that makes it.

## Order

1. `qual/run.sh gate` green.
2. `qual/run.sh contract` green, with the evidence capture on.
3. `acceptance_diff` against the previous release's capture: clean, or
   every difference accounted for by an intentional, documented change.
4. Performance guard: the serve page re-measured A/B against the
   previous release under identical ambient conditions (alternating
   runs, minima per arm); regressions are defects.
5. `cargo publish --locked --dry-run` - the dependency resolution CI
   tested is the one that ships.
6. Tag only after the release commit's CI is green.
