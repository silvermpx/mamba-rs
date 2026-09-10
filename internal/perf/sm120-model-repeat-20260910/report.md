# RTX 5090 model graph replay checks — 2026-09-10

CUDA13.2, RTX5090 CC12.0/170, driver595.84. Frozen source:
`/root/mamba-release-sm120.J6z6IG`; see `assembly-source.sha256`.
These runs precede the SM120 per-symbol reachability repair.

All five focused tests passed:

| Test | Result | Runtime |
| --- | --- | ---: |
| Mamba BF16, five-step eager versus graph | No bit-diverged master weights | 8.70s |
| Mamba-3 BF16, five-step eager versus graph | Maximum weight difference0 | 8.86s |
| Mamba-3 BF16 repeated graph training | Output/gradient/final-weight digests agree | 37.18s |
| Mamba-3 F16 repeated graph training | Output/gradient/final-weight digests agree | 16.73s |
| Mamba-3 F32 repeated graph training | Output/gradient/final-weight digests agree | 16.83s |

Each raw `.log` records the exact binary and test name. Adjacent build, test
inventory and five-sample idle/free-memory preflight logs identify each run.
The first two targets are `training_graph_parity` and
`m3_training_graph_parity`; the three digest checks are individually selected
ignored tests in `m3_training_graph_safety`. Each invocation used
`cargo test --locked --release --features cuda --test <target> <exact-name>
-- --exact --nocapture --test-threads=1`, adding `--ignored` for digest checks.

All tests explicitly enable batch-invariant Triad and disable tensor cores.
Thus mixed-precision cases exercise F32/upcast routing, not the newly assembled
native-half tensor-core winners. The Mamba-3 five-step assertion permits5e-5
weight error despite observing0 in this run. Digest tests reconstruct the same
tiny model twice and compare two captured replays from each reconstruction;
the F16 case also requires that the optimizer step was not overflow-skipped.

These results establish same-process model integration/repeatability in those
scopes. They are not cross-driver/cross-GPU reproducibility, positive inference
graph replay, or performance measurements.
