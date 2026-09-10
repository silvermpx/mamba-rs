# Ada final AUTO reachability review

Date: 2026-09-10

## Verdict

**PASS — no mapping exceptions and no unexpected fallthrough.** The completed
Ada packet's 66 actual production AUTO cells select the currently intended
Ada routes. Every integrated retained winner represented by this inventory is
reachable, and every cell without an accepted specialized Ada decision remains
on its intended portable or scalar-family route.

This is a read-only route-reconciliation check, not new performance discovery.
The comparison sources were the current release decisions in
`internal/release-0.7.0-checklist.md`, the assembled-route reports
`internal/perf/ada-final-assembly-20260910/report.md` and
`internal/perf/ada-d128-assembly-20260910/report.md`, and the frozen decision
index `internal/perf/ada-triad-discovery-shortlist-20260908.md` together with
the final TF32 candidate report. No benchmark was rerun and no source was
changed.

## Reconciliation

Input:
`internal/perf/final-auto-benchmarks-20260910/triad-ada-full/triad.jsonl`,
production Git SHA `732c11462dd9a47376e6d15b36040ca69bde3e34`.
It contains 324 pair records and one completion record declaring 66 cells, 81
views, two paths, two orders, and 21 windows/order. Each cell has one stable
physical launch digest and symbol sequence across all of its comparator,
eager/graph, and AB/BA records; all 66 report `eager_graph_equal=true`.

| AUTO family | Cells | Observed route mapping | Decision match |
|---|---:|---|---|
| Exact F32 | 15 | Four d128 NN/NT cells retain the scalar family; both d128 TN cells select their admitted one-node `triad_sm89_exact_f32_d128` symbols; the three large NN cells select the Fixed N64 CopyPlan; all three large NT cells select transpose plus that CopyPlan; d768-in TN selects transpose plus dual-chunk fused finalize, while d768-out and Prism TN select their admitted direct raw plus fixed reducer pipelines. | Exact match |
| Allow deterministic TF32 | 21 | Six accepted joint cells select the intended joint routes: NN d768-out/Prism, TN d768-in/out/Prism, and NT d768-in. The two later promotions are present specifically as NT A-ldmatrix M128N96/S3 and TN Prism pre-RNA M64N96/S2. NT d768-out, `large_deep`, and Prism retain the accepted SM89 finalist compact8 route. The other 12 cells retain their intended portable SM80 routes. | Exact match |
| BF16/F16 TC | 30 | All 18 assembled Ada half cells select `triad_sm89_half`: six large NN S3 cells; six large NT cells with B-XOR for d768-in/Prism and M96N128/S3 for d768-out; and six large TN cells with vec2 for both d768-in plus BF16 d768-out, compact for F16 d768-out and both Prism cells. The 12 d128 half cells remain on portable SM80 TC64. | Exact match |

The module-owner census is consistent with that mapping: 18
`triad_sm89_half`, 6 `triad_sm89_tf32_joint`, 3 `triad_sm89_finalist`, 2
`triad_sm89_exact_f32_d128`, 3 single-node Fixed, 4 scalar-family, 24 portable
SM80, and 6 mixed exact-F32 pipelines whose nodes explicitly identify their
transpose/raw/CopyPlan/reducer owners.

There are therefore **zero missing accepted winners**, **zero foreign/new
symbols**, and **zero unexplained scalar or portable fallthroughs** in the 66
cells. A canonical diagnostic projection of
`cell_id + physical_launch_digest + ordered physical symbols`, sorted uniquely,
has 66 rows and SHA-256
`fc1050c305ad1ae43fab883077eb942d60f5ad801fcab0d9c0ddec777fa79cfe`.
The exact serialization recipe is:

```sh
jq -r 'select(.schema=="MambaBiFinalProductionAutoPairV1") | [.cell_id,.physical_launch_digest,([.physical_nodes[].symbol]|join("+"))]|@tsv' \
  internal/perf/final-auto-benchmarks-20260910/triad-ada-full/triad.jsonl \
  | LC_ALL=C sort -u > ada-final-auto-route-map.tsv
shasum -a 256 ada-final-auto-route-map.tsv
```

The serialized rows are UTF-8 text with three tab-separated fields, physical
symbols joined by literal `+` with no spaces, bytewise C-locale `sort -u`
ordering, and the ordinary trailing newline written by the pipeline.
Changing the symbol separator (for example to a comma) intentionally changes
the digest.

## SM120 pending-edit boundary

The pending SM120 lower-toolkit cohort work is board/cohort admission only. It
must not change the Ada numerical route, symbol sequence, launch digest, or
eager/graph equality for any of these 66 keys. The canonical projection above
is the compact Ada preservation baseline for a post-edit reachability check;
any delta requires an explicit Ada decision and cannot be explained by an
SM120-only cohort addition.
