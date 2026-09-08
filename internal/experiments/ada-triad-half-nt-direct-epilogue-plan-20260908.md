# Half NT direct epilogue discovery

Measured NT Fixed-S3 B-XOR d768-out is26% below TC64 but2–6% above Fast
in graph. Hypothesis: its shared-memory output exchange and final barrier
cost more than the existing direct32-bit pair stores on this target.

Use the already implemented `scalar_epilogue` (which uses packed pair RNE
stores on aligned full pairs), not a new conversion or arithmetic path.
Remove only the vector-epilogue dispatch from the test-only NT adapter;
copy plan, K16 chain, shared staging and alpha*acc arithmetic stay identical.

- [x] Native RED/GREEN source restoration tests; bounded independent review.
- [x] After NT four-cell coverage freeze, add one candidate on d768-out half
  types; compare paired to retained S3 and explicit Fast, TC64 bit reference.
  Once7/20ops,256B, target bits+guards/resources only. No unchanged out rerun.
- [x] Sole GPU executor; retain a measured improvement or record a valid loss.
  Do not admit it or launch full gates before whole-Triad shortlist.

Result: valid loss to retained S3 and Fast, bits/resources PASS. Keep the
vector epilogue; no unchanged retry. See
`../perf/ada-triad-half-nt-direct-epilogue-20260908/report.md`.
