# Ada half S3 — Task5B

Both BF16 and F16 retain a confirmed CUDA13.2 exact-B0 production improvement after fresh final paired21 and paired101. Native-half cuBLAS remains faster. No production integration occurred.

Authoritative sources: `candidate.cu`, `benchmark-final.cu`, `run-final-cell.sh`.
Authoritative generated evidence is under `final/`; see `report.md` for full results, gate details, source hashes and exclusions. Initial top-level timing attempts are excluded because graph poisoning was incomplete. All artifacts are retained and covered by rooted `SHA256SUMS`.

Final paired101 candidate/production p50/p95:
- BF16:0.964968888968 /0.996412833347.
- F16:0.964326220305 /0.981133357944.

Final Ada release:2026-09-07T01:40:54Z, exact UUID idle with no compute apps.
