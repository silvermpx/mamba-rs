# F16 NT d768-out M64N128/S3 — 2026-09-08

New retained-family improvement on RTX6000Ada/CUDA13.2: about4.5% less time
than the previous best Fixed-S3 B-XOR candidate. Strict cuBLAS Fast screen
still fails graph p95. Keep this F16 finalist; do not call it a Fast champion
or repeat its unchanged F16 screen. BF16 has not been measured yet.

| Comparator | Eager candidate/comparator p50 | Graph p50 | Worst graph p95 |
| --- | ---: | ---: | ---: |
| Retained M128N128 S3 | .9542–.9545 | .9550–.9551 | .9580 |
| Native-half cuBLAS Fast | .9468–.9542 | .9812–.9898 | 1.0105 |

Candidate39.6–40.2us versus retained41.5–42.1us. The strict paired Fast gate
requires every eager/graph × ABBA/BAAB p50 and p95 below .99; lower medians
alone are insufficient. No public AUTO admission or full-toolkit claim.

The candidate changes M128→M64 around the same BK64/S3 K16 MMA sequence.
Grid192→384; accumulators64→32 floats/thread; compiled registers167→119.
Actual candidate resources: block256,73,728 dynamic shared,zero static/local,
occupancy1. Improved wave filling motivated the test; it is not asserted as
the sole measured cause of the gain, and B tile-request traffic increases.

## Evidence

- Target `(2048,1536,768)` equals forcedCurrentTC64 and retainedS3 bitwise:
  eager2+graph2,12 bit records. Negative-alpha tail `(67,131,69), alpha=-.75`
  also equals both references in eager/graph, including input/output guards.
- 256B logical alignment,20 GEMMs/observation, whole captured graphs. Root
  replayed all56 brackets/224 observations and nearest-rank p50/p95.
- Native source/ownership/ring-normalization tests9/9 pass. GPU exact1 PASS;
  artifact cache unchanged, pre/drain quiet. Release had no compute apps but
  11% residual utilization and is not labelled quiet.
- Exact source is the commit containing this report. Measured typed harness:
  `5abc2d62c8340beb72005613e705030532bd6aa54b572403b521e02720bf2b0b`.
  Helper `triad_half_nt_m64n128_s3_source.rs`:
  `a97c836320d90cf85c633ceaf633f77b1bdf8cd5fb5aca7ce129a609029f6fe0`.
- Composed CUDA: `e80ca6e724a1126e1ebe5fa1fc7af913a37ca909b265952442c7fb1f5c170210`.
  Binary: `ff7e273bc7ec8d84f8e97f8aa7955a3cff64ac45116f4d7fcda146b1b00470a5`.
- [Raw log](evidence/cuda132/run1/test.log):
  `319fd7a8f3fa9ad3af5e9b20041a14755f23d1e26ab7bcc481ac63165e11b87f`.
  Exact bare test: `ada_half_nt_fixed_s3_m64n128_f16_d768_out_bare_discovery_once7`.

Production, Fixed inference, SM120 and other half shapes are unchanged. The
same arithmetic body may be screened on BF16 separately; no cross-dtype win
is inferred from this F16 result.
