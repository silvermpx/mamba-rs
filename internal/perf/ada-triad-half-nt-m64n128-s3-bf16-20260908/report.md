# BF16 NT d768-out M64N128/S3: new retained-best improvement

Ada/CUDA13.2, logical(2048,1536,768), alpha1/beta0/no bias. SAME CUDA adapter
as the retained [F16 winner](../ada-triad-half-nt-m64n128-s3-20260908/report.md),
screened only for BF16 here; F16 was not rerun. This is not a robust Fast win.

| Candidate / comparator | Eager p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| Retained Fixed S3 B-XOR | .95396–.95408 | .95465–.95522 | .958546 |
| Native-half cuBLAS Fast | .98840–.98845 | .99730–1.00714 | 1.021738 |

Candidate39.6–40.2us versus retained41.5–42.1us: **4.5–4.6% less time**.
Keep M64N128/S3 for BF16 alongside F16. Fast eager medians are better, but
graph p95 fails; do not count this as a Fast-winning replacement or rerun
unchanged to chase noise. The raw Fast stop decision does not erase the paired
retained-family win.

Resources119regs, local/static0, dynamic73,728B,256 threads, grid384, occupancy1.
Target currentTC64/retained/candidate exact eager/graph checks pass12 bit rows;
the BF16 tail(67,131,69), alpha−.75, also passes strict three-arm equality.
Timed buffers256B aligned,20 complete GEMMs/observation, once7 × eager/graph ×
ABBA/BAAB. Root replayed all8screens/56brackets/224observations and quantiles.

Exact test `ada_half_nt_fixed_s3_m64n128_bf16_d768_out_bare_discovery_once7`.
Measured main SHA256 `027facf06d437a49f846628d78ea4dbc72506ecaa826afc8c2f2638650954322`;
unchanged helper `a97c836320d90cf85c633ceaf633f77b1bdf8cd5fb5aca7ce129a609029f6fe0`;
binary `57374f3195499d137a2a6321226210f204763df30c2fb641a777a3b662875b2e`.
[Raw log](evidence/cuda132/run1/test.log) SHA256
`5663cdffd5ec9743424452f246b9c02edec2d4b8d3f6deb14b28e780ba6126df`.
Exact1 PASS; four private cache artifacts unchanged. PRE13:36:18 and
DRAIN13:36:43UTC quiet; RELEASE13:36:38 has no compute app. No full gates,
production selector change, Fixed inference change or fresh SM120 claim.
