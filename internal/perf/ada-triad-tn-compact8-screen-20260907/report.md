# Ada TN Prism compact-eight-warp S2 screen

Outcome: **STOP, no retry or promotion.** The test-only candidate passed the
resource and focused numerical gates, but was slower than the current actual
AUTO route in all four paired once7 strata.

| Path | Order | candidate/AUTO p50 | candidate/AUTO p95 |
|---|---:|---:|---:|
| eager | ABBA | 1.045341191 | 1.052183557 |
| eager | BAAB | 1.048065537 | 1.052489965 |
| graph | ABBA | 1.060965417 | 1.065088773 |
| graph | BAAB | 1.063667295 | 1.066790457 |

The candidate used the distinct test-only symbol
`gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2`
at logical `(4621,384,1928)`, grid 93, 256 threads, 49,152 dynamic shared
bytes. CUDA 13.2 reported 104 registers, zero local/static shared bytes and
two resident CTAs/SM. The current actual AUTO denominator was the production
M128N64/BK32/S3 TN route with the same grid and 79,872 dynamic shared bytes.

Focused correctness passed for the full-mantissa target, tail `(129,65,36)`
with alpha 1 and -0.75, eager and graph repeats, immutable inputs and red
zones. TN zero reduction `(0,65,36)` passed eager2/graph2 against the finite
analytical `fma(1,+0,oldC)` oracle; signed zero is canonicalized to positive
zero. The public qualification facade deliberately rejects logical M=0, so
this K0 check is not described as a forced-portable launch comparison.

The first GPU attempt is retained as a harness failure: resources and the
positive-reduction bit cases passed, then the public qualifier rejected the
K0 request before timing. The narrow repaired attempt executed exactly one
test successfully. It used strict quiet PRE, an immediate busy RELEASE, a
separate quiet DRAIN five seconds later, and an unchanged private four-artifact
production cache. No candidate PTX was persisted.

Bindings for the accepted attempt:

- test source: `38b99e571ced8aae8d267eccb1fa052e6a1b160075c1e76132458e8e7923d673`
- composed CUDA source: `14bf43fb3303ee23fd5fb14ffc6f662db890f9dbf5ca97edb014cd46465d3b28`
- test binary: `9c416a412ac9d1a3a1fa2853a166d12e8ff1dab036b330a306bab755d37be6ef`
- source manifest: `449cc63da3d7853d53407fa4ccdd766d7674a33e91a5d79be290ba08c9625efc`
- raw test log: `22d4537074aa953cf704391079ec26e876f07c563f723a966b71cb0edb5956c6`

Raw logs and receipts are under `attempts/attempt0-k0-api-reject/` and
`attempts/final2/`. This was a bounded discovery screen, not full
qualification and not a Fast comparison. Production code and AUTO selection
were unchanged.

Root independently replayed all28brackets/112one-GEMM timing observations,
arm means, ratios, four p50/p95 results and STOP decision. All379 source rows
are accounted for:378 current files plus the committed unrelated SM120-test
baseline. Root independently recomposed the full CUDA source and reproduced
14bf43fb...; native harness29/29 passed after the K0 repair. Two native-only
dead-code warnings and cosmetic inherited NT error strings are retained.
The new K0 oracle initially failed to compile while undefined; this is not
claimed as a behavioral TDD RED. The earlier real GPU API rejection is retained.

Classification: rejected for this Ada/CUDA13.2 Prism TN screen; test-only,
not an AUTO route and not a global rejection for other GPUs. Do not delete
RTX5090 winners or unrelated experiment files based on this result.
Next distinct ablation: keep compact S2 staging but restore the original
four-compute-warp ownership. This separates the layout hypothesis from the
eight-warp expansion; no new speedup is established by proposing it.
