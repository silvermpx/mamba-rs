# Ada exact-F32 NT Fixed CopyPlan siblings — Batch A admission

Date: 2026-09-09. The retained exact-F32 NT d768-in
`(2048,768,3072)` and canonical Prism `(4621,384,1928)` routes are admitted
only for the three live-qualified CUDA12.8/13.0/13.2 composed identities. No
new CUDA kernel or TN export is part of this batch.

## Production routes

| Cell | tag | physical route | transpose scratch |
| --- | ---: | --- | ---: |
| d768-in | 39 | `gemm_bi_transpose_f32_32x16_d768_v1` (`TriadScalar`) → `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` (`Fixed`) | 2,359,296 f32 |
| Prism | 40 | same ordered symbols/modules | 740,352 f32 |

The selector remains exact-shape and exact-stride only. It requires CC8.9/142
SM, no bias, `alpha=+1` and `beta=+0` by bits, non-null 16-byte-aligned
output/A/B pointers, a loaded Fixed function, and a same-domain composed
identity. The admitted evidence cohort is an alias of exactly the three frozen
qualification candidates; identities cannot be added to admission without
first extending the sealed candidate table.

Every shape, stride, operand, device, compiler, module, artifact, digest,
target, toolkit-library-domain, or availability mutation falls back to the
exact prior AUTO selector. Existing NN d768-in and NT d768-out Fixed routes are
tested unchanged.

Eager execution and prepared graph construction consume the same tags 39/40
and the same physical plan. The required graph is exactly two nodes and one
default dependency: transpose then Fixed. The operation-level gate checks both
Driver ABIs, both launch configurations, and that transpose output and Fixed B
are the same non-null scratch pointer. Physical qualification independently
checks ordered module/symbol/config/numeric identity, allocation-bound argument
digests, route digest, eager/prepared equality, exact output, A/B immutability,
and red zones.

## Frozen pre-admission evidence

All three runs used the exact fail-closed source snapshot `f04584c8`. Live
correctness, Driver ABI, resource, guard, actual-prior-AUTO identity, once3 and
once7 gates passed on each toolkit. Across the four eager/graph ABBA/BAAB
once7 strata per cell, all recorded p50 and p95 ratios were:

| toolkit | d768-in candidate/prior AUTO | Prism candidate/prior AUTO | log |
| --- | ---: | ---: | --- |
| CUDA12.8 | `.33633–.33777` | `.79681–.79937` | `/root/logs/exact-nt-prequal-f04584c8-cuda128.log` |
| CUDA13.0 | `.33651–.33786` | `.79631–.79859` | `/root/logs/exact-nt-prequal-f04584c8-cuda130.log` |
| CUDA13.2 | `.43148–.43328` | `.86942–.87176` | `/root/logs/exact-nt-prequal-f04584c8-cuda132.log` |

Each bound is below the strict `.99` admission threshold. These are whole
pipeline retained-best improvements over the prior actual AUTO, not cuBLAS
Fast wins. Fixed and `TriadScalar` compiler/artifact identities are the exact
same-domain rows frozen in `dispatch.rs`; no identity was synthesized from a
different target.

Raw receipts are also preserved locally under `evidence/`. Independent replay
recomputed all 48 strata, 240 paired brackets and 960 observations across the
three logs, reproduced every p50/p95 to within `1e-12`, and confirmed every
admission ratio below `.99`. The CUDA12.8 table minimum above is corrected to
the once7-only minimum; the earlier minimum included once3.

- [CUDA12.8](evidence/exact-nt-prequal-f04584c8-cuda128.log), SHA-256
  `222295856a719a1d6958d8c0ec0222ce03bcb35f6309f985a21047cc27b22aaa`
- [CUDA13.0](evidence/exact-nt-prequal-f04584c8-cuda130.log), SHA-256
  `b6de82ffedbd4838398cfac0d3257f59181d44af6189498d164713b4a969a06f`
- [CUDA13.2](evidence/exact-nt-prequal-f04584c8-cuda132.log), SHA-256
  `0ee5b5dd3014809d1bdbb384214263b95f523f6196451fe3f837b7ba30541e62`

The ignored historical
`ada_f32_nt_copyplan_siblings_pre_admission_qualification` remains as the
reproducible evidence entry, but now fails immediately with a clear instruction
when it observes that AUTO is already admitted. It must only be run from the
fail-closed `f04584c8` snapshot.

## Post-admission gate

The ignored
`ada_f32_nt_copyplan_siblings_post_admission_qualification` is the release
follow-up for each frozen toolkit. It requires:

- exact live Driver ABI and resource caps for both physical functions;
- tail, exceptional-payload, non-unit-alpha and K0 exactness;
- target actual AUTO eager/prepared physical identity plus an exact operation
  graph with transpose → Fixed ordering/configuration/shared scratch;
- exact bits, A/B immutability and output/input/scratch red zones;
- once21 actual AUTO versus forced generic exact F32 in eager/graph ABBA and
  BAAB, with every p50 and p95 strictly below `.99`;
- a separately labeled once21 actual AUTO/cuBLAS Fast comparison with no false
  exactness or promotion claim.

No GPU or post-admission timing was run while preparing this admission commit.

### CUDA13.2 actual AUTO result

The exact `fadf5250` snapshot subsequently passed the post-admission test on
Ada/CUDA13.2: 1 passed, 0 failed. Both cells selected the admitted two-node
route, passed actual eager/prepared identity and exactness, edge cases, A/B
immutability and red zones. All 16 once21 strata were independently replayed
from 336 brackets / 1,344 observations.

| Cell | AUTO / forced generic exact p50 | Worst p95 | AUTO / Fast p50 |
| --- | --- | --- | --- |
| d768-in | `.43204–.43318` | `.43350` | `2.36147–2.36382` |
| Prism | `.73645–.73814` | `.73861` | `3.91622–3.92822` |

The forced generic comparator is `gemm_bi_nt` for both cells. Prism's previous
actual AUTO was `gemm_bi_nt_slim`, so this table is a different denominator
from the pre-admission table and must not be described as another improvement
over the previous AUTO. Fast has a distinct numerical contract.

[Raw CUDA13.2 receipt](evidence/exact-nt-admitted-fadf5250-cuda132.log).
CUDA12.8/13.0 actual-AUTO follow-ups remain pending at this checkpoint.

## TDD and host verification

The selector test first failed with an empty cohort while expecting all three
qualified rows. After admission, the scoped checks were:

```text
cargo test --no-default-features --features 'cuda,cudarc/cuda-13000' \
  --lib nt_fixed_copyplan_sibling -- --nocapture
5 passed

cargo test --no-default-features \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery -- --nocapture
4 passed

cargo test --no-default-features --features 'cuda,cudarc/cuda-13000' \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery --no-run
compiled successfully; ignored CUDA tests were not executed
```
