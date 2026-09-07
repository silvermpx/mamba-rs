# Ada Triad NT sibling-shape discovery

This CUDA 13.2 test-only batch compares the retained padded-dense and
compact32 eight-warp S2 mechanisms with the actual public AUTO route on two
sibling NT shapes. It does not compare against cuBLAS Fast. No full
qualification or production promotion ran.

The frozen 374-source manifest is
`b4a7d40dde86a31fc82cc808aa955b11dc8dea8d437b3e3805b89fb90fd37910`;
the binary is
`a9e7b2c3d39e42cdf254f2c013e9e39ce9930a54df58ca715ae40b8ebdbab0ab`.
Every arm passed finite full-mantissa, eager/graph repeat and forced-tail
checks before timing.

| Mechanism | Shape `(M,K,N)` | Path/order | AUTO median (us) | Candidate median (us) | candidate/AUTO p50 | p95 | Decision |
|---|---|---|---:|---:|---:|---:|---|
| dense | `(2048,1536,768)` | eager/ABBA | 121.149153180 | 110.489601559 | 0.912102593 | 0.912773751 | advance |
| dense | `(2048,1536,768)` | eager/BAAB | 120.911645889 | 110.283380085 | 0.911986121 | 0.912442410 | advance |
| dense | `(2048,1536,768)` | graph/ABBA | 119.536357456 | 109.872712029 | 0.919154433 | 0.919447184 | advance |
| dense | `(2048,1536,768)` | graph/BAAB | 119.534222285 | 109.830045700 | 0.918864035 | 0.919603745 | advance |
| dense | `(4621,384,1928)` | eager/ABBA | 192.649226922 | 195.289235849 | 1.013412973 | 1.014430113 | stop |
| dense | `(4621,384,1928)` | eager/BAAB | 191.940922004 | 194.560005115 | 1.013618156 | 1.014796368 | stop |
| dense | `(4621,384,1928)` | graph/ABBA | 191.522460717 | 194.422153326 | 1.015192426 | 1.015928683 | stop |
| dense | `(4621,384,1928)` | graph/BAAB | 191.488000063 | 194.499079998 | 1.015494883 | 1.015961241 | stop |
| compact8S2 | `(2048,1536,768)` | eager/ABBA | 121.081534852 | 105.940423113 | 0.874932449 | 0.875258726 | advance |
| compact8S2 | `(2048,1536,768)` | eager/BAAB | 120.993705506 | 105.810043660 | 0.874592339 | 0.874970793 | advance |
| compact8S2 | `(2048,1536,768)` | graph/ABBA | 119.641872163 | 105.611577947 | 0.882698610 | 0.883057598 | advance |
| compact8S2 | `(2048,1536,768)` | graph/BAAB | 119.603405608 | 105.577874691 | 0.882707900 | 0.883290837 | advance |
| compact8S2 | `(4621,384,1928)` | eager/ABBA | 192.412899386 | 160.833543347 | 0.835888329 | 0.837588737 | advance |
| compact8S2 | `(4621,384,1928)` | eager/BAAB | 191.786842961 | 160.816000354 | 0.838368198 | 0.839727262 | advance |
| compact8S2 | `(4621,384,1928)` | graph/ABBA | 191.355866771 | 160.834058639 | 0.840650344 | 0.841018839 | advance |
| compact8S2 | `(4621,384,1928)` | graph/BAAB | 191.286194709 | 160.850578739 | 0.840607804 | 0.841385931 | advance |

Dense used 154 registers/thread, 82,944 dynamic shared bytes and one resident
CTA/SM. Compact8S2 used 121 registers/thread, 49,152 dynamic shared bytes and
two resident CTAs/SM. Both reported zero local and zero static shared bytes.

Together with the preceding compact8S2 checkpoint on `(2048,768,3072)`, this
selects compact8S2 for all three measured heavy-NT targets. Dense-out remains a
measured retained alternative but is slower than compact8S2; dense-prism is a
valid `stop_no_retry` loss. Every retained result still has `promotion=false`.

Replay from the worktree root with:

```sh
ruby internal/perf/ada-triad-nt-sibling-two-mechanism-20260907/replay.rb
```

PRE was strict quiet/no-apps for every arm. Immediate RELEASE was non-quiet in
all four attempts and is retained; each separate five-second DRAIN was
quiet/no-apps. Candidate PTX was loaded in memory but not persisted. The
private 0700 cache was deliberately reused and was not cold.

The first host source-replay draft compared a transformed body digest with a
full composed-source digest. Root corrected that metadata-only check to hash
the actual runtime composition before these results were frozen; no CUDA
source or GPU run was changed or invalidated.
