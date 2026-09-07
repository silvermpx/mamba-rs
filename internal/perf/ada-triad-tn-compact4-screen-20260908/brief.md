# Ada TN Prism compact-four-warp S2 ablation

Base: `070a4f54`, existing branch/worktree. Test-only; no AUTO admission.
The preceding compact-eight-warp screen passed focused bits but lost by
4.5–6.7% to actual AUTO. Its source and raw evidence remain frozen.

One change of mechanism: retain compact packed S2 staging and restore the
production four-compute-warp ownership. Logical `(m,k,n)=(4621,384,1928)`,
physical C384x1928, reduction4621; M128N64/BK32/S2, grid93,256threads,
49152 dynamic shared bytes. Four compute warps, MAtoms4, NAtoms4,
warp_m=(warp>>1)*64, warp_n=(warp&1)*32. Preserve RNA conversions, each
output's ascending reduction order, async synchronization, tails and old-C FMA.
Test symbol:
`gemm_bi_tn_test_compact_four_warp_sm80_mma_tf32_v1_m128n64_bk32_s2`.

The shared layout remains A[2][32][128], B[2][32][64], with
axis XOR ((reduction&3)<<3). Eight-warp source must remain byte-identical.
This ablation separates the compact-staging hypothesis from compute ownership;
it does not establish that either caused the previous loss.

## Focused acceptance

- Resource gate: zero local/static shared,49152 dynamic,256threads,
  occupancy at least one; record actual registers and occupancy.
- This new candidate permits one CTA/SM because grid93 underfills142SM and
  actual AUTO also has one resident CTA. It does not weaken the earlier
  eight-warp gate or the production finalist's register/occupancy cap.
- Reuse target/tail/alpha full-mantissa RNA bit comparisons, eager2/graph2,
  input immutability and guards. Reuse the repaired independent finite K0
  oracle, including signed zeros and subnormals; no public API expansion.
- CUDA13.2 only, one paired once7 screen, eager/graph x ABBA/BAAB.
  Reseed identical C+A+B before every single-GEMM timed observation; download
  outputs/inputs/guards before any next reset. Both p50 and p95 ratios must
  be below .99 in every stratum to retain a finalist.
- Actual AUTO remains TN M128N64/BK32/S3. No Fast comparison in this screen,
  no winner admission, no matrix resweep, no retry after a valid loss.
- Exact source/binary hashes, build/list, PRE/raw RELEASE/separate quiet
  DRAIN and unchanged reused private cache. Preserve all raw failures.

## Ownership and process

Carver owns only the existing TN source helper, adding FOUR_WARP_SYMBOL and
four_warp_candidate_source; Avicenna owns only the existing discovery harness,
this directory's runner/raw/report, and the sole CUDA/SSH execution lane.
Root owns this brief/handoff/commits; Beauvoir reviews read-only.
Do not edit or stage unrelated SM120 qualification WIP or its sample helper.
No new branches, pushes, kernel deletion, production or Fixed changes.

The user requested autonomous fast discovery and adaptive agents. This bounded
ablation proceeds under that instruction without another approval pause.
Disjoint file writers may work in parallel; reviewed source plus focused
tests gate acceptance. Safe isolated GPU execution may overlap read-only
review; any measurement-relevant finding invalidates the affected result.

Root pre-run verification: native full harness31/31 passes; full eight-warp
CUDA composition remains14bf43fb3303ee23fd5fb14ffc6f662db890f9dbf5ca97edb014cd46465d3b28.
Four-warp composition is22d9e88d1792b842c61960309748c398020d08f30ffa20ee2f151d6d328144ee.
An independent literal enumeration covers8192 output positions exactly once
and384 scalar-fragment instruction address families, each using32 distinct
banks. This mathematical check is not a GPU performance result.
Final main hash8b921bbb65a0310bccf14c8e575d95fce1fed1e706c049bea9f9e8a2904a2e47;
supportcfcbe3146b513fdf51ec82e95059d535df0a8fa598f583adfdd41c0bcebd1924.
The reviewer briefly saw cosmetic rustfmt bytesb71a; owner restored8b before
any remote sync/build. Only the final exact bytes are eligible for run binding.
After the completed run, root's final rustfmt check required that cosmetic
assertion formatting again. Commit source is b71a; measured source remains8b.
See report.md for the verified whitespace-only delta and fresh native checks.
