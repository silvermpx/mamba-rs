# Deterministic GEMM Benchmarks

All numbers: RTX 6000 Ada (sm_89, 142 SMs), CUDA 13.2, driver 595.45,
`--release`, `--test-threads=1`, quiet GPU. The GEMM layer is shared by
Mamba SSM and Mamba-3 SISO — these results apply to both architectures.

## Tiers and contracts

| flag | tier | contract |
|---|---|---|
| (off) | cuBLAS | f32 → TF32 tensor cores; bf16/f16 → GemmEx `COMPUTE_32F_PEDANTIC` (CUDA cores, f32 accumulate). Run-to-run stable on one machine, NOT batch-invariant, no stability across cuBLAS versions. |
| `MAMBA_RS_BATCH_INVARIANT=1` | scalar deterministic | custom fixed-reduction-order kernels. Training bit-identical across runs on every dtype; inference decode strictly all-M invariant via `matvec_bi` (KL ≈ 1e-12). bf16/f16 outputs are bit-identical to "upcast → f32 kernel → RNE downcast". Which family serves the forward is selectable — see the row below. |
| + `MAMBA_RS_BI_GEMM_FAMILY=triad\|fixed` | family selector | `triad` (`kernels/gemm_bi_triad.cu`, default): the multi-tile dispatcher, all three operand layouts, per-bucket batch invariance (same dispatch bucket → row 0 bit-identical across M). `fixed` (`kernels/gemm_bi_fixed/`): the forward serving family — a ladder of bit-identical tiles (16-row thin, 64, 128, wide 128×256) with `SPLIT_K=1` everywhere, batch-invariant BY CONSTRUCTION (every rung produces the same bits per element, so tile choice is pure scheduling and no bucket exists to cross); on Hopper/Blackwell it routes to per-architecture rungs (`wgmma`, `tcgen05`) that form their own bit families, behind a first-use self-check. A backward requires `triad`. The family is part of `ctx.gemm_route()`, so a flip after a CUDA-graph capture is refused at replay. |
| + `MAMBA_RS_BI_TENSOR_CORES=1` | tensor-core deterministic | `mma.sync.m16n8k16`, f32 accumulators, no atomics/splits. OWN numeric contract (TC reduction tree ≠ scalar FMA chain) — but runs are bit-identical to each other (incl. CUDA Graph capture/replay) and the forward is STRICTLY batch-invariant across all M. The tile ladder — 16-row thin, 64×64, 128×128 and wide 128×256 — is BIT-IDENTICAL per output element (same ascending BK=64 reduction slabs, same mma chain, same tail zero-fill), so the shape-only tile routing never changes output bits. |

Accuracy cross-checks: bf16 scalar-tier training trajectory vs cuBLAS
PEDANTIC cosine 0.999999976 (5 steps); f32 vs TF32 0.999999996. TC tier
vs f32 reference on quantized inputs: bf16 cos 0.9999986, f16 0.99999998;
TC dW (f32 accumulate) cos 1.000000000.

## Training step cost (`MambaTrainer`, ms/step)

`tests/gemm_bi_determinism.rs::bench_sgemm_bi_vs_tf32`

| model | dtype | cuBLAS baseline | scalar deterministic | + tensor cores |
|---|---|---:|---:|---:|
| d128 ×2L, B=16 T=64  | f32  | 2.053 (TF32) | 2.637 (1.28×) | — |
| d128 ×2L, B=16 T=64  | bf16 | 2.124 (PEDANTIC) | 2.540 (1.20×) | 2.199 (1.04×) |
| d128 ×2L, B=16 T=64  | f16  | 1.878 (PEDANTIC) | 2.572 (1.37×) | 2.228 (1.19×) |
| d256 ×4L, B=16 T=128 | f32  | 8.724 | 11.235 (1.29×) | — |
| d256 ×4L, B=16 T=128 | bf16 | 9.029 | 10.359 (1.15×) | 9.109 (1.01×) |
| d256 ×4L, B=16 T=128 | f16  | 7.737 | 10.386 (1.34×) | 9.146 (1.18×) |
| d768 ×4L, B=8 T=256  | f32  | 24.090 | 32.218 (1.34×) | — |
| d768 ×4L, B=8 T=256  | bf16 | 25.706 | 28.462 (1.11×) | **21.650 (0.84×)** |
| d768 ×4L, B=8 T=256  | f16  | 24.328 | 28.460 (1.17×) | **21.693 (0.89×)** |
| d1536 ×2L, B=4 T=256 | f32  | 14.123 (TF32) | 21.576 (1.53×) | — |
| d1536 ×2L, B=4 T=256 | bf16 | 17.772 | 19.406 (1.09×) | **12.486 (0.70×)** |
| d1536 ×2L, B=4 T=256 | f16  | 16.761 | 19.740 (1.18×) | **12.818 (0.76×)** |

Ratios are vs the cuBLAS baseline of the same dtype. Bold = deterministic
training FASTER than cuBLAS. With the Tile64 family and BK=64 staging
the TC tier is at-or-near parity even on the smallest models (d128 bf16
1.04×, d256 bf16 1.01×) and 16–30 % faster than cuBLAS from d768 up; at
d1536 the bf16 TC step (12.49 ms) also beats the f32 TF32 baseline
(14.12 ms). The remaining f16 small-model gap (1.18–1.19×) is the
cuBLAS-f16-PEDANTIC baseline being unusually fast at tiny sizes, plus
non-GEMM kernels dominating those steps.

## Tensor-core tier — GEMM level (bf16, µs)

`tests/gemm_bi_tc.rs::bench_tc_vs_scalar_paths`, vs the scalar
deterministic tier on the same shape (BK=64 staging):

| shape (M, K, N) | fwd scalar → TC | dW scalar → TC | dX scalar → TC |
|---|---:|---:|---:|
| 2048, 768, 3072 | 293.2 → 84.1 (**3.49×**) | 400.2 → 101.5 (**3.95×**) | 424.6 → 83.6 (**5.08×**) |
| 4096, 1536, 3072 | 1131.8 → 352.8 (3.21×) | 1450.4 → 311.8 (4.65×) | 1212.7 → 301.5 (4.02×) |
| 2048, 768, 512 | 112.7 → 17.6 (**6.40×**) | 138.4 → 24.8 (5.59×) | 93.6 → 26.7 (3.51×) |

84.1 µs at M2048 K768 N3072 ≈ 115 TFLOPS bf16; the M4096 forward
reaches ~144 TFLOPS in an isolated sweep (`step0` instrumentation:
83.7 µs / 267.8 µs on the two shapes). BK=64 staging halves the per-CTA
barrier/wait_group boundaries vs BK=32 and bought +8–11 %; deeper
pipelining was measured FLAT.

### The measured denominator (`tests/gemm_denominator_probe.rs`)

Every "% of peak" figure here divides by numbers measured on the box
itself, not by spec-sheet estimates:

- **Tensor-pipe ceiling: 335.9 TFLOPS** bf16 with f32 accumulation
  (null-memory mma.sync issue-rate kernel, register fragments only).
  The f16-accumulate twin runs at 0.95x — on this part f32
  accumulation does NOT halve tensor throughput, so a halved-accumulate
  ceiling model would be wrong here.
- **cuBLAS tensor-core bf16, measured here for the first time**
  (the crate's fast arm): 118–158 TFLOPS across six fat training
  shapes — 35–47 % of the pipe.
- **The deterministic Tile128 against it, event-timed on the same
  shapes**: at parity (91–93 %) where K is large and the CTA grid
  fills whole waves (M2048 K768 N3072: 133.6 vs 146.4; M4096 K1536
  N3072: 148.0 vs 158.4), and 1.35–1.8x behind on the small-K and
  wave-cliff shapes (M4096 K768 N3072: 80.8 vs 144.9; M2048 K1536
  N1536: 80.9 vs 141.9; M2048 K2304 N768: 87.5 vs 118.0; M2048 K768
  N2304: 82.7 vs 131.0).

So the real, addressable gap is not a uniform kernel-quality deficit:
it concentrates where the wave count cliffs (cuBLAS does not feel the
cliff; a deterministic kernel without split-K must cure it with tile
geometry) and where K is small (fewer slabs amortize less staging).
Those two mechanisms — a constant-area fragment-reuse tile and
wave-aware tile choice — are the program.

**Family staging and the wide rung.** The probe's "deterministic
Tile128" arm drove the training family's forced entry; the inference
family's twin — byte-identical, differently staged — measures well
ahead of it on several fat shapes (M4096 K768 N3072: 141.9 µs /
136.6 TFLOPS against the training tile's 239 µs), so part of the gap
above was one family's staging, not the contract's price. The
fragment-reuse program then closed most of the rest: the 128x256
wide rung (warp tile 64x64, byte-identical to the 128-tile, censused
over a 24-point promotion grid) wins its wave-arithmetic band by
7-12 %, reaching 143.5 TFLOPS at (4096,768,3072) — parity with the
measured cuBLAS tensor-core rate on that shape — and 132.9 at the
(2048,1536,1536) wave-cliff cell against cuBLAS's 141.9. The
dispatcher routes to it only where the grid holds wave efficiency;
narrow-N and cliff shapes keep the 128-tile.

## Tile64 family — small/narrow shapes (bf16, µs)

`tests/gemm_bi_tc.rs::bench_tc64_vs_tc128_small_shapes`. The 64×64-tile
twins quadruple the CTA count on grids that underfill the GPU at
128×128, and cover the 64..127 output-dim band the 128 gate excluded:

| shape (M, K, N) | op | Tile128 | Tile64 | scalar bi |
|---|---|---:|---:|---:|
| 1024, 128, 512 (d128 in_proj) | fwd | 7.6 | **5.5** | 17.9 |
| | dW | 27.7 | **12.6** | 82.1 |
| | dX | 15.3 | **6.5** | 19.0 |
| 1024, 256, 128 (d128 out_proj) | fwd | 10.2 | **4.6** | 12.9 |
| | dW | 27.7 | **12.5** | 43.5 |
| | dX | 7.3 | **3.9** | 14.7 |
| 2048, 256, 1024 (d256 in_proj) | fwd | 18.0 | **16.6** | 49.9 |
| | dW | 50.8 | **22.0** | 101.7 |
| | dX | 27.0 | **13.1** | 64.5 |

Dispatch (`tc_pick_tile`): 128-tiles when the grid has ≥ 72 CTAs,
64-tiles otherwise and for the 64..127 band. Legal under the strict
all-M invariance contract because the two families are bit-identical
per output element (`tc64_and_tc128_bit_identical`). Narrow projections
of every model size (x_proj N=80, dt_proj K≤96) ride tensor cores via
Tile64 — that is why even d768/d1536 steps improved when it landed.

### Event timing corrects the wall-clock table

The wall-clock rows in the table above bracket each launch with a host
sync, folding launch and synchronization overhead into every reading -
a first-order error at these kernel durations. Re-measured with CUDA
events in alternating groups (`tests/gemm_ladder_sweep.rs`), the
`2048, 256, 1024` forward shape reads Tile128 14.6 us vs Tile64 18.6 us:
Tile128 ahead by over 20 percent, where the wall-clock table showed it
behind by 8. The dispatcher's 72-CTA threshold routes that shape to
Tile128 - correctly. The decode-band Thin16 verdicts survive event
timing with wider margins than first recorded (M=64: 9.2 vs 11.3 us at
768x2304, 12.3 vs 18.4 us at 1536x1536).

The open decode-band question is also settled: the Thin16 crossover's
responsible axis is N, not K. At M=128, Thin16 beats Tile64 at every
probed K (768/1536/2560) while N <= 1536 (margins 13-35 percent), and
loses at N >= 2304 for every K; on (1536, 1536) the M crossover sits
between 128 and 160. The shipped rows <= 64 rule therefore leaves a
banded win on the table for M in (64, 128] at N <= 1536 - the
d1536-class projection band. A threshold change is a scheduling-only
edit (the rungs are bit-identical per element), but it is promoted only
on a sweep of the family it routes, under the alternating-group
protocol; the evidence here was read on the forced triad rungs.

### Tile64 at the prefill shapes (RTX 6000 Ada, min of 3 runs)

The prefill routing question is settled by measurement: a wave-efficiency
model predicted Tile64 could win the big serve projections (Tile128 sits
at 4.17 waves / 0.834 wave efficiency on the widest one), and it does
not — redundant B traffic and lower arithmetic intensity outweigh the
tail wave everywhere at prefill M.

| M, K, N | Tile128 | Tile64 | verdict |
|---|---:|---:|---|
| 4621, 384, 1928 | **56.8** | 75.2 | Tile128, 1.32x |
| 4621, 768, 2304 | **109.1** | 132.9 | Tile128, 1.22x |
| 4621, 1928, 384 | **53.4** | 60.4 | Tile128, 1.13x |
| 2048, 768, 2304 | 63.3 | 64.6 | tie (sign flips between runs) |
| 2048, 2304, 768 | **60.2** | 65.1 | Tile128, 1.08x |

The shipped dispatcher already routes all five to Tile128 — correct.
Any remaining tile-choice value lives in the small-M / small-grid band
(the `2048, 256, 1024` row above where Tile64 measured 8.4 % faster
under the 72-CTA threshold), and a threshold change there must ride the
event-timed ABBA protocol, not these `Instant`-based readings.

## Scalar tier — typed Big / upcast-fallback cost (bf16, µs)

`tests/gemm_bi_typed_parity.rs::bench_upcast_fallback_tax`. Big-routed
shapes run the native typed kernel; split-K/Slim shapes run "upcast →
f32 kernel → RNE downcast" (bit-identical by contract):

| shape (M, K, N) | route | f32 kernel | typed | overhead |
|---|---|---:|---:|---:|
| 2048, 768, 3072 | native Big | 245.7 | 294.3 | 19.8 % |
| 4096, 1536, 3072 | native Big | 917.4 | 1328.5 | 44.8 % |
| 2048, 768, 512  | fallback (Slim) | 77.2 | 88.3 | 14.4 % |
| 256, 384, 512   | fallback (split-K) | 13.4 | 20.0 | 49.4 % |

The native Big path matches the fallback's speed while eliminating the
f32 upcast scratch (~0.5 GB at 2.8b mixed) and 3 extra launches per
GEMM. With the TC tier on, none of this is on the bf16/f16 hot path.

## Reproducing

```sh
# training step, all 3 dtypes × {cuBLAS, scalar bi, +TC}:
cargo test --features cuda --release --test gemm_bi_determinism \
  bench_sgemm_bi_vs_tf32 -- --ignored --nocapture --test-threads=1

# TC vs scalar at GEMM level (fwd/dW/dX):
cargo test --features cuda --release --test gemm_bi_tc \
  bench_tc_vs_scalar_paths -- --ignored --nocapture --test-threads=1

# Tile64 vs Tile128 on small/narrow shapes:
cargo test --features cuda --release --test gemm_bi_tc \
  bench_tc64_vs_tc128_small_shapes -- --ignored --nocapture --test-threads=1

# typed upcast-fallback tax:
cargo test --features cuda --release --test gemm_bi_typed_parity \
  bench_upcast_fallback_tax -- --ignored --nocapture --test-threads=1
```

Contract tests (non-ignored, run in the default suite): bit-identity of
training across runs (`gemm_bi_determinism.rs`), typed bit-parity vs the
f32 triad incl. a 60-shape dispatch-gate boundary sweep
(`gemm_bi_typed_parity.rs`), TC determinism / strict all-M invariance /
accuracy / cross-tile bit-identity / gate boundary sweep / launch-reality
geometry (`gemm_bi_tc.rs`), cross-batch inference parity
(`hf_batch_parity.rs`, `extreme_edge_coverage.rs`).

## Run-to-run bit determinism across GEMM tiers (RTX 5090)

`tests/parallel_run_determinism.rs`: two identical 4-step training runs in
the parallel-scan regime (T > threshold, bf16), master weights compared
bit-for-bit, one test per tier:

| tier | selected by | run-to-run |
|---|---|---|
| cuBLAS (crate default) | no flags | bit-identical |
| batch-invariant scalar | `set_batch_invariant(true)` | bit-identical |
| batch-invariant tensor-core | + `set_bi_tensor_cores(true)` | bit-identical |

A companion test pins that the three tiers do NOT agree with each other
bit-for-bit: separate reduction orders are separate bit families by
design. The `--ignored` `print_run_digests` test emits an FNV-1a digest of
the final weights per tier for A/B-ing two BUILDS across a kernel edit
(used to prove the `_Pragma("unroll")` restoration bit-neutral on all
three tiers).

## The frozen SM-count cell is a bit-family key

The scalar tier's split-K and split-M gates key on a frozen constant
(`NUM_SMS = 142`, the Ada RTX 6000 SM count) rather than a device query.
A split changes the reduction order, so the gate thresholds are part of
the numeric route identity: querying the real SM count at init would make
two boards behind the same `sm_XX` target (RTX 5090 with 170 SMs, RTX
5080 with 84; B200 with 148, floorswept parts with fewer) produce
different bits for the same shape — a per-SKU bit family that no
guarantee can state or test. The constant therefore stays a frozen table
cell, pinned by a tripwire test next to its definition; a future
architecture that needs its own wave-fill value gets its own frozen cell,
new goldens, and a route-identity entry. The tensor-core ladder is immune
by construction: it has no split-K and its tile picker keys only on
output dimensions between bit-identical kernels.

## CUDA-13 cuBLAS compute-mode probe — the pedantic pin re-examined (RTX 5090, cuBLAS 13)

`tests/cublas_compute_probe.rs` (`--ignored`, TSV artifact): bf16-input
GemmEx cells V0-V5 (compute type x handle math-mode bits) and f32-input
cells V6, each against an on-device fp64 reference computed from the SAME
bf16-widened bits. 130 shape-family combos, 20-repeat bit-stability on
the breaker/tied-head shapes. Headline rows (mean relative error / ms,
`normal` family, steady-state timings):

| shape | 32F_PEDANTIC (V0) | 32F any math (V2-V5) | f32 32F (V6d) | f32 EMULATED_16BFX9 (V6e) |
|---|---|---|---|---|
| M8 K16384 N2048 | 1.86e-6 / 0.33 | 6.42e-5 / 0.042 | 4.12e-6 / 0.18 | 4.12e-6 / 0.083 |
| tied head M8 K2048 N50304 | 9.49e-7 / 0.32 | 8.91e-6 / 0.14 | 1.11e-6 / 0.37 | 1.11e-6 / 0.30 |
| tied head M512 | 2.28e-6 / 3.1 | 1.51e-5 / 0.60 | 2.28e-6 / 1.6 | 2.28e-6 / 1.6 |

Findings, in decision order:

1. **`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` is a dead end**:
   V4/V5 match V2/V3 to the BIT in every metric on every shape — the
   flag changes nothing for bf16-input GemmEx on this stack.
2. **The accuracy gap PEDANTIC vs 32F is real and persists on CUDA 13**
   (~10-35x mean relative error, growing with K). The falsification
   guard ("V2 clean everywhere => inconclusive") did NOT trigger. The
   pedantic pin keeps its justification; **no default flip.** Version
   window: reproduced on cuBLAS 13 / sm_120, first observed on
   cuBLAS 12.8 / sm_89.
3. Handle math mode is inert under PEDANTIC (V1 == V0 bit-for-bit) —
   confirms the device.rs claim.
4. Every cell is run-to-run bit-stable (20 repeats) — determinism is not
   what separates the modes; accumulation accuracy is.
5. **`CUBLAS_COMPUTE_32F_EMULATED_16BFX9` (V6e) matches true-fp32
   accuracy bit-for-bit in error profile at up to ~2x the speed** on
   f32-input GEMMs. A candidate for the f32 cuBLAS
   lane (a new bit family, so it would need a versioned re-route; not
   adopted).
6. TF32 (V6t) is the worst accuracy option at scale (mean 1.9e-4 at
   K=16384) — reaffirms keeping it opt-in only.

Consequence for bf16 TRAINING speed: the cuBLAS default lane stays
pinned PEDANTIC; the speed lever for bf16 training remains the
batch-invariant SGEMM-BI tier (fp32 fixed-order accumulation, no cuBLAS)
and its occupancy work.

## Family comparison at a prefill shape (f32)

`tests/classifier_gemm_tier_bench.rs::classifier_shapes_cublas_vs_sgemm_bi_vs_gemm_bi`

Vision-classifier projections, f32, M = 4621 rows per page (a batched row
shows whether a dispatch bucket boundary is ever crossed). RTX 6000 Ada.

| GEMM | M | cuBLAS | `triad` | `fixed` |
|---|---:|---:|---:|---:|
| input_proj K=1024 N=384 | 4 621 | 34.1 µs | 157.3 µs (4.61×) | 140.9 µs (4.13×) |
| input_proj K=1024 N=384 | 18 484 | 152.7 µs | 520.0 µs (3.41×) | 509.0 µs (3.33×) |
| in_proj K=384 N=1928 | 4 621 | 78.4 µs | 219.4 µs (2.80×) | 218.6 µs (2.79×) |
| in_proj K=384 N=1928 | 18 484 | 287.8 µs | 780.8 µs (2.71×) | 967.9 µs (3.36×) |
| out_proj K=768 N=384 | 4 621 | 26.8 µs | 119.6 µs (4.46×) | 103.9 µs (3.88×) |
| out_proj K=768 N=384 | 18 484 | 104.3 µs | 387.0 µs (3.71×) | 335.4 µs (3.22×) |

The two families are close: `fixed` leads on input_proj and out_proj,
`triad` on the wide-N in_proj at the batched row. Both differ from cuBLAS
by the same 1.0e-4–1.8e-4 (they agree with each other more closely than
either agrees with cuBLAS), and reruns are bit-identical.

These are isolated GEMM timings. End to end on the same model the
deterministic f32 route costs +30% per page (18.6 → 24.2 ms), because
the scan, not the projections, dominates that architecture — the ratio
a caller should plan against is the end-to-end one, not the GEMM one.
With the tensor-core ladder the bf16 deterministic page removes that
tax entirely: 10.9 ms/page against the 11.8 ms non-deterministic
cuBLAS f32 page and 20.0 ms for the deterministic f32 route
(`tests/m3_prefill_bench.rs`, pooled-graph replay, RTX 6000 Ada).
