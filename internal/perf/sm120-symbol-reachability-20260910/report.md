# RTX 5090 exact-route reachability on CUDA 13.0

Production base: `ef1408a23eeb491ae0c86456c4c6da54e4ddf154` plus the
source patch identified by `runtime-cuda130/assembly-source.sha256`.
Device: RTX 5090, CC12.0, 170 SMs, driver595.84, NVRTC13.0.

## Failure and repair

The specialized module loads, but Driver resource admission excludes
`gemm_bi_nt_sm120_tma_fma_v1_m64n128_bk16_s2` because it uses8 bytes of local
memory on this toolkit. The former selector checked module availability,
not the individual symbol, and failed when preparing NT d768-out.

The binding now retains a typed mask of known rejected exact-F32 symbols.
Automatic selection tries the measured route, then a reachable generic route;
an explicit forced request for an excluded symbol is rejected early. An
unexplained missing function remains an error. The module, accepted siblings,
resource gates and CUDA arithmetic remain unchanged.

The fallback uses `gemm_bi_nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec`, M128N64,
with **the measured split count2**. Using the generic default split count1
would change the reduction tree. No bit-equivalence is claimed between this
family and the last-resort scalar family if both specialized symbols fail.

## Actual CUDA 13.0 evidence

`sm120_exact_nt_d768_out_reachable_fallback_matches_fixed_split_cpu_bits`
passed1/1 in49.25s. It pins the production symbol, tile, grid768, numeric
contract and fixed-split output ownership. Eight distributed output values
match an independent CPU FMA reference with two384-element partials and the
fixed fold. Full output eager-repeat and eager/graph bits match; red zones
remain intact. This is a sampled arithmetic oracle, not a full CPU GEMM.

The two affected policy cells then passed the existing production matrix in
9.13s. Four records all show the same reachable M128N64/kvec/S2 route and
`eager_graph_equal:true`:

| Policy | Eager p50, us | Graph p50, us |
|---|---:|---:|
| Exact F32 | 86.888 | 88.050 |
| Allow deterministic TF32, before its new toolkit admission | 87.006 | 88.081 |

These are three-window diagnostic timings, not a cuBLAS comparison or a new
performance-admission receipt. Full raw logs, source manifest, device and
preflight records are in `runtime-cuda130/`.

## Host regressions and provenance boundaries

`red/tests.log` records five behavioral tests against neutral mask plumbing
with the original selector: two passed and three failed as expected. This
RED run preceded the later split-count correction; it does **not** prove
test-first coverage of split-count preservation.

`green/tests.log` records all five updated selector/mask tests passing on a
CUDA-feature build on Ada. It does not substitute for the RTX5090 runtime
probe above. The final loader-preservation seam and regression passed5/5
in `fix1/tests.log`: one excluded exact symbol leaves the specialized binding,
all11 accepted exact siblings and an accepted TF32 function intact.

The additional source-contract check initially failed because a new assertion
looked for error-message text after a helper had stripped string literals.
`testfix/source-contract.log` records the corrected raw-scope assertion passing
1/1; `testfix/selector.log` records the existing public selector integration
target passing1/1. Production rejection behavior was unchanged by this test
repair. The final six Rust file hashes are in `final-source.sha256`; the
runtime probe preceded the loader seam extraction and test-only correction,
which have their own scoped host receipts.

`aborted-cuda128-startup/` contains an unfinished earlier TF32 selector start
with Driver JIT caching disabled. No cell or completion receipt was produced;
it is neither a numerical failure nor a qualification pass. Subsequent runs
use private per-toolkit Driver/PTX caches with full source/compiler identities.
