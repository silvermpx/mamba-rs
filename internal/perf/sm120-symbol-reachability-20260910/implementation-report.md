# SM120 exact-F32 symbol reachability implementation report

Date: 2026-09-10

## Outcome

Implemented the bounded selection-level repair on production base
`14b721362d3d9e28638cbb3995b4c1b30c5f0d30` (review HEAD
`ef1408a23eeb491ae0c86456c4c6da54e4ddf154` contains evidence/report-only
commits after that base).

The specialized SM120 module now carries a typed `u16` mask for only the
literal `SM120_FMA_ROUTE_SPECS` entries rejected by Driver resource admission.
Automatic exact selection filters the measured route before trying the
existing generic route, and filters the resulting generic route as well.
Forced excluded exact routes fail before preparation with the exact symbol and
`excluded on this toolkit`. The loaded function map and late launch lookup
remain fail-closed for unexplained absence.

For a rejected measured route, the generic tile/load geometry inherits the
measured route's `splits`. Thus CUDA 13.0 NT d768-out changes from the rejected
M64N128/non-kvec/S2 symbol to the retained M128N64/kvec symbol **with S2**, not
S1. This preserves `ScalarFmaFixedSplitFoldV1`, the 0..384 and 384..768
ascending-FMA partials, and the fixed p0+p1 fold. Unmeasured generic requests
retain their original S1 behavior. If both measured and generic symbols are
excluded, the pre-existing scalar floor remains; no cross-family bit-equivalence
claim is made for that last-resort case.

## Scope and guarantees

- `contract.rs`: compact inventory-position mask nested in
  `Tf32QualifiedModule`; its empty default preserves synthetic and non-SM120
  fixtures.
- `modules.rs`: derives the mask only from loader exclusions whose symbol is an
  exact-F32 literal inventory entry. TF32 exclusions, unknown strings, and
  sibling exact entries do not set unrelated bits. The live specialized
  binding is retained.
- `dispatch.rs`: measured-before-generic filtering, measured split preservation,
  scalar floor when both are excluded, and early forced-route rejection.
- `launch.rs` and `tests/gemm_bi_tf32_selector.rs`: mechanical empty-mask
  additions to complete binding literals.
- `tests/gemm_bi_tf32_contract.rs`: extends the existing source contract for
  both filters and forced rejection, plus one ignored CUDA 13.0 production
  fallback arithmetic probe.

No CUDA source, resource/local-memory gate, ABI, cohort identity, tuning table,
winner table, performance harness, or launch-map fallback was changed. A single
excluded NT entry cannot suppress the other eleven exact-F32 inventory entries
or any SM120 TF32 entry.

## TDD evidence

Behavioral RED was frozen after neutral mask/plumbing only and run by root with:

```text
cargo test --locked --release --features cuda --lib sm120_exact_reachability -- --nocapture
```

Receipt: `internal/perf/sm120-symbol-reachability-20260910/red/tests.log`, exit
101. Exactly five tests ran: loader-mask and unrelated-route tests passed; the
three intended assertions failed because measured selection still returned
M64N128/non-kvec/S2, the double exclusion still returned that measured arm, and
forced exclusion still returned `Ok`.

That immutable RED predates the numeric review which required retaining the
measured S2 partition count. Its fallback assertion expected the existing
generic S1 route. It is valid evidence for the exclusion-aware selection and
forced-route defects, but it is **not** a pre-implementation RED for the final
S2-preservation rule. No such S2 RED receipt exists. The S2 rule was added from
the CUDA source contract: S1 performs one ascending FMA chain, while S2 performs
two ordered partial chains followed by a fixed p0+p1 fold. The final S2 behavior
is covered by GREEN and by the independent CUDA 13.0 CPU-bit runtime receipt
below.

After the production filtering change and the S2-preservation correction, root
ran the same command on CUDA 13.2/Ada: **5/5 passed** after a fresh compile.
Receipt:
`internal/perf/sm120-symbol-reachability-20260910/green/tests.log`. This proves
mask propagation/selection behavior without SM120 GPU execution.

Round-one review found that the original loader unit test exercised only the
pure mask derivation helper. The amended test now passes the function map,
binding, exclusions, and qualification result through the same small retention
seam used by the specialized loader. It proves that one known exact exclusion
retains `availability.specialized`, preserves the accepted function map and an
unrelated TF32 sibling, masks only that inventory position, and leaves all 11
accepted exact-F32 siblings present and unmasked. This is plumbing-only; CUDA
source, route choice, launch parameters, and arithmetic are unchanged.

Root verified that amendment on CUDA 13.2/Ada with:

```text
cargo test --locked --release --features cuda --lib sm120_exact_reachability -- --nocapture
```

The fresh release build completed in 52.76s and the focused suite passed
**5/5** (772 filtered out). Receipt:
`internal/perf/sm120-symbol-reachability-20260910/fix1/tests.log`; its frozen
source manifest is
`internal/perf/sm120-symbol-reachability-20260910/fix1/source.sha256`.

The adjacent source-contract test then exposed a test-only mistake: the new
rejection-message literal had been passed through `source_mask`, which
deliberately blanks string literals. It failed 0/1 at the assertion helper;
receipt:
`internal/perf/sm120-symbol-reachability-20260910/fix1/source-contract.log`.
The correction keeps the route variant and exclusion-field structure checks on
the masked braced function scope, and checks the exact rejection text on that
same raw braced scope. The production rejection and its behavioral message
assertion are unchanged. Root's fresh CUDA-feature reruns passed:

- `exact_policy_never_selects_tf32_and_allow_policy_falls_back_to_the_exact_family`:
  **1/1 in 0.03s** after a 9.51s release compile; receipt
  `internal/perf/sm120-symbol-reachability-20260910/testfix/source-contract.log`.
- The complete `gemm_bi_tf32_selector` integration target:
  **1/1 in 0.00s**; receipt
  `internal/perf/sm120-symbol-reachability-20260910/testfix/selector.log`.

The exact tested file hashes are frozen in
`internal/perf/sm120-symbol-reachability-20260910/testfix/source.sha256`.

Local non-build checks passed:

```text
rustfmt --edition 2024 --check <six changed Rust files>
git diff --check
```

## CUDA 13.0 runtime proof

Run the focused production probe on the RTX 5090 CUDA 13.0 host:

```text
cargo test --locked --release --features cuda --test gemm_bi_tf32_contract \
  sm120_exact_nt_d768_out_reachable_fallback_matches_fixed_split_cpu_bits \
  -- --ignored --exact --nocapture
```

The probe requires CC 12.0/170SM and NVRTC 13.0. It asserts one production
M128N64/kvec node, grid x=768 (384 output tiles times S2), the fixed-split
numeric contract and ownership, then checks eight distributed outputs against
an independent host `f32::mul_add` reference over the exact two 384-element
slabs and p0+p1 fold. It also requires full-output eager repeat equality,
eager/graph equality, and intact red zones. On CUDA 13.0/RTX 5090 it passed
**1/1 in 49.25s**. The log also records that the measured symbol was excluded
for 8 bytes of Driver JIT local memory. Receipt:
`internal/perf/sm120-symbol-reachability-20260910/runtime-cuda130/runtime.log`.

The older source-matched raw test
`cuda_experiment::candidate_is_exact_guarded_and_graph_stable` in
`gemm_bi_scalar_wide_microtile_experiment` covers the same NT M128N64/kvec/S2
arithmetic against sampled CPU chains, but is not suitable as the CUDA 13.0
receipt: it runs all 27 cells and pins CUDA 13.2 resources for all 12 symbols,
including the known rejected entry.

The two affected CUDA 13.0 matrix cells
(`f32_policy_exact/nt/d768_out_proj/contiguous` and
`f32_policy_allow_tf32/nt/d768_out_proj/contiguous`) also passed in the focused
matrix run: **1/1 test in 9.13s**, four emitted rows total. Both eager and graph
paths selected
`gemm_bi_nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec`, tile `[128,64]`, grid
`[768,1,1]`, with `eager_graph_equal=true`. Receipt:
`internal/perf/sm120-symbol-reachability-20260910/runtime-cuda130/matrix-two/gemm_bi_deterministic_performance_matrix.log`.
No CUDA 13.0 full-matrix rerun is needed for this repair.

## Final file hashes

```text
6082d214a63fd51e423cde3b2054b8395e9df9122bd288b9699fa6f69db3575a  src/mamba_ssm/gpu/gemm_bi_triad/contract.rs
feeda91d908ebe3fb448dfe59bca8c166090bfb6e5ce21cf38dc0e74703ad623  src/mamba_ssm/gpu/gemm_bi_triad/modules.rs
47b627130374e3ab982172e51633522d9794e8c3caf9b14b64cff6683297a328  src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs
52d88d8b16ae41b96be307ef4343d132f452b878af70859744f4d84649f68ce3  src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
489fef58977d6db393640b1b4367c98b3e1bb4ba1d1c36112b7b00ca42ba214c  tests/gemm_bi_tf32_contract.rs
6953bed17458fde150322dfc8ea07f509e15c194b1b44f2dc3993850718d6d46  tests/gemm_bi_tf32_selector.rs
```

No index changes or commits were made by this owner.
