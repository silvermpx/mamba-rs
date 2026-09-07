# Ada Fixed finalist assembly — phase1 qualified, AUTO44 checkpoint

This directory records integration of three retained inference candidates, not
a completed AUTO promotion or a claim that inference beats cuBLAS Fast in every
cell. Current production selection remains tuning revision **44**, numerical
ABI **5**, schedule **8**. Work is on the existing
`codex/gemm-bi-triad-sm80` branch; no Triad source is part of this assembly.

## Scope and admission

| Forced route | Intended AUTO cell `(M,K,N)` | Baseline route |
| --- | --- | --- |
| `Tf32RnaM128N96S3` | TF32 E0 `(2048,2304,768)`, no bias | `Tf32RnaM128N128S3` |
| `TcM64N64Sm89S3` | F16 D0 `(2048,768,2304)`, no bias | `Tc128Sm89Swizzle` |
| `TcM128N64Sm89S2` | F16 E0 `(2048,2304,768)`, no bias | Swizzle on CUDA12.8/13.0, Pipeline on13.2 |

Each intended AUTO row requires matching-toolkit
resource/ABI/raw-bit/batch-invariance checks and fresh forced-versus-current
measurements. The source supports safe forced prefixes and N96 tails, while
AUTO promotion is restricted to measured literal cells on CC8.9/142SM with a
known matching compiler library and an independently admitted kernel holder.

Use CUDA12.8,13.0,13.2 independently. Timing reuses
`fixed_ada_forced_rungs_paired_precision_cublas`, both eager and graph paths,
both A-F-V/V-F-A orders, 21 windows per stratum. This protocol is **not ABBA**.
Candidate/current-AUTO must have worst-stratum median <=0.985 and p95 <=1.0,
in addition to correctness and resource checks. The timed TF32 denominator is
actual cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32`; F16 uses native-half cuBLAS.
PEDANTIC is an untimed numerical reference, not a substituted Fast comparator.

After qualifying rows, revision45 will select only those rows and retain
independent old-route fallbacks. One combined actual-AUTO qualification and
101-window closure follows; numerical ABI5 and schedule8 must stay unchanged.
No such promotion or closure has happened yet.

The final cap-adapted source now passes the six focused functional tests on
**all three toolkits:18/18 exact test exits0**. See
`phase1-final-functional-cuda{128,130,132}`. Each toolkit's N96 log contains
224 unique full-mantissa/exceptional prefix/view groups; each D/E forced bit
test, N96 unsafe/K0 test, holder/resource test and combined fail-closed test
passes. Root independently replayed exits, group uniqueness and source hashes.
All nine forced21 invocations are complete, with the decisions below.
CUDA13.2's original seven-source command receipt omits the transitive generator
hash; `phase1-final-functional-cuda132/binding-supplemental.json` closes that
metadata gap with explicitly observed-after-run source/binary/cache bindings.
The original receipt is preserved; this is not a repeated GPU qualification.

## Final phase1 timing decisions

Ratios below are the worst quantile across the four eager/graph and start-order
strata, computed from same-index raw triplets. Lower is better. Root and the
independent reviewer replayed all 36 records / 756 triplets, with no rejected
records, matching source bindings, exit0 and quiet release for every invocation.

| CUDA | Route | Candidate/AUTO p50 | Candidate/AUTO p95 | Candidate/Fast p50 | Candidate/Fast p95 | AUTO promotion row |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 12.8 | N96 TF32 E0 | 0.773559 | 0.773876 | 1.108488 | 1.108717 | yes |
| 13.0 | N96 TF32 E0 | 0.773548 | 0.773922 | 1.108527 | 1.108804 | yes |
| 13.2 | N96 TF32 E0 | 0.852896 | 0.853096 | 1.114358 | 1.114809 | yes |
| 12.8 | D F16 | 0.977016 | 1.010233 | 1.054040 | 1.065569 | no |
| 13.0 | D F16 | 0.987990 | 1.005549 | 1.044600 | 1.061296 | no |
| 13.2 | D F16 | 0.951482 | 0.956599 | 1.056790 | 1.064503 | yes |
| 12.8 | E F16 | 1.039732 | 1.039996 | 0.958305 | 0.959323 | no; old AUTO faster |
| 13.0 | E F16 | 1.040018 | 1.040444 | 0.957924 | 0.959722 | no; old AUTO faster |
| 13.2 | E F16 | 0.970421 | 0.971247 | 0.965707 | 0.966718 | yes |

These are exactly five literal toolkit/cell admissions, not an all-toolkit
requirement for a route. A valid loss stops that toolkit/cell; no losses were
retried. CUDA12.8/13.0 half D/E retain their current AUTO. No bias, BF16, other
shape, architecture, or unknown-library preference is inferred from this batch.
N96 and D still lose to Fast. E's fresh integration comparator is faster/slower
as shown; historical discovery used a different timing protocol and is not a
before/after Fast baseline. Both use non-PEDANTIC native-half GemmEx with F32
accumulation; no timed PEDANTIC substitution occurred.
The E ratio reversal relative to discovery comes from the Fast denominator
(historical roughly53–54us, current61–62us), not a hidden candidate speedup.
Discovery shifts active half pointers128B and uses20-operation/20-node ABBA
windows; this harness uses base allocations and calibrated paired windows.
Discovery did not record the Fast symbol. Current E captures
`ampere_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_nn`; identical physical
vendor algorithm selection across those protocols cannot be established.

Phase2 will select the five admitted rows with tuning revision45, then verify
actual AUTO and once101 against the old forced route and Fast. This phase1
checkpoint deliberately leaves AUTO44 unchanged.

## Observed integration checks

- `phase1-module-red`: preintegration production compiled, then the new module
  composition assertion failed. This is behavioral RED, not a missing-symbol
  compile failure. The writer had already prepared independent CUDA fragments;
  this receipt does not establish that every production edit followed RED.
- `phase1-performance-red`: old production compiled with the new string-based
  registry assertion, which failed because N96 was absent.
- `phase1-green-host`: CUDA13.2 release builds passed four exact tests: module
  composition; independent per-symbol PTX/ABI/resource rejection; architecture
  source contract; performance force registry. This is host test evidence,
  not GPU correctness evidence.
- `phase1-green-host-extra`: two additional exact host tests passed on the
  built performance binary: rejection of mutated finalist graph descriptors
  and architecture-specific force inventory. No GPU or rebuild was involved.
- `phase1-build-cuda132`: correctness and half-pipeline test binaries compiled.
- `phase1-gpu-cuda132`: N96 forced full-mantissa/IEEE-exceptional prefix/view/
  eager/graph corpus passed, as did its unsafe-input/K0 test. Root independently
  counted **224 unique groups**: 168 tail, 56 E-boundary; 112 finite and112
  exceptional. The first record shares a line with the libtest test name;
  parsers must locate the marker rather than require it at column zero.
- That same GPU invocation stopped at the D F16 holder's register cap **before
  any D bit case**. E and subsequent checks were not run. This is a resource
  admission failure, not a timing loss or a demonstrated numerical failure.
  PRE/POST/RELEASE receipts for that invocation are quiet0%/0%, no compute apps.

## Half integration regression and bounded repair

The first production integration forwarded runtime K/N/leading dimensions,
whereas the discovery wrappers constrained them to exact constants. It also
added safe M-tail bodies. Source comparison supports lost compiler
specialization as a cause of increased registers; resource comparison after
repair is required to distinguish it from the remaining tail-body overhead.

The raw, unmodified-module census is in `resource-census-before/resources.json`:

| Symbol family | Registers | Local/static bytes | Dynamic shared bytes | Threads | Active CTAs/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| N96 TF32 | 128 | 0 / 0 | 86016 | 256 | 1 |
| D F16 N64/S3 | 114 | 0 / 0 | 49152 | 128 | 2 |
| E F16 N64/S2 | 152 | 0 / 0 | 49152 | 128 | 2 |

D/E discovery register caps were96/132. Both fail those caps, while zero-local
and two-CTA placement remain intact. The raw census binds the cache and PTX.
Its immediate release had5% activity and is retained as failed; a separate
quiet release is recorded rather than relabelling the first receipt.

The single authorized repair restores exact device preconditions and literal
K/N/strides in both full-M and safe-prefix paths. It preserves the three
exports, ascending arithmetic, shared layouts and resource caps. Repaired
half CUDA SHA256:
`5a4c85184fb4a0263ae03cbe7f2b5e8f954b32ce3138012ad8505274d3769681`.
Updated architecture source-test SHA256:
`3bc72a3b4b9c79fe1c85a1085eec1adba4e4d62d3f17ea8c624077342149a3c0`.
The narrow repair independently passed source review. The CUDA13.2 after-census
in `resource-census-after/resources.json` reports D110 and E132 registers;
local/static bytes remain0, and both still have two active CTAs. N96 remains
128 registers/one CTA. Thus the repair restores E's discovery resource count;
D remains over the original96 cap. Attribution of the remaining D delta to
the coexisting safe-prefix body is an inference, not an isolated experiment.

At that intermediate checkpoint, `phase1-e-bits-cuda132` passed E's full forced prefix/view/raw-bit/graph
test and the combined invalid dtype/pointer/bias/shape rejection test, each
exit0, with quiet PRE/RELEASE. D correctness and timing were still stopped then;
the final cap-adapted all-three functional and timing results above supersede
that intermediate stop.

The identical repaired source was then censused on CUDA12.8/13.0, under
`resource-census-cuda128` and `resource-census-cuda130`. Root independently
replayed all nine records:

| Toolkit | N96 registers | D F16 registers | E F16 registers |
| --- | ---: | ---: | ---: |
| CUDA12.8 | 136 | 110 | 130 |
| CUDA13.0 | 136 | 110 | 130 |
| CUDA13.2 | 128 | 110 | 132 |

Every record has zero local/static bytes and the exact shared/thread/active-CTA
contract shown above. Independent review and root's replay therefore authorize
one bounded host admission change: N96 cap128→136 and D cap96→110; E remains132.
These are the exact all-toolkit observed maxima, with negative boundary tests
updated and no CUDA-body change. The discovery register count is not a numerical
contract or proof of speed. This enables subsequent correctness/timing, without
bypassing either; a valid timing loss still stops that literal AUTO row. No
second CUDA geometry experiment is authorized in this repair.

## Frozen test inputs and review

Main-owned phase1 test hashes are:

- `tests/gemm_bi_fixed_performance.rs`:
  `bd73a7389536732f08162abc043f092964f9124cc92d03dfd762335befa6fe42`.
- `tests/gemm_bi_fixed_correctness.rs`:
  `83d261680287b06421ef9dca1db71c2880a6629a90009c438e59a293ea013854`.
- `tests/support/fixed_full_mantissa.rs`:
  `234d4d87dd479780a3cacee34e5f7f1e868e0465536caba3866a46a16a39cb22`.

Independent initial frozen source review was Spec PASS / Quality PASS, with
per-symbol eligibility isolated from global module inventory. The narrow
specialization repair is reviewed separately. Source review is not a substitute
for live resource, correctness, timing or post-AUTO evidence.

Historical discovery sources and reports are preserved in their separate
directories; their timings are not relabelled as production-integration data.
