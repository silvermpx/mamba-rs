# Ada exact-F32 TN d128 assembly

2026-09-10. Production source baseline184a2673; existing isolated worktree.
This is integration of the two retained direct-fold bodies, not new discovery.
Both routes are admitted to production AUTO for the three qualified toolkits.
Module, pre-admission, actual-AUTO and independent source review are complete.

The two CUDA bodies remain unchanged. Owner SHA256:
`c684cfcc1165af0ad5cdc2e9e0e1c5c4d7d2986d48fb37b6c192a2dd40718310`.
Composed source digest:
`8835a910a8e5805fb63c9b1d6edf28f0c2c7d33a2d085c8f30d1b33a93d82652`.
The new optional `TriadSm89ExactF32D128` module does not alter the existing
large-TN, half, Fixed, TF32 joint, or portable source compositions.

## Module qualification

Frozen snapshot: `/root/mamba-d128-assembly.mD8m3k`. Root uses one Ada executor
and five consecutive idle/free-memory samples per live test. Target/private
cache directories remain separate per toolkit; the new d128 module itself was
not previously present in those caches.

Initial compilation exposed the missing backend30-to-module12 mapping in
`context.rs`; it was repaired explicitly, with the private route revision and
logical-F32 support. The next live test declined the new module because its
static PTX checker incorrectly expected `.u32` for the float alpha parameter.
The correction requires exactly three `.u64` pointers, one `.f32` alpha and
three `.u32` dimensions. The Driver ABI is still seven fields/40 bytes at
offsets0,8,16,24,28,32,36. Sealed source validation is now called before
composition; five new unused-validator warnings disappeared without suppression.
Both failed receipts are retained, and neither was a kernel performance loss.

Freeze-A2 source: `modules.rs` SHA256
`4a02eeb5a7e0acd2d32ff7f390a2232cba2f4dd69a546134e6b14b754d38d513`,
`context.rs` `4e3abdef43138b5da25f4fdbf963d4ea5c8be9cc8877882b94846ecfedf39a97`,
source adapter `00892e2dd773927ff0df9f93e67b2d6ebfa1479465684c886238d35d48e2ad64`,
module test `2894629cafc0d76ac4a2625ef35a17f4d955b99c17e0c86a2e2b095d769b7b1f`.

| Toolkit | Module test | d128-in registers / active CTAs | d128-out registers / active CTAs |
|---|---|---:|---:|
|12.8|PASS,5.69s|112 / 8|80 / 12|
|13.0|PASS,6.01s|112 / 8|80 / 12|
|13.2|PASS,5.22s|105 / 8|80 / 12|

Both routes use64 threads, local/static shared memory0, and dynamic shared
memory4096/3072 bytes. Original register caps112/96 and minimum8 active CTAs
are unchanged. These tests verify source/PTX/Driver ABI, actual module binding,
both symbol resources and artifact/compiler identities; they do not launch
the GEMM, verify numerical outputs or measure performance.

All three exact module identities and six per-symbol resource rows are also
collected in `module-identities.json`, transcribed from the raw JSON receipts.

## Route integration checkpoint

Freeze-B added the production dispatcher, eager/prepared launch and forced
qualification bindings. Its CUDA13.2 library/test compilation found one stale
`ScalarLaunchFacts` initializer in `scalar_nn_tn_tests.rs`, missing the three
new optional d128 fields. The failed compile is retained in
`cuda132/route-compile-initial/`; it ran no GEMM or timing screen. The source
writer is repairing that fixture before focused numerical qualification.

The shared F32 qualification facade keeps its existing trailing red zones.
A separate d128 raw-launch probe will supply two-sided guards with 256-byte
aligned active origins, without changing shared allocation/binding semantics.
These are distinct checks; a facade trailing-guard pass alone is not described
as a two-sided guard pass.

The fixture repair compiled successfully: CUDA13.2 library and three test
targets, 42.45s, no warnings. Focused CUDA-feature static checks passed
12 library tests (one unrelated SM120 GPU test ignored),16 source contracts
and23 identity tests.

The first selector run stopped at the K0 test because the physical facade
requires positive M/N. The corrected K0 probe uses the established public
raw backward-dW path, retaining eager/captured bits and two-sided guards.
This was a test-interface repair; production arithmetic did not change.
The failed receipt is preserved under `cuda132/pre-k0-facade-failure/`.

## Pre-admission numerical and performance qualification

Final pre-admission selector SHA256:
`7b6bd3d9d734394d48154e559420609f31151711d253d9aec8ad3c5c07c16d1f`.
Both exact retained symbols passed on all three toolkits. Finite outputs
match an independent CPU SplitM64 arithmetic oracle; exceptional payload/sign
bits match an independently launched literal GPU SplitM64 reference.
Non-unit raw alpha, seeded C, immutable A/B, repeat/eager/graph, K0, contract
neighbors and exact one-node/no-scratch identity passed. Raw probes check
256-byte leading/trailing guards at256-byte-aligned active origins; facade
trailing guards are checked separately.

| Toolkit | Test duration | d128-in candidate / prior AUTO p50 | d128-out candidate / prior AUTO p50 |
|---|---:|---:|---:|
|12.8|12.00s|0.3646–0.3788|0.4982–0.5018|
|13.0|11.16s|0.3639–0.3784|0.4979–0.5017|
|13.2|11.69s|0.3777–0.4009|0.4954–0.4993|

Ranges span eager/graph and ABBA/BAAB in the once7 screen. Lower ratios mean
less time. The reference is the literal previous actual AUTO
`TnSplitM { m_chunk:16, chunks:64 }`, not cuBLAS Fast. Every once3 and once7
p50/p95 ratio is below0.99. Root independently recomputed all48 screens,
240 brackets and960 positive arm observations; ratio/quantile replay error0.
These results support replacing old AUTO, not a new cuBLAS Fast claim.

Pre-admission raw logs are in `cuda{128,130,132}/pre-admission/`.

## Actual AUTO and closing review

Exactly the three measured compiler/artifact cohorts were admitted. Actual
AUTO selects both direct-fold symbols with one node and no split scratch:
CUDA12.8 PASS5.74s,13.0 PASS6.23s,13.2 PASS6.13s. All six route receipts verify
exact finite/exceptional outputs, repeated eager/graph execution, physical
identity, guards and immutable inputs. These post-admission tests do not time
AUTO against itself. Raw receipts are in the matching `post-admission/` folders.

Independent review required complete forbidden-PTX-family checks, the specified
inventory/ABI/resource/sibling-exclusion mutations, and a non-vacuous legacy
artifact-set digest regression. Those are now covered. Test-first CUDA checks
failed on the pending admission and missing forbidden-family gate, then passed
13 focused library tests; source/identity contracts passed16/23. The final
test-only addition checks `red`, `redux` and scoped atomic/reduction spellings;
its exact validator test passed1/1. No resource cap or CUDA body was changed.
The known absent-d128 digest is pinned from independently encoded legacy-v1
framing, rather than recomputed through the implementation under test.

Final source formatting was verified against the immutable compiled snapshot:
for each of the15 owned Rust files, the local file was byte-identical to
`rustfmt` output of the tested original. The subsequent module change adds
only four tested forbidden-opcode fixtures. Candidate timing was not repeated
for these formatting/test-only changes. Final module Rust SHA256:
`a085d0c65ba341e3e7719ba8de4665dbe47beec6018592cb382a494a503e610f`;
dispatcher SHA256:
`fac0f12dd7cb76ff5f0d1ea1bdc05b50209ac00eb41efbe4d5f456cb3ab235e1`.

The general performance matrix also gains the missing name for ModuleKind12;
that exhaustive-match repair changes no measurement behavior. Its CUDA13.2
5090 matrix passes132 eager/graph rows, retaining every prior physical route.
Separate lower-toolkit 5090 validation remains outside this Ada admission.

Closing independent review approves both specification compliance and code
quality. This completes retained d128 assembly, not the full0.7.0 release gates.
