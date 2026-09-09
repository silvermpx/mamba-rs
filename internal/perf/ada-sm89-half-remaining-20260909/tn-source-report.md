# Ada SM89 half-TN source promotion

Date: 2026-09-09. Scope: source assembly only. The active
`TriadSm89Half` owner, module registry, artifact identity, dispatcher, launch
code, graph path, and AUTO cohort are unchanged.

## Promoted retained bodies

The frozen owner contains two physical M64N64/BK64/S2 families, each expanded
for F16 and BF16:

- compact B-XOR:
  `gemm_bi_tn_sm89_m64n64_bk64_s2_compact_bxor_v1_{f16,bf16}`;
- register-pipelined plus paired `float2` epilogue:
  `gemm_bi_tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_{f16,bf16}`.

The marked family bodies are normalized-body identical to the retained
discovery generators in `triad_half_tn_compact_source.rs` and
`triad_half_tn_vec2_epilogue_source.rs`. Only the export prefixes changed.
Macro cleanup is outside the parity regions, allowing both families to share
one future module without redefinition leakage. No plain-regpipe, test,
experimental, stopped-geometry, or other foreign export is present.

All four specs pin the existing seven-field Driver ABI (40 parameter bytes),
tile 64x64, BK64, S2, block 128, static shared 32,768 bytes, dynamic shared 0,
local 0, register ceiling 128, and occupancy gate 3. These are qualification
requirements; this source-only checkpoint does not claim a fresh compiler
receipt.

The standalone composition prepends the exact dependency set used by the
retained harness: typed prelude, Triad contract, common helpers, epilogue, and
MMA16 helpers. A second adapter API omits the typed prelude so the same fragment
can later be appended to the already self-contained `TriadSm89Half` owner.
Dependency sources are fail-closed by frozen FNV-1a digests, and their SHA-256
receipts are published below. The native contract also proves that each needed
definition occurs exactly once and that dependency fragments export no CUDA
kernel.

## Qualification provenance

The CUDA13.2 actual-AUTO batch in
`half-cuda132-trace-fixed-run.log` passed all seven missing-half rows. All six
TN candidates were exact against the independent forced-TC64 oracle in eager
and graph execution, preserved guards and inputs, and beat actual public AUTO
in all four timing strata. Both d768-in tournaments selected regpipe+vec2.
That receipt authorizes source promotion only; CUDA12.8/13.0/13.2 compilation,
PTX inventory, per-symbol ABI/resources, module identity, and post-admission
AUTO gates remain root-owned follow-up work.

## TDD evidence

RED, before the owner and adapter existed:

```text
$ cargo test --no-default-features --test gemm_bi_sm89_half_tn_source_contract --no-run
error: couldn't read tests/../src/mamba_ssm/gpu/gemm_bi_triad/sm89_half_tn_source.rs:
No such file or directory (os error 2)
exit 101
```

GREEN after the minimal source-only implementation:

```text
$ cargo test --no-default-features --test gemm_bi_sm89_half_tn_source_contract
running 20 tests
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Scoped formatting and `git diff --check` passed for the three code/test files.
No CUDA feature, NVRTC, SSH, or GPU command was run.

## Frozen SHA-256

```text
066a3a28b2051e60bc3d368279dae11a8dbacdc7b52d2e53b3a97336dd237daa  kernels/gemm_bi_triad/sm89_half_tn.cu
c5c90f644b48c35f26a89ccea3e13d77b64100de0578645137f8d2b7a12ad289  src/mamba_ssm/gpu/gemm_bi_triad/sm89_half_tn_source.rs
1a49d16bdefca4ab585dccf8e4466ee24575c2c43fe3f6e42558549afeb8933e  tests/gemm_bi_sm89_half_tn_source_contract.rs
```

Dependency receipts:

```text
0c9b2345c643417406d75403df11f6fb96af7ce82f198ee551086f7c19020948  kernels/_typed_prelude.cuh
a8df19198d57a15d1fb84ea32f085c35ddd56eb407cb1bd172f1810eba12df6f  kernels/gemm_bi_triad/contract.cuh
ae91ca724cb6ae8a14753b3c7260e18e64ea4027cd36b104322d00d34d287a88  kernels/gemm_bi_triad/common.cuh
88f198be891f1316ee540f210e014b55d61e14263d71cd710c17762fb9d1674a  kernels/gemm_bi_triad/epilogue.cuh
8906c2da4db43c1b51a1ebed5d3ab8c9da29c4c4f8a248131ead8df7149d54f1  kernels/gemm_bi_triad/mma16.cuh
```

The files are intentionally left untracked and unstaged for the root-owned
index and commit workflow.
