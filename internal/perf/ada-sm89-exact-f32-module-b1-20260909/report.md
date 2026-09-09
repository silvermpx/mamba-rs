# Ada SM89 exact-F32 module integration B1 — 2026-09-09

Outcome: the three Batch B0 large-TN exports now have an isolated, optional
`TriadSm89ExactF32` production artifact. This commit intentionally leaves
dispatch, launch, graph routing and AUTO behavior unchanged.

## Owned exports

| Cell | Export | ABI | Frozen resource gate |
|---|---|---|---|
| TN d768-in `(2048,768,3072)` | `gemm_bi_tn_sm89_f32_n64_dual_chunk_fused_finalize_v1` | three pointers plus one 32-byte by-value parameter bundle | block128, static shared32768, local0, registers <=168, occupancy >=3 |
| TN d768-out `(2048,1536,768)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_d768_out_raw_v1` | three pointers plus four `i32` values | block128, static shared16384, local0, registers <=128, occupancy >=4 |
| TN Prism `(4621,384,1928)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_prism_raw_v1` | three pointers plus four `i32` values | block128, static shared16384, local0, registers <=128, occupancy >=4 |

The source owner is still normalized-body identical to the frozen discovery
adapters. Its literal SHA-256
`cdcb768216699f41553e73492a32d92717c62889a4a329ca1990360b361541c7`
is now checked against the actual owner bytes by the production composer.
Source tests independently reject a missing export, an extra production-shaped
foreign export and an extra valid d128-named export.

## Fail-closed integration

Artifact order is frozen as `Fixed`, `TriadScalar`, `TriadSm80`, optional
architecture finalist, optional `TriadSm89Half`, then optional
`TriadSm89ExactF32`. The exact-F32 artifact has an independent compile key,
artifact digest, compiler identity and module lifetime anchor. It neither
joins nor changes deterministic-TF32 availability.

Compilation/validation is optional on non-SM89 devices and on a rejected SM89
build. A whole-module failure records one rejection and preserves the existing
exact-F32 routes. Driver ABI and resource failures are isolated per export;
each rejected symbol is recorded separately while qualified siblings remain
bound. Module and symbol diagnostics are warn-once.

The PTX gate requires exact target `sm_89`, exactly the three sealed entries,
route-specific parameter ABI, ordered F32 FMA plus asynchronous-copy
instructions, the fused route's FP64 finalize, and no local/atomic/reduction/
tensor-core/TMA instruction family. The live loader additionally enforces
local0, exact static shared memory, register caps, maximum block size and
occupancy gates before exposing a function.

## Host verification

All verification in this phase was host/native; no GPU kernel was launched.

- no-default source contract: 13 passed;
- CUDA host module tests filtered by `sm89_exact_f32`: 4 passed;
- CUDA host kernel identity suite: 23 passed;
- CUDA host compile-only: module qualification harness, kernel identity CUDA
  test and performance matrix all built;
- CUDA host library check passed with `CUDARC_CUDA_VERSION=13000`.

The live qualification entry is the ignored test
`sm89_exact_f32_module_abi_and_resources_qualify` in
`tests/gemm_bi_sm89_exact_f32_module_qualification.rs`. Run it separately for
each selected CUDA 12.8/13.0/13.2 environment on an exclusive SM89 device. It
compiles and binds the isolated artifact, requires all three symbols to pass
ABI/resource gates, and emits identity plus per-symbol resource JSON. It does
not launch a kernel or claim AUTO route admission.

## Live module qualification

The exact `0184db89` snapshot subsequently passed on RTX 6000 Ada under
CUDA 12.8, 13.0 and 13.2. Each run bound all three exports with no ABI/resource
exclusions. This verifies module loading, not execution or AUTO admission;
the B2 route integration supplies those checks.

| CUDA | Registers, in/out/Prism | Local bytes | Active CTAs, in/out/Prism | Result |
| --- | --- | --- | --- | --- |
| 12.8 | 161 / 107 / 107 | 0 / 0 / 0 | 3 / 4 / 4 | 1 passed |
| 13.0 | 161 / 107 / 107 | 0 / 0 / 0 | 3 / 4 / 4 | 1 passed |
| 13.2 | 163 / 107 / 107 | 0 / 0 / 0 | 3 / 4 / 4 | 1 passed |

Each log contains the full compiler, artifact, header, driver and toolkit
library identities. The source digest is
`b83eea55e9cced220366c8160934340f8e4f4a59f007d632dbbcbb24dd2a503c`.
Raw logs copied from Ada and their SHA-256 values:

- [CUDA 12.8](evidence/exact-tn-module-0184db89-cuda128.log):
  `5664bcdd2a5836d57f247e88fa960799d80116ead47e076bed5e7f0d4b02f638`
- [CUDA 13.0](evidence/exact-tn-module-0184db89-cuda130.log):
  `ad7bbd62db1f9ef83032f16bad2124b204cab3061fc88bc5831325733519d4af`
- [CUDA 13.2](evidence/exact-tn-module-0184db89-cuda132.log):
  `6f94e6c947a3ad4e8a8e60f50eeba65f564a1d852770f3649015fad04eee8211`
