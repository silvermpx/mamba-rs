# Task 900 report: artifact-preserving Inference rename

## Status

GREEN implementation is frozen for root-owned focused GPU rerun and review.
Root authorized GREEN after the corrected real
composer baseline passed all three targets on Ada and the parser test produced
the expected RED rejection of `inference`.

Base commit: `b47ff4050b09a50c0779a84fed22142423ee80a2`.

## Tests-first checkpoint

- Extended
  `context.rs::bi_gemm_family_environment_accepts_only_semantic_family_names`
  with canonical `inference` and mixed-case/whitespace
  ` \nInFeReNcE\t`, while both still expect the old
  `BiGemmFamily::Fixed` variant. Existing default, legacy `fixed`, unknown and
  non-Unicode cases remain present.
- Added the private real-composer test
  `modules.rs::inference_composed_source_identity_is_frozen`. It calls
  `compose_module_source_for(ModuleKind::Fixed, ...)` for portable `sm_80`,
  `sm_89` and `compute_120`, then checks literal byte lengths and real
  `FramedSha256::bytes` digests.
- Corrected tests-only snapshots:
  - `task-900-red-context.rs`:
    `a70d7d5635a355739db685da79b49d92274da0a163d142a648ea85cf2c755e56`
  - `task-900-red-modules.rs`:
    `db0e898eb3a551de1c2940fc82db89a995073441f4985035da6e6a3bc542d4cd`
- `git diff --check`: pass.
- Root-observed parser RED: exit 101; `inference` returned the expected
  not-recognized error advertising the old canonical set.

The preliminary audit's externally reconstructed composer fixture failed
before production edits. A diagnostic run of the unchanged real Rust composer
gave these corrected baselines:

| target | bytes | SHA-256 |
|---|---:|---|
| portable `sm_80` | 615645 | `fab86c118b49e88eaa566ef1c4322d039641fa11fbfa3c2dde402470b97b1d11` |
| `sm_89` | 723257 | `8ab4752c6f4b766486db09d22ddfc99df93dc1eb0507177f7edf92cc731233f4` |
| `compute_120` | 719063 | `8a9a5186c2a7763fab0027ab03417f6189730eac103e57d4863becd4a8a90a71` |

The UTF-8-length hypothesis was falsified because both lengths and hashes
differed. The exact external reconstruction defect was not recoverable from a
recorded script, so its values were not blessed. The real `sm_89` and
`compute_120` hashes independently match already-committed full Inference
performance packets at `732c1146`; root also confirmed all 60 tracked
CUDA/header files equal `b47ff405`. Initial failure and diagnostic evidence are
retained under `internal/perf/inference-rename-20260910/`. No production byte
or admission constant was changed to satisfy the test.

## Implemented coherent rename mapping

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs` -> `gemm_bi_inference.rs`
- `gpu::gemm_bi_fixed` -> `gpu::gemm_bi_inference` (no old module alias)
- `kernels/gemm_bi_fixed/` -> `kernels/gemm_bi_inference/`
- `BiGemmFamily::Fixed` -> `BiGemmFamily::Inference` with ordinal/behavior
  unchanged; `inference` canonical and `fixed` retained only as an input alias
- `FixedTile` -> `InferenceTile`
- `FixedSm120HalfTile` -> `InferenceSm120HalfTile`
- `FixedFwdOperands` -> `InferenceFwdOperands`
- `FixedShape` -> `InferenceShape`
- `fixed_forward` -> `inference_forward`
- `fixed_forward_with_tile` -> `inference_forward_with_tile`
- `fixed_forward_f32_legacy_baseline` ->
  `inference_forward_f32_legacy_baseline`

All active Rust callers, test imports, live filesystem references, and current
release documentation were updated to the Inference-facing names. Existing GPU
test target filenames were deliberately not renamed. No three-mode API or
default-policy change was introduced.

## GREEN freeze evidence

- Frozen source hashes:
  - `src/mamba_ssm/gpu/context.rs`:
    `9a257fe9414c821158ad93fb1947b37c511da76f28123a3b4d4a15e236dad23d`
  - `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs`:
    `8c5459399fbb8a8b961d29d05c740afd3f48a004c7e9b2d75ac79e61bfa4b12c`
  - `src/mamba_ssm/gpu/gemm_bi_inference.rs`:
    `6658fe9e9649bca979cec10d25dd2b803ea956e2fca8e7c1c2c7d85d6bb1b54a`
- Root independently compared all 60 tracked CUDA/header files to base; all
  matched byte-for-byte, including the 22 files relocated into
  `kernels/gemm_bi_inference/`.
- Before this freeze, `rustfmt --edition 2024 --check` passed on the modified
  production Rust files and `git diff --check` passed.
- No CUDA build, staging, commit, branch, or index operation was performed by
  this implementer.

Root's initial GREEN run passed all-target CUDA compilation after supplying
three missing existing log fixtures, the host composer/path/parser checks, 56
Inference tests with 2 ignored, 20 SM120 cohort tests, 23 kernel-identity tests,
and 84 non-CUDA tests. The live Ada
`tf32_cohort_binds_on_this_board` check then failed because its stale assertion
required the portable SM80 M128N128/S3 winner for every wide NN shape; the
accepted d768-out winner was instead
`TriadSm89Tf32Joint:gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1`.

The bounded test-only correction in `tests/gemm_bi_tf32_cohort_binding.rs`
adds `TriadSm89Tf32Joint` to the TF32 classifier and asserts the exact accepted
toolkit map: d768-in remains portable SM80 on CUDA 12.8/13.0/13.2; d768-out is
joint N96 on all three; Prism remains portable SM80 on 12.8/13.0 and uses joint
direct-N96 on 13.2. Small-shape behavior and production code are unchanged.
The frozen corrected test-file SHA-256 is
`ee940e50caf60989cc8b66cc563781e2461cc70fbf01169893910b1c0f2e1247`;
`rustfmt --edition 2024` and its scoped `git diff --check` passed.

## Intentionally retained legacy identity vocabulary

The GREEN patch retains `ModuleKind::Fixed = 1`, `ArtifactSetIdentity::fixed`,
`BackendSet::FIXED`, `NumericContractSet::FIXED_*`, physical backend and
resolved numeric contract `*Fixed*` identifiers, fixed-order/fixed-fold terms,
CUDA symbols/GBF namespace, compiler/tuning/composer revisions, cache and
qualification constants, old receipt/snapshot/benchmark labels and existing
qualification target filenames. Every `SourceFragment.logical_name` and
emitted legacy `kernels/gemm_bi_fixed/...` `#line` boundary remains frozen;
only its `include_str!` physical path moved.

Specifically retained identity terms include
`PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1 = 22`,
`ResolvedNumericContract::ScalarFmaFixedSplitFoldV1 = 19`,
`ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1 = 20`, the
`LastCtaPerOutputTileFixedSplitK{2,4}ReduceV1`,
`OneThreadPerOutputElementFixedSplit{M,K}ReduceV1`,
`OwnerCtaPerOutputTile{StreamKFixedOrder,FixedSplitFold}V1` contracts,
`HalfTriadPolicy::AllowStreamKFixedOrderV1 = 1`, private `Fixed*` ABI/map/cache
types, and `FIXED_*` constants. `fixed` remains accepted only as a compatibility
selector alias. CUDA `GBF`/`_fixed_` symbols and logical source names are
unchanged.

## Concerns / remaining evidence

No concrete rename requirement is known to remain incomplete. All-target
CUDA-feature compilation, focused host checks, and the broad GREEN cohorts
listed above passed under root ownership. Outstanding evidence is the root-owned
focused rerun of the corrected live Ada cohort test and independent review.
Until that rerun completes, this report does not claim the full GPU suite
passes. Any rerun/review findings remain for the same implementer to fix from
this frozen state.
