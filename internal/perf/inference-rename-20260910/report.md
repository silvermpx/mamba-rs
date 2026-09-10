# Inference rename verification — 2026-09-10

Base: `b47ff4050b09a50c0779a84fed22142423ee80a2`.
This is a naming and artifact-preservation check, not a new speed measurement.
Three-mode API/default changes are not part of this patch.

## Source and identity

The Rust module and 22 CUDA/header files move from `gemm_bi_fixed` to
`gemm_bi_inference`. Root compared all 60 tracked CUDA/header files directly
against the base commit: all are byte-identical. Compiler logical filenames,
CUDA symbols, versioned contract/module identifiers and admission keys stay
unchanged. All local `src`, `kernels` and `tests` files match the final frozen
GPU-run manifest in `green-binding/source.sha256`; Cargo inputs and the three
existing compile-time test fixtures match their manifests as well.

Public role names use Inference; the environment accepts `inference` as the
canonical family token and retains `fixed` as an input alias. The old Rust
module is not retained as a second public path.

## Tests-first evidence

The initial externally reconstructed golden was wrong before production
changes. `initial-red/` and `diagnostic/` preserve that failure and the actual
unchanged Rust composer output. Corrected tuples were checked before the
rename in `baseline-red/`: composer PASS, parser expected RED rejecting the
new `inference` spelling. The Ada and SM120 hashes independently match the
already committed Inference performance packets measured at `732c1146`.

Real-composer tuples before and after the rename:

| Target | Bytes | SHA-256 |
|---|---:|---|
| `sm_80` | 615645 | `fab86c118b49e88eaa566ef1c4322d039641fa11fbfa3c2dde402470b97b1d11` |
| `sm_89` | 723257 | `8ab4752c6f4b766486db09d22ddfc99df93dc1eb0507177f7edf92cc731233f4` |
| `compute_120` | 719063 | `8a9a5186c2a7763fab0027ab03417f6189730eac103e57d4863becd4a8a90a71` |

## GREEN checks

CUDA-feature commands ran on RTX 6000 Ada with CUDA 13.2. Each live case had
an idle/free-memory preflight. No full performance tournament was repeated.

| Check | Result | Evidence |
|---|---|---|
| `cargo check --locked --release --features cuda --all-targets` | PASS; existing discovery warnings remain | `green-initial/all-targets-check.log` |
| Composer identity, source boundaries, family parser | 3 passed | `green-initial/` |
| Inference module tests | 56 passed; 2 SM120 live tests ignored | `green-initial/inference-selectors.log` |
| SM120 TF32 host cohort tests | 20 passed | `green-initial/sm120-cohorts.log` |
| Kernel identity integration tests | 23 passed | `green-initial/kernel-identity.log` |
| `cargo test --locked --lib`, without CUDA | 84 passed | Root tool output, fleet tree `abed64ba`, 0.76 s tests |
| Ada TF32 joint toolkit-winner map, eager/repeat/graph | PASS, 12.29 s | `green-remaining/live::sm89_tf32_joint_post_admission_auto_uses_the_toolkit_winner_map.log` |
| Ada Inference RNA wide route and graph | PASS, 4.63 s | `green-remaining/fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph.log` |
| Ada Inference half prefix/view/graph bits | PASS, 20.07 s | `green-remaining/fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits.log` |
| Ada Inference exact-F32 prefix/view/graph bits | PASS, 157.60 s | `green-remaining/fixed_sm89_exact_n64_auto_prefix_view_graph_bits.log` |
| Ada TF32 cohort binding, corrected existing expectation | PASS, 5.07 s | `green-binding/tf32_cohort_binds_on_this_board.log` |

Both final GPU runners exited 0 and completed source/fixture integrity checks.
The 102 focused CUDA-host checks and 84 non-CUDA tests are distinct from the
five live GPU checks. This rename does not add fresh live RTX 5090 evidence;
it preserves the already measured compiler identities and checks its host
selectors. No cross-device or all-toolkit speed claim is inferred.

## Problems found and resolved

The first all-target build lacked three existing compile-time log fixtures
under `internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/`.
Supplying their exact local bytes in a new immutable snapshot made the build
pass. `missing-fixtures/` retains the initial diagnostic. Release packaging
must keep or relocate those fixtures when excluding internal evidence.

The initial live TF32 binding test incorrectly forced an old portable SM80
winner for every wide Ada NN shape and omitted `TriadSm89Tf32Joint` from its
classifier. Production selected the accepted winner. Only the test was
corrected, using the existing qualified toolkit map: d768-in portable for all
three toolkits; d768-out joint N96 for all three; Prism portable on 12.8/13.0
and joint direct-N96 on 13.2. `green-initial/` retains the failing expectation;
`green-binding/` contains the focused rerun. Production hashes were unchanged.

Existing discovery-test unused/dead-code warnings are deferred to the explicit
release cleanup phase; no `allow(dead_code)` was added. Independent task review
is recorded separately before this task is marked complete.
