# Ada SM89 TF32 joint module plumbing

## Scope

This phase adds the isolated optional `TriadSm89Tf32Joint` module and its
artifact, PTX, Driver-ABI, and resource contracts. It does not add a physical
route, launch path, evidence cohort, or AUTO selector entry.

The artifact set now permits seven canonical members in this order:
`Fixed`, `TriadScalar`, `TriadSm80`, optional architecture-specialized,
optional `TriadSm89Half`, optional `TriadSm89ExactF32`, optional
`TriadSm89Tf32Joint`. The new module is compiled only for exact `sm_89` on
CC 8.9 and is owned independently from the existing Ada finalist, half, and
exact-F32 modules. An optional compile failure is retained as a diagnostic and
does not disable those siblings.

## Fail-closed contracts

Composition verifies the actual framed SHA-256 of both sealed source owners
before NVRTC:

- joint CUDA owner:
  `e1e8a2ad1d2d03b4d0e02730f087eab1c26cfc7712f867fbbead13b032e3624c`
- joint primitive owner:
  `c16e81fdcc4745352c97ee7daa39f2629716d7ebe38b6eea0a91393268303b0e`

PTX admission requires the exact five-export inventory. Each export is
checked against its typed static ABI and retained instruction family. Live
Driver ABI and resource checks are symbol-local: one failing symbol is
excluded without discarding its valid siblings. The gates cover zero local
memory, register cap, exact static shared memory, max-thread floor, dynamic
shared-memory availability, and the retained occupancy floor where one was
frozen. The loaded CUDA module is kept in the kernel lifetime anchors.

The ignored qualification entry is:

```text
cargo test --release --features cuda \
  --test gemm_bi_sm89_tf32_joint_module_qualification \
  sm89_tf32_joint_module_abi_and_resources_qualify \
  -- --ignored --exact --nocapture
```

It reports the complete compiler/artifact identity and the live resources for
all five symbols. It does not launch a kernel or qualify a selector.

## Native verification

The source-contract suite was rerun from an isolated temporary copy with no
CUDA feature:

```text
/Users/silvermpx/.cargo/bin/cargo test \
  --manifest-path /tmp/mamba-tf32-joint-module-native.Pb7eNG/Cargo.toml \
  --no-default-features --test gemm_bi_tf32_joint_source_contract
```

Result: 28 passed, 0 failed, 0 ignored. The 15 emitted warnings are imported
test-helper dead-code warnings. `git diff --check` passed.

No local `--features cuda`, SSH, GPU launch, correctness measurement, or
performance measurement was run in this phase. CUDA compile/load and live
resource qualification remain external gates; this report does not claim
their result.

## Root CUDA-host compile and focused tests

The immutable `0c162501` base plus this module-only overlay was copied to
`/root/mamba-tf32-joint-module.yAtdDV`. Root compiled release library tests,
`kernel_identity`, and the ignored live module target on Ada with CUDA13.2,
`--no-default-features --features cuda,cudarc/cuda-13000` and
`CUDARC_CUDA_VERSION=13000`.

The first focused run found two fixture/validator issues: a four-export-era
remaining-symbol count, and rejection of the retained `.L2::128B` async-copy
opcode suffix. The fixture now expects four remaining symbols out of five;
the validator requires the exact retained qualified opcode. Other forbidden
instruction checks remain unchanged.

After that correction, the fresh CUDA-host compile passed. Focused CPU tests
on the CUDA host passed: four module composition/PTX/ABI/resource tests, two
artifact-set tests, and one stable-discriminant test (7/7 total). Independent
runtime review approved the isolated loading/identity scope. Live Driver
loading and resources are still pending; no route or AUTO admission is claimed.

## Live CUDA13.2 module result

The `772faabe` runtime snapshot passed the ignored module test on RTX6000 Ada
CC8.9 in 107.95 s: 1 passed, zero failed. Five preflight samples each showed
0% compute/memory utilization and 48,463 MiB free. All five symbols bound,
with no per-symbol ABI/resource exclusions and zero local memory:

| Symbol family | Registers | Static shared | Dynamic shared | Active CTAs |
| --- | ---: | ---: | ---: | ---: |
| NN direct N96 (Prism) | 124 | 0 | 86,016 | 1 |
| NN baseline N96 (d768-out) | 124 | 0 | 86,016 | 1 |
| TN pre-RNA N96 | 127 | 0 | 86,016 | 1 |
| TN pre-RNA M64N64 | 83 | 0 | 49,152 | 2 |
| pre-RNA transpose | 26 | 4,224 | 0 | 6 |

Frozen CUDA13.2 module identities:

- compile/invocation: `50671423cfddb180ee60c618e638d0b1d396ecfb1203364f338c0e934b4a1381`
- artifact: `f9b43258813e70331206c1e042d1daa62f5f60143761f792a61c016dde1736ef`
- composed source: `ae4b432ccde278a0c4d4a7a742fc4a7c9e503298b01f98360356a42334f90355`
- headers: `0b64102d321829920d022e321ad0d13a722f33299378c0e9d65476c886f90e59`
- NVRTC library domain: `d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687`
- driver build: `d1edc5a5bc3e10a2688e21568dccbde5e280dc39144dd81e853843ca98b2d0e1`

The [raw module receipt](module-cuda132-772faabe.log) and
[CUDA-host build receipt](module-cuda132-772faabe-build.log) are archived.
This test loads functions but launches no GEMM. Kernel output correctness,
SASS stack/spill/issue-order proof, selector/graph wiring, paired timing and
CUDA12.8/13.0 remain pending. Do not use these module-only identities as AUTO
admission evidence.
