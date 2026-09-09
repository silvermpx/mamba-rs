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
