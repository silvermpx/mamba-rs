# GEMM mode API: bounded 0.7.0 proposal

2026-09-10. Design only; no implementation or GPU validation performed.
The owner has selected **Deterministic as the default**, with explicit
**CublasFast** and **CublasPedantic** alternatives. This is settled. The choices
below resolve the remaining implementation details; they are recommendations,
not claims about current behavior.

## Facts checked in the current tree

`context.rs` currently stores independent batch-invariant, fast-compute and
cuBLAS-TF32 cells. `new*` starts with `(false, false, true)`, a hybrid vendor
configuration. `apply_env_route` parses individual flags and invokes setters
sequentially. `device.rs::create_cublas` enables TF32 handle math.
`dtype.rs::compute_type` returns PEDANTIC for BF16/F16 but ordinary 32F for F32.
`disable_tf32` uses DEFAULT_MATH, asserts on failure, and cannot reverse itself.

`tests/gemm_bi_typed_parity.rs` commonly configures `set_batch_invariant(true)`
then `set_bi_tensor_cores(...)` then `set_fast_gemm(false)`. Turning the last
call unconditionally into a vendor-mode selector would break this setup.
The context's resources are immutable and retained by graphs; replacing its
cuBLAS handle is not a small atomic-transition solution.

The surface audit in `internal/release-mode-surface-audit-20260910.md` remains
the authoritative list of direct-cuBLAS gaps and high-level constructor seams.
An enum alone does not close those gaps.

## Canonical types and signatures

Use the existing public context module, also re-exporting `GemmMode` beside
other GPU configuration types. Do not create another mode type per model.

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GemmMode {
    #[default]
    Deterministic,
    CublasFast,
    CublasPedantic,
}

impl GemmMode {
    pub fn parse_env_value(value: &str) -> Result<Self, String>;
    pub const fn as_str(self) -> &'static str;
}

impl GpuCtx {
    pub fn new_with_mode(device: &GpuDevice, mode: GemmMode)
        -> Result<Self, String>;
    pub fn new_with_state_cap_and_mode(
        device: &GpuDevice, state_cap: usize, mode: GemmMode,
    ) -> Result<Self, String>;
    pub fn set_gemm_mode(&self, mode: GemmMode) -> Result<(), String>;
    pub fn gemm_mode(&self) -> GemmMode;
}
```

Existing `new`/`new_with_state_cap` delegate with `GemmMode::default()` and
continue ignoring ambient route variables. Existing `new_from_env*` resolve
the full environment once, before construction, then build that configuration.
Explicit-mode constructors ignore environment mode selectors. Keep the crate's
current `Result<_, String>` convention for this slice.

Store one `Cell<GemmMode>` as the authority. Derive the compatibility getters
and `GemmPolicy` booleans from it; do not retain independently writable mode
booleans. Keep custom family/precision/tier cells separate. No `Custom`, `Auto`,
`Legacy`, `Mixed`, or optional/unknown fourth mode is introduced.

| Mode | batch_invariant | fast_gemm | tf32 | Handle math | GemmEx compute |
|---|---:|---:|---:|---|---|
| Deterministic | true | false | false | PEDANTIC_MATH, dormant | forbidden in public GEMM dispatch |
| CublasFast | false | true | true | TF32_TENSOR_OP_MATH | COMPUTE_32F for F32/BF16/F16 |
| CublasPedantic | false | false | false | PEDANTIC_MATH | COMPUTE_32F_PEDANTIC for F32/BF16/F16 |

**Precision decision:** make the public Pedantic name literal, including F32.
Handle PEDANTIC_MATH covers legacy SGEMM; GemmEx receives explicit pedantic
compute. Do not describe DEFAULT_MATH + COMPUTE_32F as the same contract.
Put context-dependent compute selection in the central BLAS helper, not a
global reinterpretation of dtype-only helpers used by standalone probes.
Fast permits vendor optimizations/TF32; it does not guarantee a particular
kernel, speedup, or numerical result. Driver/vendor overrides can restrict it.
These meanings follow NVIDIA's definitions of
[math modes and compute types](https://docs.nvidia.com/cuda/cublas/index.html).

## Custom numeric policy and execution role

Mode changes preserve `F32TriadPolicy`, `HalfTriadPolicy`, tensor-core enablement,
and `BiGemmFamily`. They are dormant under vendor modes, and become effective
again under Deterministic. The release defaults are `ExactScalarFmaV1`,
`TiledParityV1`, and **tensor-core permission true**, enabling the qualified half
tensor-core paths immediately. This last default intentionally changes the old
context's false value; explicit `set_bi_tensor_cores(false)` still selects the
scalar/custom fallback policy. Modes do not reset an explicit policy override.
Never infer custom TF32 permission from CublasFast or `tf32()`.

Generic contexts default to `BiGemmFamily::Triad`, which supports all operation
roles. High-level inference constructors explicitly select `Inference` for NN;
training constructors select `Triad`. An inference context's TN/NT/tied-head
operations still dispatch to the appropriate Triad operation. Family selection
is not permission to send TN/NT to an NN kernel. A trainer environment requesting
`family=inference` should return a clear role-conflict error, not silently change
the training family. A generic context retains the existing explicit family
override and safe backward-to-Triad behavior.

Keep all frozen scalar/MMA/TF32, reduction, physical-launch, artifact and
compiler identities intact. `GemmMode` can be uniquely derived from the three
canonical mode booleans, so no redundant enum field or enum discriminant needs
to enter the versioned identity. The new vendor pedantic contract must get a
new version/identity rather than reinterpret `CUBLAS_POLICY_V1`; update the
dispatch policy revision/hash for the semantic change. This does not authorize
renaming old custom contract identifiers or changing kernel/compiler digests.

## Fallible, coherent transition

Use an internal reversible `set_cublas_math_mode_checked(math) -> Result<(),
String>` for both construction and transitions. Validate the proposed complete
configuration and stream capture status before mutation. Reject changes during
active capture or route recording. A no-op same-mode request remains a no-op.

Compute the target on the stack. Read the previous actual handle math, perform
the single cuBLAS math update, and only on success publish the canonical mode
cell. No fallible work follows publication. No allocation, compilation, cache
clearing, or resource replacement belongs in this transition. A mode change
after completed capture is allowed; existing exact route comparison rejects
replay while its recorded mode differs, and recapture is the normal next step.

Do not assume an arbitrary failed foreign call left its handle unchanged:
on update failure, query and, if necessary, restore the previous math mode.
Return the original error only once the old state is verified. If restoration
or verification fails, mark the context unusable and reject subsequent GEMM
dispatch/capture/replay with that diagnostic. This exceptional invalid-context
state is an error, not a fourth usable mode. The implementation must centralize
that guard across supported dispatch paths. This gives failure-atomic usable
state without falsely promising that an unrecoverable CUDA failure is usable.
Pure injected math-backend tests must cover this path before GPU probes.

Changing away and back to the exact captured configuration may reuse a graph
under the existing equality contract; this proposal does not introduce an
unrelated permanent invalidation generation. Rejection tests compare differing
live and captured configurations.

## Deprecated setters: explicit transition table

Keep the existing signatures for one migration cycle, mark them deprecated,
and route each through `set_gemm_mode`. Because they currently return `()`, a
failed transition panics with a migration diagnostic after the coherent/error
handling above. New application code uses the fallible canonical setter.

| Deprecated call | From Deterministic | From CublasFast | From CublasPedantic |
|---|---|---|---|
| set_batch_invariant(true) | Deterministic | Deterministic | Deterministic |
| set_batch_invariant(false) | CublasPedantic | CublasFast | CublasPedantic |
| set_fast_gemm(true) | CublasFast | CublasFast | CublasFast |
| set_fast_gemm(false) | Deterministic | CublasPedantic | CublasPedantic |
| disable_tf32() | Deterministic | CublasPedantic | CublasPedantic |

There is no hidden remembered vendor preference. Positive selectors are ordered:
the last `set_fast_gemm(true)` or `set_batch_invariant(true)` wins. False fast
and disable-TF32 calls are neutral under Deterministic, preserving normal custom
setup order and never disabling the custom deterministic TF32 policy. Both
`fast(true); bi(false)` and `bi(false); fast(true)` select Fast. Both
`bi(true); fast(false)` and `fast(false); bi(true)` select Deterministic.

Legacy numeric equivalence is intentionally limited: old hybrid behavior cannot
be preserved within three modes. For example `bi(false)` on a fresh context now
selects Pedantic, so the benchmark label “cuBLAS-TF32” must use explicit Fast.
`fast(true); disable_tf32()` selects full Pedantic, including half compute.
Document these migrations; do not hide a hybrid for compatibility.

## Environment resolution

Canonical variable: `MAMBA_RS_GEMM_MODE=deterministic|cublas-fast|cublas-pedantic`.
Trim ASCII surrounding whitespace, accept exactly these lowercase spellings,
reject empty/unknown/non-Unicode values, and default absent to Deterministic.

If the canonical variable is present, reject the presence of either deprecated
mode selector `MAMBA_RS_BATCH_INVARIANT` or `MAMBA_RS_FAST_GEMM`, even if false or
apparently consistent. The error names both variables and tells the user to
remove the old selector. This is simple documented conflict handling, not
precedence dependent on parse order.

Without the canonical variable, parse legacy selectors as optional booleans
(preserving their old recognized true/false/empty spellings) and resolve once:

| Legacy selectors | Resolved mode |
|---|---|
| neither present | Deterministic |
| BI=true, FAST absent/false | Deterministic |
| BI=true, FAST=true | error: conflicting selectors |
| BI=false, FAST absent/false | CublasPedantic |
| BI absent, FAST=false | CublasPedantic |
| BI absent/false, FAST=true | CublasFast |

Presence matters: `FAST=false` alone explicitly chooses the vendor pedantic
replacement for the old hybrid. It is not equivalent to an absent variable.
The setter table and env table have different jobs: incremental setup versus a
complete declarative configuration. Document both.

`BI_F32_POLICY`, `BI_HALF_POLICY`, `BI_TENSOR_CORES`, and `BI_GEMM_FAMILY` remain
orthogonal deterministic controls, not deprecated mode selectors. With resolved
Deterministic, accept them and preserve existing validation, including stream-K
requiring the tensor-core tier and the old `fixed` family alias during migration.
Absent tensor-core permission inherits the new true default; absent half policy
resolves `TiledParityV1`, including when tensor-core permission is true.
Absent family inherits the constructor's role choice. Explicit deterministic
controls under a vendor mode are errors, including explicitly false/default
values, instead of silent no-ops. **Migration change:** the existing parser's
implicit stream-K when TC=true and half policy is absent is removed. Callers
wanting that versioned contract must set `BI_HALF_POLICY=streamk` explicitly.
This avoids turning the new default TC permission into a silent numeric-policy
change and keeps env and no-env defaults aligned on the approved tiled policy.
No project disable-TF32 env flag exists in the checked context parser; do not
invent one. Preserve `MAMBA_RS_ARCH_RUNG` validation separately.

## Minimal first implementation slice and migration checks

First land the mode enum, pure resolver/transition tables, canonical context
storage, checked reversible handle math, central vendor compute selection,
default constructors, and deprecated adapters in context/device/BLAS. No kernel
source changes. Add/update identity handling only for the actual vendor-policy
semantic change. Keep the work separate from the mechanical Inference rename.

Test all nine mode transitions, same-mode no-op, malformed/env-conflict cases,
setter sequences shown above, default customTC=true/exact-F32/tiled-half,
explicit customTC=false, explicit stream-K opt-in, failures before publication,
failed restoration,
and preservation of custom policy/family across vendor round trips. Update
vendor comparison fixtures to request their intended mode explicitly; do not
allow the new default to turn both sides into the same custom implementation.
GPU probes must inspect handle math and GemmEx compute for all three dtypes,
including F32 pedantic, and verify replay rejection for different live modes.

Then thread explicit mode through M1/M3 inference/backbone, both LM surfaces
(a field on `Mamba3LmBuild`), and both trainers using additive `*_with_mode`
constructors for positional APIs. Existing convenience entry points resolve the
environment/default consistently; explicit-mode entry points do not read it.
Test every public construction seam and its inference/training role family.

**Release completion depends on the separate direct-cuBLAS routing work** from
the audit: raw pointer NN, tied F32 and typed heads, typed no-bias calls, and M3
F32 projections. Include true F32-output handling for typed tied logits and M3
physical graph manifests. Until those land with no-vendor end-to-end evidence,
the mode slice is an integration foundation, not a completed Deterministic
no-cuBLAS release contract. Unavailable custom coverage must fail closed.
