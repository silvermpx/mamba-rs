# Architecture release wiring audit

Date: 2026-09-10

Audited source: `569367b6` (including cohort commit `02191e37`)

Scope: read-only architecture-selection and module-loading audit for
`gemm_bi_fixed` and `gemm_bi_triad`. No GPU result is extrapolated to another
board, compute capability, driver, toolkit, dtype, operation, or shape.

## Release conclusion

The source provides a non-vendor deterministic floor on every enumerated
architecture, but that is not the same claim as having the fastest qualified
route on every architecture.

- Portable half kernels are real production fallbacks for the Triad family,
  and the scalar exact-F32 module is a real production floor. The scalar and
  portable modules are mandatory loaded/anchored owners
  (`modules.rs:9126-9133`, `modules.rs:9206-9224`).
- The portable Triad module has exact target mappings for SM80, SM86, SM89,
  SM90a, SM100-family targets and SM120-family targets
  (`modules.rs:5687-5703`, `modules.rs:10482-10540`). Loading and ABI-validating
  one of those modules proves availability, not a performance win.
- TF32 cohort and measured-override admission is intentionally literal. Current
  measured overrides exist primarily for SM89 and CC12.0/170-SM identities.
  They must not be described as evidence for SM80, SM86, SM90/90a, SM100, or
  another SM120 identity.
- No accepted-winner constant was found disconnected from its launch path in
  this bounded trace. The concrete omissions are missing qualification cohorts
  on architectures listed below, plus stale documentation/comments that blur
  measured admission and heuristic/native availability.

## Actual routing by architecture

| Device family | Half, non-vendor general path | Exact F32 general path | Deterministic TF32 AUTO | Shape-specific / native override status |
|---|---|---|---|---|
| SM80 | Portable `TriadSm80` tensor-core ladder when enabled; typed scalar fallback otherwise (`launch.rs:11689-11724`) | `TriadScalar` ascending-K FMA | Fixed has its portable TF32 tile ladder (`gemm_bi_fixed.rs:2450-2528`); Triad has no matching SM80 evidence cohort and therefore returns to exact F32 (`dispatch.rs:3476-3593`) | No source-proven SM80 board/driver measured override in this audit |
| SM86 | Same portable half/scalar floor, compiled for `sm_86` | Same scalar exact floor | Same Fixed portable route; Triad TF32 is not admitted merely because its portable module loads | No source-proven SM86 measured override |
| SM89 | Portable half/scalar floor remains available | Scalar exact floor | Portable TF32 plus SM89 finalist/joint candidates only when the exact frozen identity, tuning revision, operands and cell match (`dispatch.rs:3450-3530`) | Measured SM89 half, exact-F32 and TF32 selectors are wired ahead of portable fallback; for half the NN/TN/NT call order is visible at `blas.rs:673-740`, `832-895`, `985-1049` |
| SM90 / SM90a | Portable floor exists; on CC9.0 the exact target is `sm_90a`. Triad tries the native WGMMA path before portable half | Scalar exact floor | Fixed can use its general TF32 ladder. Triad specialized TF32 has an empty evidence cohort and therefore serves exact F32, not portable TF32 AUTO (`dispatch.rs:3351-3367`, `3476-3593`) | Triad half's SM90a table is empty, but its resolver synthesizes Wg1/Wg2 by reduction depth and launches it when module/map/operand gates pass (`dispatch.rs:4097-4101`, `4258-4308`; `launch.rs:1457-1508`). This is a guarded heuristic native route, not a board-timed winner. Fixed's WGMMA rung is guarded by a first-use numeric self-check (`gemm_bi_fixed.rs:3556-3595`, `4247-4265`) |
| SM100 family | Portable floor exists. CC10.0/10.3/11.0 can bind the native tcgen module; CC10.1 has only the portable module in this selection | Scalar exact floor | Fixed retains its general TF32 ladder. Triad specialized TF32 cohort is empty, so AUTO remains exact F32 | Triad half tables are empty, yet the resolver synthesizes tile/stage/schedule and can launch the native module after exact target, map and operand validation (`dispatch.rs:4100-4209`; `launch.rs:1092-1138`). This is not measured-board admission. Fixed's tcgen rung uses the same first-use self-check as Hopper |
| SM120 (CC12.0) | Portable floor plus specialized half selection from measured entries or guarded nearby-shape interpolation | Scalar exact floor, with qualified SM120 exact-FMA routes when binding/operands permit (`dispatch.rs:3599-3613`, `4038-4094`) | Only exact frozen cohort identities/cells are admitted. Cohort `02191e37` retains 23 of the final 24 CUDA-12.8/13.0 cells and the matching CUDA-13.2 set; G10 is deliberately absent. Portable selections inside such a specialized cohort additionally require the frozen portable twin (`dispatch.rs:3476-3530`) | Half tables contain 60 tiled and 12 stream-K entries. Exact entries take priority; other shapes can use a nearest entry inside the guarded band, with underfill and stream-K restrictions (`dispatch.rs:5699-5824`). Outside that band or when launch gates fail, portable serves the request (`launch.rs:8006-8069`). Fixed also has SM120 F32/TF32/half selectors (`gemm_bi_fixed.rs:3960-4245`) |
| SM120 (CC12.1) | Portable half fallback; both specialized half tables are empty (`dispatch.rs:5683-5696`) | Scalar exact floor; a qualified generic exact-FMA route may bind | No matching measured TF32 cohort established by the audited source; unsupported identity/cell falls to exact F32 | Specialized module availability is not a measured CC12.1 performance claim |

SM90 versus SM90a needs careful release wording: runtime CC9.0 is mapped to
the exact `sm_90a` portable target (`modules.rs:10526-10539`), and the native
Triad family is `TriadSm90a`; this audit found no separately admitted generic
SM90 measured cohort.

## Concrete release gaps

1. **Triad portable TF32 availability is broader than Triad TF32 AUTO
   admission.** The module can be compiled and loaded for all listed targets,
   and function lookup is ABI-gated (`modules.rs:9446-9465`), but AUTO requires
   a matching evidence cohort. SM90a and SM100 cohorts are explicitly empty;
   SM80/SM86 have no matching cohort in the inspected selection. This is not a
   dead winner unless corresponding board-local winning evidence exists; no
   such evidence was established by this read-only audit.
2. **SM90a/SM100 half comments misdescribe live behavior.** Tables are empty
   and comments say uncovered requests decline, but resolvers create heuristic
   native routes and the BLAS call sites run them before portable half. Release
   notes must call these guarded architecture routes, not measured cells or
   fastest-qualified routes.
3. **The public 18-cell SM120 half description is stale.** Current source has
   60 tiled and 12 stream-K entries, while `README.md:49-51`, `CHANGELOG.md:10`,
   `docs/mamba1-architecture.md:117`, `docs/determinism-benchmarks.md:18`, and
   `context.rs:946` still say 18. `README.md:75` also says stream-K TN cells on
   CC12.x although CC12.1's tables are empty. These claims should be corrected
   before release without changing dispatch.
4. **“Fastest qualified typed route” is too broad as a global release claim.**
   `README.md:42-45` should be scoped to a concrete measured identity/cell.
   Generic portable fallback, Fixed's self-checked architecture rung, and
   Triad's heuristic SM90a/SM100 route are useful coverage but are not proof of
   board-local optimality.
5. **SM120 half coverage is not limited to the listed entries.** Controller
   source verification corrected the initial audit's strict-non-cell-fallback
   description: `nearest_sm120_cell` selects a same-op/dtype entry within a
   factor of eight on each geometry axis, then applies tile-underfill and
   stream-K grid guards. Exact entries are preferred. Interpolated shapes
   are not independently timed winners; diagnostics describing every such
   route as measured for that exact shape also need accurate release wording.

## Release constraints

Do not fill an empty cohort, reuse an identity, or relax a fail-closed gate to
satisfy an “all cards fastest” statement. The performance playbook requires
independent qualification for each toolkit/device/dtype/op and explicitly says
portable kernels remain fallbacks while architecture AUTO promotion remains
literal (`docs/performance-playbook.md:253-262`). Release CI alone proves no GPU
behavior (`docs/release-qualification.md:3-6`). The release must still run the
gate and contract lanes, compare acceptance evidence byte-for-byte or account
for each change, perform the controlled performance guard, and only then do the
locked publish dry-run/tag sequence (`docs/release-qualification.md:32-48`).

Accordingly, the defensible handoff is: **general deterministic non-vendor
coverage exists across the enumerated architectures; fastest-qualified claims
exist only for the exact board/driver/toolkit/shape cells named by retained
evidence.** Performance on the unmeasured architecture rows remains unknown,
not inferior and not qualified.
