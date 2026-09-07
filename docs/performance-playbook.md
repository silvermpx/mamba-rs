# Performance playbook

Distilled from the kernel-optimization work that took the production
training step (d_model 384, 24 layers, B=8, T=1300, bf16, batch-invariant +
tensor-core tier) from 441.4 to 131.5 ms/step (M1, -70%) and the M3 step
from 636 to 179.4 ms/step (-72%). Most changes were bit-identical on all
nine digest arms; the handful that regrouped reductions landed inside one
deliberate bit-family window with re-recorded baselines. Every rule below was paid for with a measurement or a bug on
this codebase; apply them in order.

## 1. Measure first — predictions about GPU walls are usually wrong

- The pre-work analysis's top prediction (sector amplification from the
  scan tape layout) measured CAMPAIGN-NEUTRAL: L2 absorbed it. The two
  real walls (a conv sliding window living in global memory, a wrong
  env-var name silently disabling the tensor-core tier) were found only
  by isolated kernel benches and env auditing. Build the bench arm
  BEFORE the optimization: an `--ignored` test that launches the kernel
  standalone at the production shape, 3 warmup + 20 timed iterations.
- Keep a per-kernel ledger in ms/step terms (per-launch ms x layers).
  Rank by it, not by how suspicious the code looks. The biggest M3 item
  (m3_dqkv at 73% of the whole step) was invisible until timed alone.
- When a profiler is unavailable (a rented box that blocks ncu, no
  nsys), the
  isolated-bench harness is a full substitute for time attribution.
- Compile bench kernels at the PRODUCTION state cap / flags. A
  default-64 cap where production injects 16 can overstate
  register-array kernels.
- Bench the exact env the production trainer stamps. The "BI+TC"
  rows once measured plain BI for a whole night because the flag
  name was wrong (`MAMBA_RS_BI_TENSOR_CORES`, not
  `MAMBA_RS_BATCH_INVARIANT_TC`) — and a day of follow-up readings
  measured the cuBLAS lane because the tier rides TWO env flags
  (`MAMBA_RS_BATCH_INVARIANT` plus the TC flag on top) and only one
  was set. Same class hit the GEMM micro-bench twice in one day.
  The cure is structural, not discipline: benches print their
  RESOLVED tier in the reading itself (`tier=bi+tc`), so a missing
  base flag is visible in every number it produces.

## 2. Occupancy: small blocks behind big smem tiles are the #1 GPU wall

- m3_dqkv ran ONE 32-thread warp per block behind an 88 KB smem tile:
  one block per SM, ~2% occupancy, 73% of the M3 step. The cure is
  lane redistribution (t-split): keep every output element owned by
  exactly one lane running the same inner-loop order, widen
  `blockDim.y` so per-timestep loops stride over it. 11.4 -> 4.0
  ms/launch with zero bit movement.
- Sweep the split factor — it is launch geometry, bit-free by
  construction: T_SPLIT 8/16/32 gave 4.17/4.02/6.24 ms. More threads
  is not monotonically better (register pressure, scheduler).
- Check `__launch_bounds__` before widening a block: a stale
  `(32, 4)` pin turns a 256-thread launch into
  CUDA_ERROR_INVALID_VALUE.
- The forward twins (grid covers (b*chunk, head)) did NOT share the
  disease — plenty of blocks. Same kernel family, different structure:
  measure, do not pattern-match.

## 3. Redundant compute: stage shared work in smem once per chunk

- Pair dots recomputed per lane (hd-fold), decay exponentials
  recomputed per consumer (~7.4k exp2f per lane), cos/sin computed 3x
  per angle: stage them in shared memory / registers ONCE with the
  CONSUMER-EXACT expression and every read stays bit-identical.
- Give large-shape configs a fallback tier (`use_pair_mats`,
  `use_staging`): when the tile exceeds the smem cap, fall back to the
  inline forms — lose the speedup, never the launch. The launcher's
  tier maths and the kernel's slice stride MUST be one formula: a
  packed-slot stride that does not grow with the tile silently
  corrupts neighbor slots (caught only because a parity test covered
  the packed shape).

## 4. Coalescing: per-thread row I/O is a sector per access

- A thread walking its own row of 16 floats while its warp neighbors
  sit nh*ds floats away pays one 32-byte sector per access. Stage the
  [CS][ds] tiles cooperatively (thread x loads flat element x, x+CS,
  ...), then read your own row from smem: 0.61 -> 0.27 ms/launch,
  values and bits unchanged.
- Global-memory shift registers are the extreme case: the conv1d
  window lived in global memory (~7 dependent accesses per timestep);
  registerizing it alone was -28% of the whole training step.

## 5. Over-materialization: tape what is small, replay what is cheap

- The scan h tape stored T+1 states per (b,d,n): 12.28 GB at the
  production shape, written once and read twice. The slim tape keeps only per-chunk
  (run_a, run_b, h_entry) rows (3 floats per chunk) and REPLAYS h in
  the backward with the same thread-local scan, the same block scan
  helper and the same compose chain on the same inputs — bit-identical
  by construction, -14.5 ms/step, and a 4x bigger micro-batch fits.
- Replay rules for bit-exactness:
  - Reuse the same helper functions and the same smem-slot classes the
    forward used; recompute nothing in a different association.
  - Store boundary values VERBATIM instead of recomposing them — the
    run-prefix composition associates differently from the stored
    value ((comp_a*run_a)*h0 vs comp_a*(run_a*h0+run_b)).
  - Reuse the backward's already-loaded operands (delta/u/B, da
    registers) — the replay's marginal cost is mostly FLOPs the SM had
    spare.

## 6. Bit-discipline instruments — build them before the change

- If a kernel has no run-digest coverage, add an output-hash arm
  first: FNV-1a over every output buffer at the production shape on
  deterministic inputs, record the hashes, compare after. This is what
  proved the tape removals and the staging bit-clean and what makes lane
  redistribution reviewable at all.
- Digest gates run on EVERY GEMM tier and EVERY chunk-count shape
  (single-chunk + multichunk): the inter-chunk carry was uninstrumented
  for a whole release until the multichunk arms landed.
- Known bit-movers to avoid: FFMA contraction (an inlined `acc += a*b`
  contracts to one rounding where the old code had two — pin with
  `__fadd_rn(__fmul_rn(...))`); changed reduction order (per-lane
  partials combined in lane order != serial t order — resum in the
  historical order on one lane instead); `__shfl_down_sync` masks that
  name non-executing lanes once trip counts diverge (segment-local
  masks).

## 7. Process rules

- Phase-batch: write the whole change, gate once (parity + digests +
  bench), fix red as one batch, commit green. Per-edit compile loops
  waste hours.
- Argument-order bugs in hand-built launch harnesses are real: the
  isolated fwd bench arm had h_saved in the wrong slot and measured a
  kernel reading wrong buffers. When a bench number looks odd, check
  the arg list against the kernel signature first.
- Direct-launch test sites are part of every kernel-signature change
  (grep tests/ for the kernel name) — the unit test's own smem formula
  was the last place the old tile size survived.
- Keep escape hatches for one release (`MAMBA_RS_SCAN_TAPE=full`) and
  verify BOTH modes against the same digest baseline.

## 8. Lessons from the inference pass

- At B=1 the grid is the first suspect: the serve prefill launched its
  conv at 3 blocks on a 170-SM GPU. T-tiling the nosave conv (with the
  carry-in state seeding tile 0 and the last tile owning the carry-out)
  was the single biggest win of the wave.
- A launcher that reserves shared memory at a compile-time maximum
  (MAX_DSTATE) instead of the runtime dimension quietly caps occupancy
  on every shape smaller than the maximum. Address-only re-stride, the
  cheapest class of change, moved the fold backward 2 -> 3 blocks/SM
  and the serve another half millisecond.
- Deleting launches is not automatically visible time: removing two
  elementwise kernels per layer from a scan-dominated chain measured
  ~0 at the serve shape. Keep such fusions when they are bit-identical
  and simplify the chain, but book the win honestly as zero.
- A macro body in an NVRTC source dies silently when one line loses its
  backslash continuation: the kernel compiles up to that line and the
  rest becomes top-level garbage the error log attributes far away.
  Audit rule: after the first non-continued line of a #define, the next
  line must not end with a backslash.
- include_str! kernels mean scp-then-test is NOT enough - cargo must
  rebuild for the new source to reach NVRTC. Check the test binary's
  mtime against the source before trusting a green gate.
- The bit gates earn their keep in hours: the 16-cell prefill hash
  suite caught nothing all wave precisely because every edit was
  designed against it - and the one time a whole test binary went red,
  the failure pattern (every GPU test failing at context creation)
  pointed at an NVRTC syntax error, not a numeric defect.

## 9. Kernel optimization protocol: diagnose before designing

Use this protocol for Fixed inference and Triad NN/TN/NT work. Its purpose is
to prevent long sequences of plausible but causally unsupported tile changes.

**User-approved fast discovery cycle (2026-09-07).** Separate candidate search
from production qualification. This cost ordering supersedes older task briefs
that require the entire qualification matrix before every candidate timing:

- Discover on one representative installed toolkit (currently CUDA 13.2 on
  Ada). Reuse the current census, known losing experiments and existing test
  routes; do not rebuild the inventory for every edit.
- For each hypothesis, check compilation/resources and the affected layout
  model, then a small meaningful bit-correctness, repeat, eager/graph and bias
  set. Use short paired timings to reject losers early. These are discovery
  results, not full qualification or authority to change AUTO.
- Collect promising candidates, integrate a finalist batch, freeze its
  source/binary identities, and run the full numerical, physical and paired
  performance qualification for the promoted domains once. A changed shared
  CUDA module still requires retained-route qualification on that final build.
- After a fix, rerun affected checks and reuse evidence whose source, binary
  and contract dependencies are unchanged. Do not automatically restart all
  toolkits, all precisions or all 101-window runs.
- Do not hold Triad discovery until every Fixed inference cell beats cuBLAS
  Fast. Run a bounded inference experiment, record remaining gaps, then include
  the worst Triad cells in the next discovery cycle.

The detailed safety requirements below remain production-promotion gates;
they are not a mandate to run every expensive gate before a prototype can be
discarded. A quick pass never implies complete bit-exact qualification, a
cross-toolkit admission, or a cuBLAS Fast victory.

1. **Freeze the exact cell contract from machine-readable evidence.** Read the
   current production census/raw JSONL and record operation, dtype, M/K/N,
   strides, bias, alpha/beta, CUDA toolkit, CC/SM count, dispatcher revision,
   selected symbol, grid/block/shared bytes, and the exact cuBLAS compute/math
   mode. Treat prose and copied task briefs as hints only. The Ada half pass
   caught a stale `K=256` brief only by checking the authoritative B0 record,
   where `K=768`; the wrong-shape timing remains useful evidence but is not an
   admission result.
2. **Inventory before inventing.** Enumerate every existing kernel, loader
   holder, force route, AUTO selector cell, graph identity, supported CUDA
   toolkit and architecture fallback. Reuse the verified census; update only
   changed or missing domains to find unwired champions. A candidate is not production
   work until the dispatcher, loader, tests and graph route all select it.
3. **Define the comparator literally.** `CUBLAS_COMPUTE_32F`,
   `CUBLAS_COMPUTE_32F_FAST_TF32` and PEDANTIC are different denominators.
   Record algorithm, math mode, bias work and graph/eager semantics in every
   result. Never infer a FAST victory from a PEDANTIC comparison.
4. **Diagnose the bottleneck before changing geometry.** Capture an isolated
   full-shape baseline. When the mechanism is unclear, consult authoritative
   NVIDIA/PTX/CUTLASS sources and profile with Nsight Compute (SpeedOfLight,
   SchedulerStats, WarpStateStats, MemoryWorkload, Occupancy, LaunchStats and
   InstructionStats). Rank experiments from measured barrier/scoreboard/MIO,
   eligible-warp, tensor-pipe, cache/DRAM and tail-wave evidence. If profiling
   is unavailable, use physical graph/resource/SASS evidence and paired
   microbenchmarks, but label the causal inference.
5. **Change one mechanism per experiment.** Examples are stage schedule,
   stage count, warp tile, CTA tile, copy plan, raster order or epilogue—not
   several at once. Write a short hypothesis with the expected counter change
   and a stop condition. Preserve architecture limits explicitly; Hopper
   TMA/WGMMA or warp-specialized recipes do not apply to SM80/SM89.
6. **Prove the schedule on the host first.** Before CUDA, write an exhaustive
   RED/GREEN model for copy coverage, shared ranges, fragment addresses,
   stage-ring lifetime, output ownership and tail/drain behavior. For
   deterministic GEMM, also prove one CTA per output and the unchanged
   ascending MMA/reduction order. Host layout success is necessary, not a
   performance claim.
7. **Use fail-closed GPU gates in increasing cost order.** During discovery,
   apply the small targeted checks above. For production qualification, compile/PTXAS and
   reject illegal shared memory, register overflow, local memory or spills;
   check runtime occupancy and exact function/grid/block/shared/ABI; then run
   ordinary, exceptional, tail, alignment, stride, view, redzone,
   input-immutability, eager-repeat and graph-repeat bit gates. Before each
   independent replay, poison every output storage word so it differs from
   the expected bits, preserve guards, and verify the poison was uploaded.
   A constant sentinel may equal exceptional-value output; complementing the
   expected bits avoids that gap. Require a no-op graph negative test to fail,
   and decode captured argument values, not merely their count. Cover both
   bias states even when timing only the no-bias cell. For nonzero beta, the
   graph must restore the required old C before each GEMM; poison is not a
   substitute for that input. Only then run paired
   alternating ABBA/BAAB timing for 21 windows. Advance to 101 windows when
   both p50 and p95 beat the incumbent. Retain confirmed own improvements
   even while cuBLAS remains faster; vendor victory is a separate comparison,
   never an extra condition for preserving the owner's measured improvement.
8. **Qualify each domain independently.** A win on CUDA 13.2 does not admit
   CUDA 13.0/12.8, and a win on SM89 does not prove SM80/SM120. Rebuild with
   separate source/target/cache identities and rerun correctness, physical
   graph and paired timing for every promoted toolkit/device/dtype/op cell.
   A standalone NVCC win must also pass these gates after integration into
   the actual production NVRTC module. Appending a source fragment changes
   that module's artifact: requalify its retained routes even when their
   individual source files remain byte-identical.
   Portable fast kernels should remain available as fallbacks on unoptimized
   architectures; architecture-specific AUTO promotion remains literal.
9. **Preserve evidence and report every decision.** Keep raw samples,
   analyzer plus synthetic rejection tests, source snapshots, telemetry and a
   rooted SHA256 manifest. Never overwrite a wrong-shape or losing run; mark
   why it is non-admissible. After each robust victory report exact p50/p95,
   resources and promoted dispatcher cells. Keep measured losers until the
   release cleanup pass, where only proven duplicate/unreachable candidates
   are removed.
10. **Split roles without splitting truth.** One owner controls the GPU lane;
    read-only agents may audit sources/evidence or authoritative external
    references in parallel; one implementer owns an isolated candidate; an
    independent reviewer checks any production promotion. The root owner
    verifies manifests and full gates, updates the decision ledger, and makes
    small green commits with human repository identity.

For Triad, begin with a complete matrix of NN/TN/NT × dtype × shape × bias ×
toolkit × architecture and profile the worst release-weighted cells first.
Reuse the deterministic arithmetic contract and this gate order, but do not
assume a Fixed NN schedule transfers to transposed operand layouts: copy
coalescing, shared layout, raster order and cuBLAS's selected kernel must be
measured separately for each operation.
