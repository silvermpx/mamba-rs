# Performance playbook

How kernels in this crate are measured, changed and admitted. This page
is for contributors who change kernels; users do not need it.

## Measure first

- Build the bench arm before the optimization: an isolated launch of the
  kernel at the production shape (`benches/`, or a qualification tool when
  it needs a specific board), a few warmup iterations, then timed
  iterations under CUDA events. Predictions about GPU walls are usually
  wrong; the largest kernel of the Mamba-3 backward was invisible until it
  was timed alone.
- Keep a per-kernel ledger in ms/step terms (per-launch time times layers)
  and rank by it, not by how suspicious the code looks.
- Compile bench kernels at the production state capacity and flags.
- Print the resolved route in every reading: mode, family, tensor-core
  permission and numeric policy (`ctx.gemm_mode()`, `ctx.bi_gemm_family()`,
  `ctx.gemm_route()`; the shared `STAMP` line of the benches). A number
  without its route is not a reading; readings have been taken on the wrong
  tier because a selector was assumed rather than printed.
- Name the comparator literally. Fast TF32, Fast f32 compute and Pedantic
  f32 are different denominators; a win against one says nothing about the
  others.

## Measurement protocol

- Paired alternating windows (ABBA or BAAB) on a quiet board, medians per
  arm; a single run cannot separate a 1 % change from thermal and clock
  drift. Two variants of one kernel measured inside the same run agree to
  about 0.1 %; the same kernel measured in two separate runs drifts by
  several percent.
- Event timing, not wall clock: a host sync around every launch folds
  launch and synchronization overhead into the reading, a first-order error
  at kernel durations of tens of microseconds.
- Argument order in hand-built launch harnesses is a real defect class:
  when a reading looks odd, check the argument list against the kernel
  signature before anything else.
- `include_str!` kernels mean copying a source is not enough; cargo must
  rebuild for the new source to reach NVRTC. Check the binary's time stamp
  against the source before trusting a green gate.

## What usually costs the time

- Occupancy. Small blocks behind large shared-memory tiles are the first
  suspect: one warp per block behind an 88 KB tile ran at about 2 %
  occupancy and was 73 % of a training step. Redistribute lanes so each
  output keeps one owner running the same inner-loop order and widen the
  block; sweep the split factor, because more threads is not monotonically
  better.
- Redundant compute. Stage shared work (pair dots, decay exponentials,
  trigonometric factors) in shared memory once per chunk with the
  consumer-exact expression, so every read stays bit-identical. Give large
  shapes a fallback tier that drops the staging rather than the launch.
- Coalescing. A thread walking its own row while its warp neighbours sit a
  stride away pays a sector per access; stage the tile cooperatively and
  read the row from shared memory. Global-memory shift registers are the
  extreme case.
- Over-materialization. Tape what is small and replay what is cheap: the
  scan tape once stored T+1 states per element (12 GB at the production
  shape); keeping per-chunk boundary rows and replaying the state in the
  backward with the same helpers was bit-identical and freed the memory.
- Grid size at small batch. A serve prefill launched its convolution on 3
  blocks of a 170-SM GPU; tiling it over T was the largest single win of
  that pass. A launcher that reserves shared memory at a compile-time
  maximum quietly caps occupancy on every smaller shape.
- Deleting launches is not automatically visible time in a scan-dominated
  chain; keep such fusions when they are bit-identical, and book the win as
  what was measured.
- Registers before barriers. `ptxas -v` on the old and the new source is
  the first filter and costs seconds: a stack frame with no spills is an
  array indexed by the variable of a rolled loop, and merging two block
  scans to halve the barriers can cost sixteen live registers and make
  the kernel slower. Unroll the loop that indexes the accumulators, keep
  the scans apart, and let the hand-off slots ride on the scans' own
  barriers.
- A kernel that runs one block per SM hides nothing. A branch one lane
  walks alone (the edge lane reading a slot while thirty-one wait) cost
  the f32 fold backward a whole gain that the same code gave the bf16
  kernel at three blocks per SM; read the slot warp-uniformly and select
  per lane. When the shared-memory tile is what holds occupancy at one
  block, stage fewer lanes and read the rest from global memory into the
  registers they were going to occupy anyway.
- A column reduction with one thread per column is latency, not
  bandwidth: a few hundred threads on a 142-SM board wait on eight loads
  at a time. Stage a row tile through shared memory with a full block and
  keep the ascending chain in one owning thread; the bits stay, the
  memory pipeline widens.

## Bit discipline

- Before changing a kernel without digest coverage, add an output-hash arm:
  FNV-1a over every output buffer at the production shape on deterministic
  inputs, recorded once and compared after. Digest gates run on every GEMM
  tier and every chunk-count shape.
- Replay rules for bit exactness: reuse the forward's helpers and
  shared-memory slot classes, store boundary values verbatim instead of
  recomposing them, and reuse operands the backward already loaded.
- Known bit movers: FMA contraction (pin with `__fadd_rn(__fmul_rn(..))`),
  a changed reduction order, and shuffle masks that name non-executing
  lanes once trip counts diverge.
- Deterministic PTX is pinned at the instruction level, not at the C++
  type level: NVRTC may re-read a shared-memory fragment instead of keeping
  the register when the source leaves it the choice.
- A gate's reference can itself be wrong. Compare families with a different
  summation order against the exact reference, not against another tiled
  kernel; on a numeric failure, diagnose candidate, reference and exact
  values element by element before rejecting the candidate.
- A macro body in an NVRTC source dies silently when one line loses its
  backslash continuation; after the first non-continued line of a
  `#define`, the next line must not end with a backslash.
- The kernel's shared-memory layout and the launcher's byte count are two
  sources for one number; when one moves (a per-warp slot replacing a
  per-thread row), the other must move in the same change, and the memory
  checker on the hash suite is the test that says so.
- The digest recorders are `#[ignore]` tests: run them with `--ignored`
  and count the `HASH` lines in the log, since a run without the flag
  exits green with nothing recorded. A recorder prints one key per
  activation dtype whenever the kernel takes a different path per dtype.

## Admission

Discovery and qualification are separate. Screen candidates on one
representative board and toolkit with short paired timings and the small
correctness set (compile and resources, layout model, bit correctness,
repeat, eager and graph, bias off and on), and reject losers early. Promote
finalists together, freeze their source and artifact identities, then run
the full qualification once for the promoted domains:

1. Freeze the exact cell contract from machine-readable evidence: operation,
   dtype, shape, strides, bias, toolkit, device, dispatcher revision,
   selected symbol, launch geometry, and the exact cuBLAS compute and math
   mode of the comparator.
2. Inventory before inventing: every kernel, loader, forced route, automatic
   selector cell, graph identity and fallback. A candidate is not production
   work until the dispatcher, loader, tests and graph route select it.
3. Diagnose before changing geometry. Profile where a profiler is
   available; otherwise use physical graph, resource and SASS evidence and
   paired microbenchmarks, and label the causal inference as such.
4. Change one mechanism per experiment (stage schedule, stage count, warp
   tile, CTA tile, copy plan, raster order, epilogue) with a written
   hypothesis and a stop condition. Hopper and Blackwell recipes do not
   transfer to SM80 or SM89.
5. Prove the schedule on the host first: exhaustive models of copy
   coverage, shared ranges, fragment addresses, stage-ring lifetime, output
   ownership and tail behaviour, plus one owner per output and the unchanged
   ascending reduction order for deterministic GEMMs.
6. Fail-closed GPU gates in increasing cost order: compilation and PTXAS
   resources; runtime occupancy and exact function, grid, block, shared and
   ABI; ordinary, exceptional, tail, alignment, stride, view, red-zone,
   input-immutability, eager-repeat and graph-repeat bit checks with
   poisoned outputs and a negative no-op graph test; then paired alternating
   timing, 21 windows, extended to 101 when both p50 and p95 beat the
   incumbent. A confirmed improvement over the crate's own route is kept
   whether or not cuBLAS remains faster.
7. Qualify each domain independently. A win on CUDA 13.2 does not admit
   CUDA 13.0 or 12.8; a win on SM89 does not admit SM80 or SM120. Appending
   a source fragment changes the module's artifact identity, so retained
   routes of that module are requalified on the final build even when their
   own files are byte-identical. The same holds for any byte of any fragment
   the module composes, a reworded comment included, and the closure is
   silent: the admission is a predicate over the live identity, so the route
   falls back to the plain kernel and every functional test stays green.
   The Fixed module in particular composes the Mamba kernels, so a scan or
   norm edit closes the copy-plan cohorts. Keep a host test that holds each
   cohort's source digest to the source the tree composes, run the
   discovery stage before repopulating a cohort (it expects the plain route
   to still be the automatic one), and re-read the paired matrix for the
   selected symbol, not only the timing.
8. Preserve evidence: raw samples, source snapshots, telemetry and a
   SHA-256 manifest covering every file in the evidence directory. Never overwrite a wrong-shape or losing run; mark why it
   is not admissible. Measured losers stay until a release cleanup pass
   removes proven duplicates.

For the Triad family, start from the full matrix of NN, TN and NT by dtype,
shape, bias, toolkit and architecture and profile the worst release-weighted
cells first; an Inference NN schedule does not transfer to transposed
layouts, whose copy coalescing, shared layout, raster order and cuBLAS
kernel selection are measured separately.
