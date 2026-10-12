# Performance playbook

How kernels in this crate are measured, changed and admitted. This page
is for contributors who change kernels. Users do not need it.

## Measure first

- Build the bench arm before the optimization: an isolated launch of the
  kernel at the production shape (`benches/`, or a qualification tool when
  it needs a specific board), a few warmup iterations, then timed
  iterations under CUDA events. Predictions about GPU walls are usually
  wrong. The largest kernel of the Mamba-3 backward was invisible until it
  was timed alone.
- Keep a per-kernel ledger in ms/step terms (per-launch time times layers)
  and rank by it, not by how suspicious the code looks.
- Compile bench kernels at the production state capacity and flags.
- Print the resolved route in every reading: mode, storage precision and
  the route (`ctx.gemm_mode()`, the model's `dtype()`, `ctx.gemm_route()`.
  The shared `STAMP` line of the benches). A number
  without its route is not a reading. Readings have been taken on the wrong
  tier because a selector was assumed rather than printed.
- Name the comparator literally. Fast TF32, Fast f32 compute and Pedantic
  f32 are different denominators. A win against one says nothing about the
  others.

## Measurement protocol

- Paired alternating windows (ABBA or BAAB) on a quiet board, medians per
  arm. A single run cannot separate a 1 % change from thermal and clock
  drift. Two variants of one kernel measured inside the same run agree to
  about 0.1 %. The same kernel measured in two separate runs drifts by
  several percent.
- Event timing, not wall clock: a host sync around every launch folds
  launch and synchronization overhead into the reading, a first-order error
  at kernel durations of tens of microseconds.
- Argument order in hand-built launch harnesses is a real defect class:
  when a reading looks odd, check the argument list against the kernel
  signature before anything else.
- `include_str!` kernels mean copying a source is not enough. cargo must
  rebuild for the new source to reach NVRTC. Check the binary's time stamp
  against the source before trusting a green gate.

## What usually costs the time

- Occupancy. Small blocks behind large shared-memory tiles are the first
  suspect: one warp per block behind an 88 KB tile ran at about 2 %
  occupancy and was 73 % of a training step. Redistribute lanes so each
  output keeps one owner running the same inner-loop order and widen the
  block. Sweep the split factor, because more threads is not monotonically
  better.
- Redundant compute. Stage shared work (pair dots, decay exponentials,
  trigonometric factors) in shared memory once per chunk with the
  consumer-exact expression, so every read stays bit-identical. Give large
  shapes a fallback tier that drops the staging rather than the launch.
- Coalescing. A thread walking its own row while its warp neighbours sit a
  stride away pays a sector per access. Stage the tile cooperatively and
  read the row from shared memory. Global-memory shift registers are the
  extreme case.
- Over-materialization. Tape what is small and replay what is cheap: the
  scan tape once stored T+1 states per element (12 GB at the production
  shape). Keeping per-chunk boundary rows and replaying the state in the
  backward with the same helpers was bit-identical and freed the memory.
- Grid size at small batch. A serve prefill launched its convolution on 3
  blocks of a 170-SM GPU. Tiling it over T was the largest single win of
  that pass. A launcher that reserves shared memory at a compile-time
  maximum quietly caps occupancy on every smaller shape.
- Deleting launches is not automatically visible time in a scan-dominated
  chain. Keep such fusions when they are bit-identical, and book the win as
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
  kernel at three blocks per SM. Read the slot warp-uniformly and select
  per lane. When the shared-memory tile is what holds occupancy at one
  block, stage fewer lanes and read the rest from global memory into the
  registers they were going to occupy anyway.
- A column reduction with one thread per column is latency, not
  bandwidth: a few hundred threads on a 142-SM board wait on eight loads
  at a time. Stage a row tile through shared memory with a full block and
  keep the ascending chain in one owning thread. The bits stay, the
  memory pipeline widens.

## Index width

- A tensor past 2^31 elements must compute correctly at every shape the
  model's `validate()` accepts, never with a silent wrap and never with a
  refusal. That does not make every index 64-bit. Unconditional `long long`
  indexing cost Mamba-1's prefill 18 % on an RTX 6000 Ada: the parallel
  scan went from about 76 to about 90 registers, a 128-thread block lost
  one resident block per SM (6 to 5), and the 768-block grid of a
  one-sequence prefill needed a second wave on 142 SMs. The 170-SM
  RTX 5090 kept its single wave and showed nothing, so an index-width
  change is measured on both boards.
- Pick per kernel, and compare `ptxas -v` against the 32-bit source first:
  1. A 64-bit base once per block, chunk or row, and 32-bit offsets inside
     the hot loop, when every offset inside is provably below 2^31 for
     every accepted shape. Write the bound next to the base. The reference
     Mamba scan, FlashAttention and CUTLASS index this way.
  2. When that still costs registers in a register-bound kernel, compile
     the body twice from one template on the index type. The `int`
     instantiation is the 32-bit code, bits and speed included. The host
     picks it whenever every index the launch computes fits under the
     `check_int_args` headroom, and the `long long` one otherwise. PyTorch
     dispatches its fast kernels the same way.
  3. Element-wise kernels whose source is pinned run in pieces under 2^31
     (`launch_cast`). Pieces are bit-identical along independent axes only,
     never along a reduced or scanned one.
  4. One-time 64-bit arithmetic per thread or block costs nothing
     measurable and stays.
- No 64-bit division or modulo per element in a hot loop. A 2-D grid or a
  row base replaces the decomposition of a flat index.
- Cast before the first multiply that can pass 2^31, check every
  intermediate product, and size the host predicate by the largest tensor
  the launch touches. The full scan tape `[B, D, N, T + 1]` passes 2^31 long
  before the activations do.
- The same change moves registers both ways. The 64-bit SISO patch made
  some chunk kernels 36 % faster at one shape and the sequential angle
  backward 25 % slower at another, so the decision is per kernel and
  measured.
- Both instantiations are tested: the wide one forced at small shapes and
  compared bit for bit with the narrow one, and one real test past 2^31
  per family (8 GiB of f32 fits a 48 GB board).

## Graph capture and other threads

- A synchronize of the device or of the context is invalid while any stream
  of that context captures a CUDA graph, and it ends that capture whichever
  thread issues it (CUDA Programming Guide, CUDA Graphs). Library code
  waits on its own stream only. `cublasDestroy` synchronizes the device, so
  cuBLAS handles live in a list the process keeps and are never destroyed
  (`src/gpu/cublas_pool.rs`). `cuCtxSynchronize` and a synchronous
  `cuMemFree` are the same hazard.
- `tests/capture_beside_teardown.rs` captures in one thread while another
  builds and drops models, and fails within seconds when anything brings a
  device-wide wait back.

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
  kernel. On a numeric failure, diagnose candidate, reference and exact
  values element by element before rejecting the candidate.
- A macro body in an NVRTC source dies silently when one line loses its
  backslash continuation. After the first non-continued line of a
  `#define`, the next line must not end with a backslash.
- The kernel's shared-memory layout and the launcher's byte count are two
  sources for one number. When one moves (a per-warp slot replacing a
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
   available. Otherwise use physical graph, resource and SASS evidence and
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
   resources. Runtime occupancy and exact function, grid, block, shared and
   ABI. Ordinary, exceptional, tail, alignment, stride, view, red-zone,
   input-immutability, eager-repeat and graph-repeat bit checks with
   poisoned outputs and a negative no-op graph test. Then paired alternating
   timing, 21 windows, extended to 101 when both p50 and p95 beat the
   incumbent. A confirmed improvement over the crate's own route is kept
   whether or not cuBLAS remains faster.
7. Qualify each domain independently. A win on CUDA 13.2 does not admit
   CUDA 13.0 or 12.8. A win on SM89 does not admit SM80 or SM120. The
   first-use bit proof carries a route's BITS to a board that holds no
   cohort for it, never its timing: a proven route is admitted, not
   measured, and no table may quote one board's microseconds for another. Appending
   a source fragment changes the module's artifact identity, so retained
   routes of that module are requalified on the final build even when their
   own files are byte-identical. The same holds for any byte of any fragment
   the module composes, a reworded comment included, and the closure is
   silent: the admission is a predicate over the live identity, so the route
   falls back to the plain kernel and every functional test stays green.
   The Inference module therefore holds only GEMM sources, the shared
   prelude and the casts its routes launch. The Mamba-1 kernels and the
   kernels the families share compile in modules of their own, which no
   cohort pins, so a scan or norm edit moves nothing. `tests/gemm_source_freeze.rs`
   pins every file the GEMM modules compose and names the one that moved.
   Keep a host test that holds each cohort's source digest to the source
   the tree composes, run the
   discovery stage before repopulating a cohort (it expects the plain route
   to still be the automatic one), and re-read the paired matrix for the
   selected symbol, not only the timing.
8. Preserve evidence: raw samples, source snapshots, telemetry and a
   SHA-256 manifest covering every file in the evidence directory. Never overwrite a wrong-shape or losing run. Mark why it
   is not admissible. Measured losers stay until a release cleanup pass
   removes proven duplicates.

For the Triad family, start from the full matrix of NN, TN and NT by dtype,
shape, bias, toolkit and architecture and profile the worst release-weighted
cells first. An Inference NN schedule does not transfer to transposed
layouts, whose copy coalescing, shared layout, raster order and cuBLAS
kernel selection are measured separately.
