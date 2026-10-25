# GEMM architecture

This page, `docs/gemm-architecture.md`, is the contract for every GEMM the
crate runs in the `Deterministic` mode: which bits a product writes, which
kernels may write them, how the dispatcher chooses among those kernels, and
what a change proves before it merges. A change that cannot meet it is
redesigned, not merged.

For GEMMs it replaces `docs/architecture-dispatch.md`, which keeps the
per-architecture rule for the other kernel families (the scans and the
shared kernels) and points here. `docs/gemm-modes.md` (the user-facing
promises) and `docs/release-qualification.md` (the release runbook) must
agree with this page; step 0 of section 8 lists what changes in them.

The code does not meet every rule yet. Where it falls short, the rule says
so under **Status** and the gap is an entry in the ledger (section 7). A
rule is never weakened to match the code.

## 0. Scope and terms

The contract covers:
- both deterministic families: Triad (`kernels/gemm_bi_triad/`: the NN
  forward, the NT input gradient, the TN weight gradient) and Inference
  (`kernels/gemm_bi_inference/`: the forward of bf16 and f16 models).
  Weight gradients always run on Triad;
- every deterministic policy: exact f32 (`ExactScalarFma`,
  `ExactFmaBatchInvariant`), strict TF32 (`RnaTf32BatchInvariant`, which
  `WeightDtype::Tf32` selects) and bf16/f16 (`HalfTriadPolicy::TiledParity`, `AllowStreamKFixedOrder`);
- the route controls (`GpuCtx::route_controls`), which count as policy:
  they fix a context's bit classes when it is created, never per call.
  Through them the Inference family also serves exact f32 forwards (FFMA
  tiles, and TMA and copy-plan tiles at measured cells on CC 12.0 and CC
  8.9) and TF32 forwards (its own `cvt.rna` tiles, and a force-only Triad
  wide tile with the half-ulp conversion). They never cut a reduction, so
  on a layer the f32 plan cuts they write other bits than Triad; they are
  bit classes of their own, and no storage precision selects them.

`AllowDeterministicTf32` mixes TF32 and exact products by measured cell and
falls to the exact family when a TF32 route does not bind, so it does not
meet goal 1; no storage precision selects it, and its documentation says
so. V0 to V4 (section 5) run per family and per policy.

**Status.** The route controls have live setters (`set_family`,
`set_tensor_cores`, `set_f32_policy`, `set_half_policy`), and tests,
qualification tools and `classifier_gemm_tier_bench` switch them between
calls on one context.

Terms:
- **Shape.** NN: `Y[m][n] = X[m][k] W[k][n]` (+ bias), reduction k. NT:
  `dX[m][k] = dY[m][n] W[k][n]^T`, reduction n (the forward's outputs). TN:
  `dW[k][n] (+)= X[m][k]^T dY[m][n]`, reduction m. The layer is (k, n); m is
  the row count. A shape is written `op m x k x n` in these letters: NN
  64 x 768 x 32000 runs 64 rows through a layer of 768 inputs and 32000
  outputs; NT 4096 x 256 x 2112 is the input gradient of a layer of 256
  inputs and 2112 outputs at 4096 rows; TN 1024 x 256 x 256 sums a
  256 x 256 weight gradient over 1024 rows.
- **Epilogue class.** Bias or none; alpha 1 or other; beta 0, 1 or other;
  the output type.
- **Cell.** One policy, op, shape, epilogue class and operand alignment
  class on one architecture.
- **Census.** The generated grid of shapes behind the qualification set:
  2384 shapes (674 forwards, 828 input gradients, 882 weight gradients),
  each a cell under the policy measured. **Qualification set:** the cells
  V3 times (5.4); a qualification cell is one of them.
- **Plan.** The reduction order a cell must produce (section 2). **Bit
  class:** the plan with its arithmetic (precision family, operand
  rounding, MMA instruction).
- **Route.** A kernel, or a short sequence of kernels, with its launch
  geometry. **Candidate:** a route that implements the plan of a cell.
  **Placement:** how a route maps the plan's pieces onto CTAs (section 3).
  **Portable route:** a route of the SM80 base tier. For TF32 and half, a
  cell's portable `mma.sync` route is its bit oracle on each compute
  capability (5.3).
- **S:** the SM count of the context a launch runs in. **o:** a kernel's
  resident CTAs per SM. **Unit:** the work of one CTA, a tile or a
  (tile, piece) pair.
- **Architecture.** A compute capability and the module family that serves
  it: the SM80 base, SM89, SM90a, SM100, SM120.
- **Arithmetic class** of a comparator: the precision of the operands the
  multiplier reads (f32, TF32, bf16 or f16) and of the accumulation. The
  rounding mode of an operand conversion, the summation order and the cut
  are not part of it, so cuBLASLt's TF32 mode is the strict TF32 family's
  arithmetic class.

## 1. Goals

1. **Bits by shape.**
   - Exact f32: a product's bits are one function of the policy, the op,
     the shape and the epilogue class, on every architecture, card,
     toolkit and driver. The CPU twin in `src/gemm_fold.rs` reproduces
     them.
   - TF32 and half: the plan, the operand rounding, the MMA instruction
     (shape, k step, accumulator chaining) and the epilogue are the same on
     every route and every architecture; only the tensor core's own
     inner-product rounding differs between compute capabilities (it does
     between an A100 and an H100: Khattak and Mikaitis, arXiv 2512.07004).
     The bits are a function of the policy, the op, the shape, the epilogue
     class and the compute capability. Two compute capabilities share them
     only where V2 has shown it on a card of each.
   - Forward and input gradient: a row's bits never depend on the rows
     around it. Their plan reads the layer, never m.
   - Weight gradient: the reduction runs over the batch, so the bits depend
     on m and on the order of the rows, and on nothing the forward
     excludes. The contract fixes the bits of one call; how a batch is
     split into calls, accumulation steps or devices, and the order of a
     cross-device reduction, change the bits by design and are outside it.
   - Nothing else chooses the bit class: not the SM count, a partition or
     the occupancy; not pointer alignment or leading dimensions; not which
     modules bound or which kernels a toolkit excluded; not a measured
     table, a cohort or a first-use verdict; not the workspace size, the
     stream, or eager against captured execution. These inputs choose only
     among routes of one class. A table entry that names a route of another
     class is a contract change, not a speed choice.
   - Roles: f32 and TF32 models serve on Triad with the bits they trained
     with. bf16 and f16 models train on Triad and serve on the Inference
     family, whose forward is another bit class; both are deterministic and
     batch-invariant. This is the one declared place where training and
     serving bits differ.
2. **Automatic per architecture.** The SM80 portable tier is the base; each
   architecture adds its fast paths in its own module (SM89 finalist and
   joint tiles; SM90a WGMMA and TMA; SM100 tcgen05; SM120 TMA with
   `mma.sync`). Inside an architecture every card runs at its best with no
   per-card table, constant, cohort or tuning file, and no rule or constant
   is keyed to one card's SM count.

   **The card rule.** A rule, constant or fitted model ships only after a
   second physical card confirms it: a card of another architecture for one
   that applies on several architectures, a second card of the same
   architecture for one that applies on one. A green-context partition
   checks how a rule depends on S, on compute-bound cells only; it never
   stands in for the second card.
3. **Deterministic.** The same inputs write the same bits on every launch,
   whatever else runs on the card, whichever CTA arrives last, and whether
   the product runs eagerly or from a captured graph. Every process on a
   board admits the same routes.
4. **No speed paid for 1 to 3.** On every qualification cell of every
   reachable architecture, the chosen route is no slower, beyond the noise
   band (5.4), than:
   - the previous release on the same card;
   - cuBLASLt in the product's arithmetic class, at its default heuristics
     and workspace: f32 compute with TF32 and emulated modes off for exact
     f32, `CUBLAS_COMPUTE_32F_FAST_TF32` for strict TF32, f32 compute for
     bf16 and f16;
   - any other implementation the maintainers time, compared where the
     stored timings are kept (5.1) and never named or quoted in tracked
     files.

   A comparator of another arithmetic class (TF32 tensor cores for an exact
   f32 product, emulated or reduced-precision modes) is reported and never
   gates. A release is judged cell by cell, never on a total (5.4). An
   architecture the release does not reach carries no speed claim.

## 2. Layer 1: the fold contract

### 2.1 One definition

`gemm_fold::plan(op, family, shape, epilogue, cc) -> FoldPlan` in
`src/gemm_fold.rs` is the only code that decides a reduction order.
- An NN or NT plan reads the layer (k, n) and never m; a TN plan reads
  (m, k, n).
- The compute capability `cc` enters only a TF32 or half plan, and only to
  fix the instruction class (2.2 item 6) and, for half, the tensor-core or
  scalar class. It never fixes a cut, and an exact plan never reads it.
- The dispatcher computes the plan first, before it reads the SM count, an
  occupancy, the bound modules, the operand alignment or any table.
  `FoldPlan` has no constructor outside `gemm_fold`: selection takes it by
  reference and can neither build nor alter one.
- Every kernel receives the plan as launch parameters: piece length in
  inputs, piece count, the stream-K deal where the plan has one, the close
  grain while one exists, the bias place. No planner and no kernel derives
  a cut from constants of its own.
- No workspace capacity decides a cut. A bound on the piece count that
  exists for a workspace is a plan constant in `gemm_fold`, and changing it
  is a contract change (2.6). A route whose workspace cannot hold the
  plan's pieces is not a candidate; the plan stays.
- A family's plan is the same function of its inputs on every
  architecture. The architecture chooses kernels, never a cut.

**Status.** Only the exact NN and NT cut lives in `gemm_fold` (`is_cut`
and `exact_piece_len`, read through `exact_chunks` and `exact_chunk`), and
`GEMM_CONTRACT_REVISION` has no reader. Decided elsewhere:
- the exact TN split, `splitm_tn_partition_over_tiles`: a 288-block target
  from `ScalarSplitPolicy` and a scratch-size condition.
  `gemm_fold::exact_tn_piece_len` and `exact_tn_output` are unused and disagree with it in the cut (TN 1024 x 256 x 256: 32 pieces of 32 rows in the twin, 64 of 16 rows
  on the GPU) and in the close (2.4);
- the TF32 TN split, `rna_backward_rule_route` and `rna_shape_cut`, which
  read the split workspace and the architecture: TN 1024 x 256 x 256 splits eight ways off CC 12.x and runs as one chain on CC 12.x. `gemm_fold::tf32_tn_piece_len` is unused and disagrees with both: it asks for sixteen pieces wherever a batch of at least 1024 rows meets at most eight 64 x 64 output tiles, and lacks the GPU's 128-column floor and its CC 12.x exception;
- in the kernels: the exact and TF32 pieces kernels derive their piece
  length from the reduction and a compiled count of four;
- the stream-K deals of the half and TF32 weight gradients;
- the warp partitions inside the matvec kernels (`kernels/gemm_bi_inference/matvec.cu`), which the Triad family's bf16 and f16 forwards run for fewer than 32 outputs or an f32 output.

Each moves into `plan`. Until a family's cuts are there, V1 cannot check
that family's routes.

### 2.2 What the plan fixes

1. **The cut:** whether the reduction is cut, the piece length and the
   piece count. Piece boundaries fall on the family's granule, so that no
   boundary splits a k step of any tile or instruction (today 32 inputs for every NN and NT cut, 16 rows for the exact TN split, 32 rows for the TF32 TN split).
2. **The chain of a piece.**
   - Exact f32: one correctly rounded `fma` per input (`__fmaf_rn`;
     `f32::mul_add` in the twin), inputs in ascending order.
   - TF32: each operand rounded once with `cvt.rna.tf32.f32` (3.10), then
     `mma.sync.m16n8k8` with f32 accumulation over ascending k steps of 8,
     the accumulator chained from one instruction to the next.
   - Half on the tensor cores: bf16 or f16 operands as stored,
     `mma.sync.m16n8k16` with f32 accumulation over ascending k steps of
     16.
   - Half on the scalar tier: the exact f32 plan over the exactly widened
     operands, then one conversion.
3. **The start of a chain.** A one-piece chain of a product with a bias and
   alpha 1 starts from the bias. Every other chain, every piece of a cut
   reduction included, starts from +0.0.
4. **The fold.** s = p0, then s = `__fadd_rn(s, p_q)` for q = 1, 2, ... in
   ascending order. No pairwise or tree fold.
5. **The epilogue (2.3), the close (2.4) and the output conversion.**
6. **For TF32 and half, the instruction class.** Every compute capability
   serves the `mma.sync` class. A route that uses wmma, another mma shape,
   wgmma or tcgen05 is another bit class until V2 on a card of that compute
   capability shows its words equal to the portable route's at every row
   count. Until then one class serves a given op, family, layer and compute
   capability at every row count; the row count never chooses between
   classes.

The plans in force:
- **Exact NN:** cut when k is at least 512, or at least 384 with n under
  128, into pieces of 128 inputs, the last one ragged
  (`FoldRule::CHUNKS_128`); otherwise one chain from the bias.
- **Exact NT:** the NN rule applied to the forward's layer: cut when n is
  at least 512, or at least 384 with k under 128.
- **Exact TN:** a split of the batch read from (m, k, n), today the
  288-block target over 128 x 128 tiles (64 x 32 tiles for 2 to 127
  outputs) on a 16-row granule; for a single output column, 32 lane chains
  over the rows l, l + 32, l + 64, ... from +0.0, added by a 16-8-4-2-1 warp
  butterfly, which the twin models (`tn_column_output`); otherwise one
  chain.
- **TF32 NN and NT:** the exact threshold, cut into four pieces of whole
  32-input groups, `32 * ceil(ceil(k / 32) / 4)` inputs each, the last
  shorter (`FoldRule::FOUR_PIECES`); the TF32 pieces kernels still compile
  the count. **TF32 TN:** eight partitions for a
  batch of at least 2048 rows and, off CC 12.x, for a batch of 1024 to 2047
  rows into at least 128 columns and at most 16 tiles of 64 x 64; one chain
  otherwise, and wherever the eight partial planes overflow the split
  scratch.
- **Half NN and NT:** the tensor-core or the scalar class, chosen by the
  op, the layer and the compute capability; a layer the f32 plan cuts takes the scalar class. A forward with fewer than 32 outputs, or with an f32 output, runs the matvec kernels instead: each output's reduction is split across eight warps into contiguous ranges of ceil(k / 8) inputs rounded up to an even count, each chained from +0.0, and the partials are added in a fixed tree, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)), before alpha and the bias. **Half TN:** one chain on the tiled kernels, or, under `HalfTriadPolicy::AllowStreamKFixedOrder`, a stream-K schedule whose deal reads the shape.
- **Inference:** no cut, across CTAs or inside one; its bf16 and f16 forwards with fewer than 32 outputs run the legacy wmma tile.

**Status.** The one-column TN route (strided pieces, a butterfly) and the matvec kernels (warp ranges off the 32-input granule, a tree) break items 1 and 4. Both are
fixed by the shape, so they are deterministic, but no other placement
writes their bits. Each is a ledger entry until step 3 of
section 8 moves its kernel to the ascending fold or adopts its tree as that
plan's fold under 2.6, with a twin.

**The cut of deep, narrow layers: 128-input pieces.** On an RTX 6000 Ada,
over the census's 779 cut cells under the batch-invariant exact policy,
with the in-CTA tiles keeping their sum in registers and reading the
piece length at launch, all cuts interleaved in one process and each cell
at its fastest same-bit route of each cut, the time against four pieces
was (geometric means; cells faster or slower by more than 3 percent):

| piece length | all | faster | slower | 1-64 rows | 65-256 | 257-1024 | over 1024 |
|---|---|---|---|---|---|---|---|
| 64 inputs | 0.967 | 325 | 197 | 0.940 | 0.967 | 1.058 | 1.014 |
| 128 inputs | 0.938 | 397 | 151 | 0.926 | 0.928 | 0.957 | 0.985 |
| 256 inputs | 0.982 | 278 | 250 | 0.984 | 0.993 | 0.958 | 0.974 |

The worst cells of the 128-input cut are input gradients (NT 4096 x 256 x
2112 1.55); SM120 was not measured with the register tiles. The earlier
measurement on an RTX 5090, before them:

Four pieces cap such a layer at
four units per output tile at every depth; a fixed piece length on the
32-input granule gives a deeper layer more pieces. On an RTX 5090, over the census's cut cells under the batch-invariant exact f32 policy (`ExactFmaBatchInvariant`), each piece length built as its own binary (the
plan is still a compiled constant) and each cell taking its fastest
same-bit route of that run, the time against four pieces was (geometric
means; cells faster or slower by more than 3 percent):

| piece length | all | faster | slower | worst | 1-64 rows | 65-256 | 257-1024 | over 1024 |
|---|---|---|---|---|---|---|---|---|
| 64 inputs | 0.998 | 293 | 291 | 1.8 | 0.970 | 0.964 | 1.096 | 1.151 |
| 128 inputs | 0.968 | 353 | 190 | 1.39 | 0.950 | 0.953 | 1.010 | 1.062 |
| 256 inputs | 0.973 | 322 | 134 | 1.44 | 0.964 | 0.976 | 0.986 | 1.004 |

Pieces of 512 inputs came to 1.013. The large-batch losses come from the
in-CTA fold kernels: the exact in-CTA pieces kernels compile in four
pieces, so under any other cut their bits differ and they leave the
candidate set, and the SM120 TMA exact kernel folds through its slab in L2
at every piece end (NT 4096 x 256 x 2112: 93 us with four pieces, 105 with
pieces of 128 inputs, 144 with pieces of 64). Separate binaries are only
indicative, since timings move between processes (5.4). The 128-input cut
is in force on the RTX 6000 Ada measurement above; the SM120 kernels follow
it, their TMA fold through L2 at every piece end being the open cost, and
the measurement on a second card is owed (section 8).

### 2.3 Epilogue

After the fold, in this order, each operation rounded once:
1. v = s.
2. If alpha is not 1: v = `__fmul_rn(alpha, v)`.
3. If the product has a bias that did not start the chain:
   v = `__fadd_rn(v, bias)`.
4. If beta is not 0: v = `__fmaf_rn(beta, old, v)`, which for beta 1
   equals `__fadd_rn(old, v)`.
5. The close (2.4).
6. One conversion to the output type: none for f32, round to nearest even
   for bf16 and f16.

Every Triad entry refuses a bias with alpha other than 1
(`validate_f32_triad_operands`, `validate_bias_preseed`), so steps 2 and 3
never both run; f32 Triad also requires beta 0 and no bias on NT, and beta
1 and no bias on TN.

The old output enters once, last. A placement that folds in the output
buffer reads the old output before piece 0 overwrites it, or folds through
a workspace (3.3).

Every policy serves alpha 1 with beta 0 or 1 by the plan of 2.2, or
refuses the call. Any other scale follows this epilogue exactly or is
refused. No policy sends a scaled or accumulating forward or input
gradient to a plan that reads the row count.

Every operation on a chain, the fold, the bias, alpha, beta, the close and
the conversion is written with an explicit rounding intrinsic
(`__fmaf_rn`, `__fadd_rn`, `__fmul_rn`) or its PTX `.rn` form. The modules
compile with `--fmad=true`, so NVVM may contract a bare `a * b + c` before
any PTX exists; `ptx_rounding` pins the bare adds and multiplies left in
the PTX against the driver's assembler but cannot undo a contraction
already made. A host source scan over every GEMM kernel tree, the Triad
counterpart of `tests/inference_epilogue_pins.rs`, rejects a bare `*`, `+`
or `+=` on these values. No GEMM module flushes subnormals: no `.ftz`, no
`--use_fast_math`, no `-ftz=true` (`tests/arch_compile_gates.rs`). Zero
padding and out-of-bounds fill are +0.0 in both operands: the tensor map's
zero fill, never its NaN fill. Together these make the bits independent of
the toolkit and the driver; V2 runs on every supported toolkit to show it.

**Status.**
- The default exact policy (`ExactScalarFma`) sends a scaled or
  accumulating NN or NT to the legacy plan of its shape
  (`batch_invariant_exact_plan` returns no layer plan for it), and that
  plan reads the row count: below 32 rows it adds the k-slab partials of
  eight warps in a tree. `ExactFmaBatchInvariant` refuses such products.
- The Triad epilogues contain bare `alpha * x`, `+= alpha * x` and `+= beta * c`: the exact scalar tiles (`scalar.cu`, `scalar_nn_m64n64.cu`), the SM120 exact kernels (`sm120/exact.cu`) and the half tensor-core tiles (`sm80/mma.cu`, `sm80/half_tn.cu`, `sm80/half_small.cu`, `sm80/half_relay.cu`, `sm80/streamk.cu`). With alpha 1 and beta 0 or 1 every
  possible contraction is exact; with any other scale the bits depend on
  the toolkit.
- TN accepts any alpha, and for alpha other than 1 its routes disagree
  with this order and with each other: the TF32 TN tiles write
  `__fmaf_rn(alpha, acc, old)`, one rounding; the exact direct TN tiles, the one-column route and the half TN tiles (`sm80/mma.cu`, `sm80/half_tn.cu`, `sm80/half_relay.cu`, `sm80/streamk.cu`) write a bare `old += alpha * x`; `splitm_reduce`
  writes `__fadd_rn(old, __fmul_rn(alpha, sum))`, the order above.

### 2.4 The written zero

The close is `__fadd_rn(v, +0.0f)`, the last arithmetic before the
conversion on every route, and the twin closes the same way
(`ZeroSign::Positive`). An exactly zero output is then +0.0 on every route,
and no tile's k depth, zero padding, relay seam or piece restart can move a
bit. The change moves only words that are exactly zero, once, under a
contract revision.

Until it lands, `ZeroSign::Grain` is part of the plan, and a tile's k depth
is not free:
- a zero keeps the sign its chains leave, except where padding adds +0.0:
  after a one-piece chain whose reduction is not a multiple of 16, and
  after the last of several pieces when that piece is not a multiple of 32
  long;
- each route declares the grain it closes at, V1 checks the declaration,
  and a route that pads at another grain is not a candidate for the shape
  (today the SM120 layer route declines a cut whose reduction is 16 past a
  multiple of 32, and the native slim NT chain serves only where its
  reduction is not);
- no seam or restart adds +0.0 before a chain ends.

**Status.** Exact f32 bits are equal across architectures except for
exactly zero outputs of those shapes, which the SM120 boards (RTX 5090 and
RTX PRO 6000) pad differently from the other architectures. The twin
closes an accumulated weight gradient to +0.0, which `splitm_reduce` does
not. No GPU test produces an exactly zero output from a nonempty reduction: the twin test and the cell tool draw uniform random operands, and the empty-axis test covers only empty reductions.

### 2.5 What may vary

Free, because none of it enters the plan: the tile's rows and columns, the
stages, the grid and its rasterization, the CTAs per SM, which CTA or warp
holds which piece, relay seams (3.4), the SM count, the stream, where the
workspace lives.

Fixed by the plan: the cut and its granule, the start of every chain, the
chain's arithmetic including operand rounding and the instruction with its
k step, the fold order, the epilogue, the close, the conversion.

A CTA, or a warp group inside one, that holds several pieces keeps each
piece's chain separate until the running sum of every lower piece has been
added, then adds its own pieces one at a time in ascending order; a
sliced-k tile folds its warps' pieces in piece order. Adding pieces 2 and 3
first would compute (p0 + p1) + (p2 + p3), not ((p0 + p1) + p2) + p3.

### 2.6 Changing the plan

A change of the plan moves bits. It is allowed only when all of these
hold:
1. The new plan reads only the inputs of 2.1.
2. The CPU twin moves with it and `GEMM_CONTRACT_REVISION` is bumped. The
   revision and the active fold rules enter `gemm_dispatch_policy_digest`
   and every recorded route identity, so a cut that moves without a kernel
   byte moving still changes the identity.
3. The acceptance capture is re-recorded in the same commit on a board of
   every architecture whose bits move, and the changelog names the products
   that moved.
4. It is qualified like a release: on every qualification cell of every
   reachable architecture, the fastest same-bit route under the old plan
   and under the new one are timed (V3). The new plan is adopted only if no
   cell slows beyond the noise band, or if the old plan broke goal 1 or 3
   and the new plan is the fastest of the shape-only plans measured (the
   `FoldRule` variants and the fixed piece lengths under test).
5. Plan constants are lengths and counts read from the shape, chosen by
   the measurement of item 4 on every reachable architecture at two SM
   counts each, under the card rule (goal 2). A fixed piece length is
   preferred to a target count. A constant chosen to fill one card, such as
   a CTA target or a width "too narrow to fill the card", is not a shape
   rule even though it reads only the shape. The constants of today's plans
   that came from one card or from a total (the 288-block TN target; the narrow threshold of 384 inputs under 128 outputs; the four pieces themselves, chosen from a total on one RTX 5090; the TF32 weight gradient's eight-way split from 2048 rows, or from 1024 rows into at most 16 tiles of 64 x 64, both set to fill the card, the second measured on an RTX 6000 Ada; and the bounds of the half weight gradient's stream-K schedule in `Sm80TcPolicy`, measured on an RTX 6000 Ada and applied on every architecture but CC 12.x) are decided again under
   item 4.

## 3. Layer 2: placements

A placement maps the plan's pieces onto CTAs. Every placement writes the
plan's bits and is chosen by speed alone; the selector charges each its
real costs (4.3). A placement is built for a tile family where a measured
gap needs it, not for every family on every architecture. A placement or
kernel that is the fastest measured same-bit route on any qualification
cell is not removed before its replacement is measured at least as fast
there.

### 3.1 In-CTA fold

One CTA walks all pieces of its output tile. A load of the running sum
never waits on a store to an address the compiler cannot tell apart from
it. The running sum is kept, in this order of preference:
1. in registers: a second accumulator set, `kept = acc` at the first piece
   end and `kept = __fadd_rn(kept, acc)` at each later one, `acc = 0` after
   each; the last piece meets `kept` before the epilogue;
2. in shared memory, each element owned by one thread, or with barriers
   between its writers and readers;
3. where neither fits without a spill or the loss of a resident CTA: in a thread-private slab at offsets fixed at compile time, or in the output tile at its runtime stride, written at each piece end and read back with all of an atom row's stored sums loaded before any of that row's stores (batched loads).

The serialized form, each load next to the store before it under a runtime
stride, costs a memory round trip per element at every piece end: on an RTX
6000 Ada it made the exact 128 x 128 pieces tile 1.1 to 1.8 times slower
than the plain tile on the same shapes, whose main loops matched
instruction for instruction. Batched loads overlap those round trips but
still leave the chip: over the census's cut cells on an RTX 5090, the wide
TF32 pieces tiles, which fold that way, ran up to 1.7 times slower than the
fastest same-bit route.

Register fit, measured on an RTX 5090 under CUDA 13.2: a second set fits
the tiles with at most eight m16n8 accumulator atoms per warp, which are
the portable 64 x 64 and 16 x N `mma.sync` tiles and the multistage 64 x 64
TF32 tile; the portable 64 x 64 pieces tiles then take 162 to 174 registers
by target, under a gate and contract cap of 176. A second set spills 184 to
224 bytes of local memory in the multistage tiles with 64 accumulators per
thread (128 x 64 and 64 x 128) and in the portable 128 x 64 tile. A
shared-memory sum for a 128 x 64 tile takes 32 KB, which does not fit beside
the multistage tiles' 72 to 75 KB operand ring in the 99 KB a block may
take on CC 8.9 and CC 12.x.

A register fold is not equivalent to the batched fold by inspection. In
the wide 128 x 96 and 96 x 96 tiles and in the 128 x 128 tile, a register
sum wrote other bits than the batched fold in the batch-invariance test (a
768 x 768 layer, 1 row against 8192 rows), for a cause not yet found; those
tiles stay on the batched fold until it is known. A fold change counts only
after the batch-invariance test and the every-route test (5.3) pass on a
card.

**Status.**
- TF32: the tiles of at most eight atoms keep the sum in registers; every
  other tile folds through its output tile with batched loads. The change
  passed the TF32 contract test (87 of 87) and the strict TF32
  batch-invariance test (6 of 6) on an RTX 5090, but it landed as its own
  kernel change, with the source
  freeze table and the TF32 RNA owner digest updated and no identity mint
  or repin. The class-2 requalification it owes (6.2), the SM89 finalist
  identity composed from `sm80/mma.cu` included, is a blocking ledger
  entry. The multistage 128 x 64 pieces tiles then held 64 accumulators a
  thread and spilled with a second set; they now spread the tile over
  eight warps of 32 x 32, so each thread keeps 32 accumulators and the
  pieces' sum in registers. Run on their own on an RTX 5090, they took the
  input gradient NT 4096 x 256 x 2112 from 60.7 to 54.5 us, with outputs
  equal bit for bit to the four-warp tiles; the dispatcher's tests and the census of the
  cut-layer rule with them have not run on a card yet, a blocking ledger
  entry. Host overhead on small cells was the other gap (4.6).
- Exact: the 64 x 64 and 128 x 128 pieces tiles keep the sum in registers.
  The sum starts at -0.0, the additive identity of IEEE addition, so every
  piece end is `kept = __fadd_rn(kept, acc)` with no first-piece select, and
  the last piece meets it the same way. On SM89 under CUDA 12.8, 13.0, 13.2
  and 13.4 the 64 x 64 tile takes 143 to 152 registers at three CTAs per SM
  and the 128 x 128 tile 197 to 217 at one CTA per SM, neither with a spill;
  the 128 x 128 tile that folded through its output at two CTAs per SM
  spilled under three of those four toolkits and was removed. Rows of both
  tiles equal their CPU fold on an RTX 6000 Ada (the exact batch-invariance
  test at 1 to 4096 rows). A cut input gradient whose weights fit the
  transpose scratch runs as the forward of the transposed weight on the
  128 x 128 register tile where that tile's grid covers the card: over the
  census's 414 cut input gradients on an RTX 6000 Ada it ran in 0.62 to
  0.65 of the chunk partials' time there, while the 64 x 64 tile over the
  transposed weight lost to them below four waves of its tiles and tied
  beyond. The native slim pieces tile, which folded through its output
  tile, was the fastest in none of those cells and is removed, and with it
  the last fold through an output tile in the exact family. The pieces
  tiles take the piece length at launch. Their time against the former batched fold on
  large batches is not measured yet. The SM120 exact in-CTA fold keeps a
  thread-private slab with independent 16-byte `__stcg` and `__ldcg`
  accesses, which goes through L2 at every piece end.

### 3.2 Sliced pieces

The warps of one CTA each walk one piece of the same output tile, and the
pieces are folded in piece order in shared memory at the end. This serves
layers whose pieces are too short to deal and whose output tiles are too
few to fill the card, such as a thin batch over a narrow cut layer. Not
built yet.

### 3.3 Dealt pieces, folded in order

Each (tile, piece) unit runs on its own CTA; a tile's pieces are folded in
piece order and the epilogue runs once, at the end. There is no second
launch. Two forms:
- **Ordered in place.** Piece 0 stores its chain; piece p waits for its
  turn, adds its chain to the running sum and releases p + 1; the last
  piece adds its chain and runs the epilogue. The running sum may live in
  the output tile only when beta is 0 and the output is f32; otherwise it
  lives in an f32 slot of the launch's workspace, and the CTA that runs the
  epilogue reads the old output once.
- **Last arriver.** Every piece stores its chain to its slab, then counts
  in; the CTA that completes the count reloads every slab and folds them in
  ascending piece order, so which CTA arrives last never moves a bit. A
  fixed owner that holds the last piece may keep its own chain in
  registers and add it last. Nothing waits. The TF32 split members are last
  arrivers whose slabs are whole planes.

The workspace is one slot per tile in flight for the ordered form, unless
the output serves, and P - 1 or P slabs per tile in flight for the last
arriver, inside the split scratch; the selector charges it. What these
forms remove is the second launch and its pass, not output-sized storage.

### 3.4 Stream-K relay

A persistent grid walks (tile, k-step) units. At a seam inside a piece the
earlier CTA stores its accumulators bit for bit (f32, no conversion,
nothing added), and the later CTA loads them into the same fragment layout
and continues the same instruction sequence. Seams fall on k-step
boundaries and never pad or close; only such seams may follow the SM
count. A piece end goes through 3.3. The half relay (`sm80/half_relay.cu`)
is a relay.

A stream-K schedule that finishes a tile by adding the partial sums of its
k ranges is a cut, not a relay. Its seams are part of the plan, computed by
`gemm_fold` from the shape alone (a fixed deal of units to workers), and
the twin models them; seams dealt by the SM count are forbidden. The
portable tc64 and the SM120 stream-K weight gradients are cuts of this
kind; when their seams were dealt by the CTA count, an RTX 5090 and an RTX
PRO 6000 wrote different bits.

### 3.5 Planes and a reduction pass

Each (tile, piece) unit writes its chain to a plane of the split workspace,
and a second kernel folds the planes in piece order and runs the epilogue:
the exact chunk partials with `splitk_reduce`, the exact split-M weight
gradient with `splitm_reduce`. It is a regular candidate, charged its plane
writes and reads and the second launch. On an RTX 5090 it beats the 64 x 64 in-CTA tile for a thin batch over a wide cut layer: from 1 to 32
rows, the chunk partials ran the census's 32000-output layers (768 to 2048
inputs) in 0.54 to 0.91 of the 64 x 64 in-CTA tile's time, while at 64 rows
the tile ran 1.6 times faster. An ordered fold (3.3, 3.4) replaces it on a
cell only once measured at least as fast there on every reachable
architecture. Planes that do not fit the split scratch make the route no
candidate; they never change the cut.

### 3.6 Forward progress

A CTA may wait on another CTA of its grid only if one of these holds:
1. **Cooperative launch** (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`, which graph
   kernel nodes also take), with the grid sized from the kernel's occupancy
   and the SM resource of the context the launch runs in, so that the
   driver makes the whole grid resident or refuses the launch. Whether the
   driver's residency check honours a green-context partition is not yet
   verified; the partition runs of V2 check it.
2. **Tickets.** A CTA takes its unit from an atomic counter when it starts,
   in an order where every unit's predecessors in the fold hold lower
   tickets (piece-major order does this), and waits only on lower tickets,
   which running CTAs already hold. This is the argument of decoupled
   look-back (Merrill and Garland).

A last-arriver fold never waits and needs neither. Refused in review: a
plainly launched grid whose CTAs wait on statically indexed CTAs, which
relies on blocks being dispatched in `blockIdx` order (CUDA requires that
blocks can run "in any order, in parallel or in series"), and a persistent
grid sized from the device's SM count that relies on all of its CTAs being
resident. Green contexts and MPS clients carry no forward-progress
guarantee, so the reduced-SM and concurrent runs of V2 are regression tests
of these constructions, never their proof.

**Status.** Three shipped kernels wait under a plain launch: the SM120
exact dealt fold (its owner is the last split by `blockIdx.x` and spins on
the lower units of its tile), the portable tc64 stream-K (it waits on lower
workers and is correct only while the whole grid is resident) and the half
relay (it waits on lower CTAs of a two-CTA-per-SM grid). Only the SM120
stream-K grid barrier is launched cooperatively.

### 3.7 Memory ordering

- A publisher writes its partial or its accumulators, all its threads pass
  a barrier, and one thread releases at gpu scope (`st.release.gpu`, or
  `__threadfence()` and then the flag store).
- A consumer acquires the flag at gpu scope in one thread, all its threads
  pass a barrier, and it reads with loads that cannot hit a stale L1 line
  (`ld.global.cg`, `__ldcg`, or acquire loads); never `ld.global.nc`,
  `__ldg`, or a `const __restrict__` pointer the compiler may send down the
  non-coherent path.
- Memory written through one proxy and read through the other (generic
  accesses on one side, TMA or another `cp.async.bulk` operation on the
  other) is separated by `fence.proxy.async` on the side that changes
  proxy, besides the release and the acquire. A TMA store is waited for to
  completion before the release (`cp.async.bulk.wait_group`, not its
  `.read` form, which waits only for the source reads).
- A running sum in shared memory is owned element by element by one
  thread, or barriers separate its writers and readers.

A violation shows as bits that change from run to run, not as a crash.

### 3.8 Workspace, streams and graphs

Every workspace word a fold uses (planes, slabs, flags, tickets, counters)
belongs to one stream:
- the context that owns it issues all of its products on its one stream
  (today one scratch and one counter buffer per module set, and one context
  per stream);
- a graph captured from the context launches only on that stream or
  ordered after it, and two executable graphs, or two instantiations of one
  graph, that hold the same workspace never run at the same time (CUDA
  keeps the launches of one executable graph from overlapping, not those of
  two);
- a kernel launched with programmatic dependent launch waits for its
  predecessor (`griddepcontrol.wait`) before it touches a workspace the
  predecessor may still use;
- flag, ticket and counter words are reset by a memset ordered before the
  kernel on the same stream and captured with it, or by the consumer once
  no CTA reads them again, never by the last CTA to start; after a failed
  or aborted launch they are zeroed before the next use;
- the workspace is sized before a capture, as the staging buffers already
  are;
- a cell takes the same route eagerly and in a capture (4.2), and the
  recorded route carries `GEMM_CONTRACT_REVISION`.

cuBLAS gives up its determinism with several streams for the same reason.

### 3.9 Integer atomics

Integer atomics are allowed only as tickets, arrival counters or barriers,
whose value decides which CTA folds or when, never what is summed.
Floating-point atomics and reductions stay forbidden in every product.
`tests/determinism_source_gates.rs` lists every site by file and line with
its reason, and its scan matches every spelling: `atomicAdd`, `atomicSub`,
`atomicInc`, `atomicDec`, `atomicAnd`, `atomicOr`, `atomicXor`,
`atomicCAS`, `atomicExch`, `atomicMin`, `atomicMax`, `cuda::atomic_ref`,
`atom.*`, `red.*`.

**Status.** The scan misses `atomicInc` (`sm80/mma.cu`,
`sm80/tn_splitk.cu`, `sm80/tf32_rna.cu`, `sm120/tma.cu`) and `atomicOr`
(`kernels/loss_scaler.cu`), and the TF32 PTX scan skips the split-K
symbols.

### 3.10 Tensor-core placements and TF32 rounding

- Pieces combine with `__fadd_rn` on f32 values, never by letting an MMA
  accumulate one piece into another piece's accumulator. A piece restart
  feeds the instruction a +0.0 accumulator; a zero scale-d on wgmma or
  tcgen05 is used only where V2 shows it equal.
- The strict TF32 family rounds each operand once with `cvt.rna.tf32.f32`:
  in registers, after a tensor-map load of the raw 32-bit words, or in a
  pre-rounding pass whose output the MMA reads unchanged. Each of these
  makes another family: unrounded f32 into the tensor core, the tensor
  map's tfloat32 conversion, the half-ulp integer shortcut (a subnormal
  rounds toward zero), `cvt.rn`, `cvt.rz`, `.ftz`, `.satfinite`.
- A wgmma or tcgen05 tile joins a strict family only when its operands are
  rounded as the family requires before the instruction reads them (for
  TF32, a `cvt.rna` pass into shared memory or a pre-rounded copy) and V2
  shows its words equal to the portable route's on that compute capability
  at every row count.

**Status.** The SM90a and SM100 TF32 kernels declare the tensor map's
tfloat32 conversion, so they are not members of the strict TF32 family.
`mma.sync` and tcgen05 have been reported bit-equal only for fp16 and bf16
on one Blackwell part, not for TF32, and wgmma has not been compared
(arXiv 2609.11356).

### 3.11 What each architecture has

- **SM80, SM86, SM87:** SIMT exact tiles and `mma.sync` TF32 and half
  tiles; in-CTA folds, planes, and the TF32 split members' last arriver.
  Shared memory per block is 163 KB on CC 8.0 and 99 KB on CC 8.6, so a
  shared-memory running sum costs residency differently on each.
- **SM89:** the same, plus the multistage TF32 tiles, which stage 16-byte
  aligned operands, and the joint and finalist TF32 tiles, which have no
  pieces twin, so a strict cut layer cannot use them.
- **SM90a:** wgmma kernels for half and TF32, under 3.10; register-held
  running sums through `setmaxnreg` where a placement needs them.
- **SM100:** tcgen05 kernels for half and TF32, with the same rounding gap;
  a two-buffer tensor-memory fold is not yet shown.
- **SM120:** TMA with `mma.sync`, and the multistage TF32 tiles. The exact
  family has a dealt fold and an in-CTA slab fold. The TF32 pieces fold has
  no TMA tile, so a cut input gradient runs the multistage or portable
  pieces tiles; on uncut input gradients the TMA 64 x 64 tile ran in 0.90
  to 0.91 of the portable tile's time on an RTX 5090 and an RTX PRO 6000.

## 4. Layer 3: selection

### 4.1 Order

1. Compute the plan (2.1).
2. **Enumerate** the candidates: every tile and placement the bound modules
   of this architecture implement for this plan and this operand class
   (alignment, leading dimensions, bias, alpha, beta, output type). The
   enumeration is one function; the forcing interface and the
   qualification tools call the same function, so what V3 times is exactly
   what the selector can return.
3. **Admit** them (4.2).
4. **Choose:** the argmin of the cost model (4.3) where the family's
   constants are measured and validated on this compute capability, the
   family's current measured rule otherwise (4.4). Ties break in a fixed
   order.

The selector also reports, per cell, the row counts and the S values at
which its choice changes; V2 and V3 test both sides of each.

Selection is a pure function of the plan, the operand class, S, the
compute capability, the bound modules with each kernel's occupancy
(queried at bind time and recorded with every timing) and the stored
constants. Nothing timed at run time enters it, so V0 replays it on the
CPU.

S is the SM count of the context the launch runs in:
`cuCtxGetDevResource(ctx, CU_DEV_RESOURCE_TYPE_SM)` and its `smCount` (CUDA
12.4 and later; the device attribute where the driver lacks the call). In
a primary context that is the device's count, in a green context the
partition's. Every grid sized for residency (persistent, stream-K,
cooperative) uses this S. A wrong S costs speed; it never changes bits
and, under 3.6, never stalls a launch.

**Status.** The dispatcher reads `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
(`src/gpu/device.rs`), so inside a green context every decision and every
residency-sized grid sees the whole card. There is no single enumeration:
family-specific interfaces force routes by name (`resolve_tf32_forced`,
`resolve_sm120_forced` and their siblings), the portable exact layer plans
have none, and the cell tool reaches those through a patched copy of the
dispatcher.

### 4.2 Admission

A route enters the candidate set of a cell only by static proof, all of:
1. its declared fold signature (cut, granule, chain start, close, operand
   rounding, instruction, epilogue) equals the plan (V1);
2. its compiled module identity is one that V2 ran, or it is an SM80 base
   route, whose bits follow from its source on any toolkit once its
   epilogue meets 2.3;
3. for TF32 and half, a route outside the `mma.sync` class has shown words
   equal to the portable route's on a card of this compute capability
   (V2).

On a toolkit V2 has not run, only the SM80 base routes serve; a toolkit
joins the measured set by running V2 on it. On a compute capability V2 has
not run, TF32 and half serve only the `mma.sync` class, whose routes are
equal by construction. The context says which, once. cuBLAS is never a
candidate: it guarantees its bits only per SM count and stream.

A comparison at first use, on live operands, may decline a candidate as a
safety net; it never admits one. Every admission is therefore a function
of route, cell, board and toolkit: every process on a board admits the
same routes, and an eager call and a capture take the same route.

**Status.** `proof.rs` admits a Triad candidate for the rest of the process
when its words equal those of the route it is checked against on the first
eager call's operands, which need not reveal an order difference, and
serves that route inside a capture. On a toolkit outside the measured
set, Triad declines its frozen identities; the Inference family instead
follows the tables of the nearest older measured toolkit, or of the oldest
for an older toolkit (`serving_toolkit`), with no proof at all. Until the
Triad epilogues meet 2.3, an SM80 base route with a scale other than 1
writes bits that depend on the toolkit.

Resources: a kernel whose driver-assembled code exceeds its recorded
register gate or local-memory pin is excluded at load, symbol by symbol
(`tf32_symbol_admission`, `tf32_register_cap`). The selector then takes
another candidate of the same plan and class, never another plan, class or
precision family. When no candidate of the class binds, the call returns an
error that names the missing module. Every decline (identity mismatch,
resource refusal, missing measured cell, alignment) is counted per route
and reported by a qualification query, and a V2 or V3 run fails if a
qualification cell ran on a declined route. Gates and pins are the measured
maximum over every supported toolkit and the oldest supported driver,
recorded with the module identity: compute_120 and sm_90a modules are
assembled by the driver at load, and the driver's assembler decides their
registers and spills. Gates leave no headroom for a newer driver: a symbol
its assembler pushes past a gate declines, which costs speed only, and V2
and V3 on that driver fail until the records are measured again.

**Status.** Inputs that choose a bit class today: `AllowDeterministicTf32`
declines an excluded TF32 symbol to the exact family; in the SM120 half and
TF32 weight gradients, operands off a 16-byte boundary or a declining
toolkit take the portable stream-K schedule, which cuts the sum
differently; the SM120 half weight gradients choose between the stream-K
fold and the tiled chain by measured cells and a neighbour band whose bound
a new winner moves; the route controls' live setters (section 0).

### 4.3 The cost model

For a candidate r at the live S:

```
U  = output tiles x (pieces, for dealt pieces and planes; 1 otherwise)
o  = resident CTAs per SM of r's kernel, from its attributes at bind time
N  = ceil(U / S)          units on the busiest SM
R  = ceil(N / o)          rounds on that SM
j  = min(N, o)            CTAs sharing it
Tc = R * (a_r + K_u * c_r(j) + F_r)
Tm = bytes_r / (e_r(j) * BW)
T  = L * launches_r + max(Tc, Tm) + X_r
```

- `K_u`: k steps per unit, in the kernel's own k step.
- `c_r(j)`: SM cycles per k step per CTA with j CTAs on the SM. `a_r`: the
  per-CTA prologue and epilogue. `F_r`: the fold each unit pays, (P - 1) *
  f_r for an in-CTA fold, the owner's slab reads for a dealt one. `X_r`:
  the reduction pass of planes. `L`: the GPU-side gap between back-to-back
  kernels on a busy stream.
- `bytes_r`: r's own traffic, including the re-reads of each operand per
  tile row and column and the plane writes and reads. `BW`, the L2 size and
  the clocks are read from the device at bind time and never stored per
  architecture, inside which they differ widely. `e_r(j)`: the fraction of
  BW that r reaches with j CTAs per SM.
- Short units are charged through `a` and `f`; no k-step count filters
  them out.
- S enters only through N and R. It picks how much runs at once and which
  same-bit route is fastest, never the plan.

What the terms are for: a binary rule such as "the grid covers the card" is
not a cost model, and one sent a thin batch over a 32000-output cut layer to
a 64 x 64 tile that was almost all padding (3.5). Weighting by the busiest
SM's work alone picked faster tiles on an RTX 5090 (0.92 and 0.85 of the
time) and slowed 20 and 61 shapes on an RTX PRO 6000, up to 1.42 times,
because a CTA's rate depends on j. On memory-bound cells every route reads
the weight once, yet measured routes differ by 1.10 to 1.84 times, so
`bytes / BW` alone ties them.

Constants (`c(j)` for every j up to the kernel's o, `a`, `f`, `X`, `e(j)`,
`L`) are kept per compute capability, major and minor, and per tile family:
CC 8.0 and 8.6 run the same kernels at different FP32 rates and
shared-memory sizes. They are measured on one card of the compute
capability with the even-load cell tool (5.5): grids of j * S CTAs within
one round over a sweep of `K_u` fit `c(j)` and `a`; one to P pieces at an
equal reduction fit `f`; the reduction kernel alone fits `X`; the route's
own kernel on memory-bound cells with the L2 flushed fits `e(j)`.
Constants are recorded with the module identity they were measured on;
when that identity changes they are stale, and the family keeps its
previous rule until they are measured again.

A fitted model ships for a tile family only after validation:
- on a second physical card of the compute capability (the card rule),
  with green-context partitions on the far side of each decision boundary
  checking its S dependence on compute-bound cells, and memory-bound cells
  checked on a physical card of another bandwidth;
- with every census cell held out of the fit once;
- passing V4 (5.6).

Its choice displaces a family's measured rule only where it predicts a
gain larger than the 95th percentile of its relative ranking error on the
validation cells.

### 4.4 Before the constants exist

Until a family's constants are measured and validated on a compute
capability, the family keeps its current measured rules there. Each is
listed in the ledger with the cards and cells it was measured on (a rule
whose provenance is not recorded counts as measured on one card), and V0
checks it against the stored timings. No new threshold on the SM count,
the grid or the row count is added meanwhile, and there is no interim
guard. A measured misroute is fixed only by:
1. adding a missing same-bit candidate to the options a rule chooses
   among, or
2. editing a rule with V3 timings of the old and new routes on the cells it
   changes, under the card rule, on every architecture it applies to.

A rule measured on one card is a blocking ledger entry until the card rule of goal 2 is met: a card of another architecture confirms a rule that applies on several architectures, a second card of the same architecture one that applies on one, and a green-context partition only checks its S dependence on compute-bound cells.

The measured rules and tables of today:
- the strict TF32 cut-layer rule (`rna_layer_pieces_selection`). It weighs
  work = rows x ceil(columns / 64) against S: a forward takes the
  multistage 64 x 64 pieces tile once work is at least 16 S (a quarter of a
  wave of its tiles); a forward of at most 16 rows over a layer with ceil(columns / 32) at least 4 S takes the portable 16 x 32 pieces tile first;
  an input gradient takes the multistage 128 x 64 pieces tile once work is
  at least 64 S and the portable 64 x 64 pieces tile once it is at least
  32 S (half a wave of each); everything else, and every forward whose beta is not 0, takes the
  four-way split members. The rule serves where the multistage tiles stage
  (16-byte aligned operands, the module on CC 8.9 or 12.x); elsewhere the
  previous rule serves: the pieces twin of the tile the one-chain table
  picks, and the four-way split members under 256 rows or under half a
  wave of that tile's CTAs. It was measured at one SM count of
  one card: over the 779 cut cells of the census on an RTX 5090 (170 SMs),
  with every same-bit route of the cut timed on every cell, it came within
  0.13 percent of the fastest route per cell in geometric mean (1.11 at the
  worst cell) and took 0.83 of the previous rule's time. The previous rule
  split every batch under 256 rows and folded larger ones in the pieces
  twin of the tile the one-chain table picks; it ran 6 percent over the
  fastest route in geometric mean and up to 3.1 times (NN 64 x 768 x 32000:
  160 us against 51 us). It applies today where the multistage tiles stage and blocks the release until an SM89 card confirms it under the card rule; a second SM120 SM count checks its S dependence on compute-bound cells;
- the previous strict TF32 cut-layer rule, which serves where the
  multistage tiles do not stage; its provenance is not recorded;
- `EXACT_PIECES_M64_MIN_ROWS` (the evidence of 3.5) and the input gradient's
  64 x 64 chain tile over the transposed weight from a third of the card
  (over the 828 input gradients of the census it ran in 0.40 to 0.97 of the
  16 x 16 tile's time from 64 of its tiles up and in 1.0 to 2.3 times it
  from 48 down, the same bits on every cell), both measured on one RTX 5090;
- the exact NN plan's grid bounds: the 128 x 128 tile from four waves on
  CC 12.x and three quarters of a wave elsewhere (`quarter_waves`, from an
  RTX 5090 and an RTX 6000 Ada), the 64 x 64 pieces tile from a full wave,
  the 64 x 64 chain tile from half a wave, the slim chunk partial from
  three quarters of a wave, and the 16 x 16 partial where the 32 x 64
  partial would leave an SM idle (`exact_small_partial_underfills`, from an
  RTX 6000 Ada);
- the exact NT plan's: the slim tile once its grid covers the card, the
  forward's slim chunk partial over the transposed weight for a cut layer
  above 64 rows once its grid covers three quarters of a wave and the
  weight fits the transpose scratch (one RTX 5090: NT 1024 x 384 x 1536
  from 69 to 48 us, the same bits as the native partials), and
  `exact_small_partial_underfills`;
- the SM120 exact measured cells (`Sm120FmaMeasuredCell`) and
  `sm120_fma_exclusions`;
- the SM89 exact weight-gradient cohorts, measured on an RTX 6000 Ada, which choose the SM89 kernels by cell (`sm89_exact_f32_plan_for_route`, `sm89_exact_f32_d128_plan_for_route`) and the direct split-M partial;
- the SM120 half measured cells and their neighbour band;
- the TF32 cohorts, their measured cells and the portable neighbourhood
  band; the SM89 TF32 forward table; the SM120 TF32 gradient table; the
  split-member picks of `rna_backward_rule_route` (half a wave and four
  waves of 64 x 64 CTAs) and its NT 128 x 128 multistage tile from two
  thirds of the card (an RTX 6000 Ada);
- `rna_pre_tn_wide_pays`, which reproduces the picks of an RTX 6000 Ada;
- the half tile picks (`tc_pick_*`), whose SM89 preferences were measured on an RTX 6000 Ada alone and apply on CC 8.9 only;
- `ScalarSplitPolicy`'s thin and slim tile bounds: cuts of the legacy plans
  that only scaled or accumulating products still reach, which leave with
  that path;
- the Inference family's measured cells and cohorts, among them the SM120
  half picker's wave-pair keys and the exact f32 points measured at 170 SMs
  on CC 12.0, which are keyed to one card's SM count;
- the SM90a and SM100 automatic tables, which hold no cells.

**Status.** Several rules above apply today beyond the architecture they were measured on: `EXACT_PIECES_M64_MIN_ROWS`, the transposed 64 x 64 input-gradient tile and the TF32 cut-layer rule (one RTX 5090) and `exact_small_partial_underfills` (one RTX 6000 Ada) on every architecture; the three-quarter-wave bound of `quarter_waves` (one RTX 6000 Ada) on every architecture but CC 12.x; the split-member picks of `rna_backward_rule_route` for a deep weight gradient (no card recorded) on every architecture but CC 8.9 and 12.x.

Until a table is retired, its decision and the cost model's are both
logged and V0 compares them. A table is retired in the release where V0
shows the cost model within bound on the cells the table was measured on
and V3 confirms it on a second physical card of the architecture. No new
table, cohort, exclusion list or per-card cell is added.

### 4.5 A complete candidate ladder

The cost model chooses among what exists; only the comparators show what
is missing. Each op (NN, NT, TN), precision family and architecture offers
chain tiles in three row classes: at most 32 rows, 64 rows, and at least
128 rows. One frozen configuration for every batch size is not a ladder:
published batch-invariant matmuls lost about 20 percent (Thinking Machines, "Defeating Nondeterminism in LLM Inference") to 63 percent (LLM-42, arXiv 2601.17768) that way. Each tile
stages its operands with 16-byte copies where the alignment allows, in its
own layout or through a transposed copy of the weight, and each placement
of section 3 is offered on at least one tile of every class. A missing
member is a ledger entry that lists the cells it costs. (The exact input
gradient had two chain tiles, 128 x 64 when its grid covered the card and
16 x 16 otherwise, both reading the weight across with 4-byte copies,
until the transposed 64 x 64 tile of 4.4 joined them.)

### 4.6 Host cost

- The route decision is cached by plan inputs, operand class and S,
  separately from the pointer-keyed prepared launch; a new pointer never
  re-runs selection.
- A cached call enqueues within 10 percent of the host time of issuing the
  same launches directly through `cuLaunchKernel`, measured in the same
  process while the GPU is kept busy, with no allocation and no lock on the
  path.
- An uncached call (a new shape, operand class or S) finishes enumeration,
  admission, selection and preparation within 20 microseconds.
- V3 reports host overhead separately on small cells. A cell that runs the same kernels as the previous release but is slower in wall time beyond the noise band is
  a host regression and blocks the release. (On an RTX 5090 the
  pre-rounded TF32 weight gradient looked both of its kernels up by name on
  every call, scanning the spec tables and hashing the symbol: its small
  cells ran 3 to 7 percent slower than the same kernels launched directly
  until the prepared launch held the kernels. Every prepared route holds
  its kernels now; no route resolves a kernel by name on a cached call.)

## 5. Layer 4: proof

| gate | when | where | proves |
|---|---|---|---|
| V0 | every commit that touches selection, a plan or a kernel | CPU, where the stored timings are kept, outside the repository | the selector's choices against stored timings |
| V1 | every commit | CPU, CI host lane | the plan reaches every route unchanged |
| V2 | every board run (subset), before a release (full) | GPU, gate and contract lanes | every route writes the plan's bits |
| V3 | with every selector or kernel change; per board and toolkit before a release | GPU, the maintainers' cell tool | speed against the candidates, the previous release and the comparators |
| V4 | when constants are fitted | GPU, qualification lane | the cost model against the hardware |

### 5.1 V0: stored timings against the selector

- The stored timings are the latest V3 timings of every enumerated
  candidate on every qualification cell, per architecture, board, SM count
  (partitions included) and toolkit. They are kept with the cell tool,
  outside the repository (5.5), and V0 runs there. A row holds the cell,
  the candidate's name, the median, the run-to-run spread, the run count,
  the output hash, the kernel's occupancy and its module identity. A row
  whose module identity no longer matches is stale, and its cell counts as
  unmeasured.
- A host program runs the crate's selector on every stored cell with the
  stored S, compute capability and occupancies, and reads the ledger
  (section 7). It fails when:
  1. the chosen route was measured slower than the best stored same-hash
     candidate beyond the cell's bound (5.4);
  2. the chosen route has no stored timing on the cell;
  3. two candidates of one cell stored different output hashes;
  4. a ledger entry marked as blocking is open at a release commit.
- It prints every cell whose route changed since the previous commit. A
  change to a rule, a constant or a kernel source lands only with V3
  timings of the old and new routes on those cells, for every architecture
  it affects. The verdict, not the data, goes into the commit or pull
  request.
- Unit tests assert bits and plans, never which route a given card picks.
  A test that pins a route per SM count from one card's measurement becomes
  V0 data.

### 5.2 V1: the plan reaches every route

The structure of 2.1 is what keeps S from the bits: the plan exists before
S is read, selection cannot build one, and kernels take its
bounds as parameters. V1 checks that structure: every route the selector
can return carries a launch plan equal to the plan, compared field by field
(piece bounds, split count, tail chain, close grain, bias place, epilogue
order, instruction class), never by route name. Its sweep:
- **Crossed:** a grid of shapes the test generates around tile, wave,
  piece and padding boundaries, covering every reduction residue mod 32;
  for each, every S from 1 to 512 and every row count at which the
  selector's choice changes (one on each side) and 1, on the default
  operand class.
- **Sampled:** every other axis (alignment and leading-dimension class,
  alpha, beta, bias and output class, module, toolkit, table and admission
  state, eager and capture) varied one at a time over every shape of the
  grid at S = 1, 512 and a fixed-seed sample of S values, with every pair
  of axis values covered at least once.
- **Budget:** ten minutes of one CI core. A sweep that outgrows it moves an
  axis from crossed to sampled, never out of the sweep.

V1 runs in the host lane: the planner compiles without the `cuda` feature,
or CI runs the CPU-only tests of the `cuda` build; either way the test
binary needs no GPU and no driver. A source gate in the same lane rejects
cut arithmetic outside `gemm_fold`: piece lengths, split counts or deals
computed in the dispatcher, the launch planners or the kernels.

V1b runs in the toolkit lane on every build of a fold kernel: in the PTX
the module loader composes, each fold routine loads its stored running
sums in batches of at least one atom row, every batch before any store of
that row, and each loaded kernel reports registers and local memory within
its records.

**Status.** `gpu` sits behind the `cuda` feature and CI only type-checks
it, so the existing sweeps (`exact_batch_invariant_plan_folds_by_the_layer_alone`
and its siblings) run on GPU boxes only.

### 5.3 V2: GPU bits

A subset runs in the gate lane on every board run (`exact_batch_invariant`,
`rna_tf32_batch_invariant`, `half_batch_invariant`), the full set in the
contract lane before a release (`gemm_bi_invariance_matrix`), on every
supported toolkit (CUDA 12.8, 13.0 and 13.2) and the oldest supported
driver.
- **Forcing.** A qualification-only interface behind the `qualification`
  feature forces any enumerated candidate of a cell by name, as
  `resolve_tf32_forced` does for TF32 today. No patched dispatcher.
- **Every route.** Every admissible candidate of a cell is compared word
  for word with the automatic route.
- **Oracles.** Exact f32: the automatic route against the CPU twin on
  sampled rows (the first and last tile of rows, the ragged tail and 32
  random rows; every row of a small output). TF32 and half: against the
  cell's portable route on that compute capability, and the release's
  acceptance capture against the last accepted one (`acceptance_diff`),
  the one oracle independent of the current kernels; a moved cell is an
  intentional contract change re-recorded in the same commit, or a red.
- **Operands** from generators keyed by the shape, with fixed seeds: random
  values of mixed sign and exponent; rows and columns that cancel to
  exactly +0.0 or -0.0; all +0.0 and all -0.0 operands; a bias of +0.0 and
  of -0.0; subnormal operands and products; plus and minus infinity and NaN
  (a NaN must be NaN on both sides; its sign and payload are not compared,
  as PTX leaves them unspecified); every reduction residue mod 32, 0 and 16
  included; lengths on piece boundaries; misaligned base pointers and odd
  leading dimensions; alpha other than 1 and beta other than 0 and 1, with
  and without a bias and accumulation.
- **Rows.** The same input row, placed first, last, inside a ragged last
  tile and alone, in batches of 1, 7, 64, 1000 and 4096 rows and at each
  row count where the selector's choice changes at the live S (with its
  neighbours), writes identical words. An exact weight gradient is checked
  against the CPU twin of its m's plan, a TF32 or half one against the
  portable route, at every S tested.
- **Determinism.** Each route runs at least twice on an idle card, once
  beside a concurrent kernel from a second context, and once replayed from
  a captured graph; every run writes the same words.
- **Reduced SM counts.** Green-context partitions
  (`cuDevSmResourceSplitByCount`, CUDA 12.4 and later), sized by the
  resource's minimum partition and co-scheduling alignment (multiples of 2
  SMs on CC 8.x, of 8 on CC 9.0 and later), run with nothing in the primary
  context, the cell tool's ballast included. Green contexts on GeForce
  boards rest on user reports rather than documentation, so each run
  asserts that the S the selector used equals the partition's count.
  Flag-based placements run under a watchdog at the smallest partition and
  beside a kernel that holds SMs.

### 5.4 V3: speed

V3 runs once per board and toolkit, with the tool of 5.5.
- **Cells.** The qualification set: the census; decode and
  language-model-head cells; and every GEMM shape each bench under
  `benches/` issues, taken from a route trace of the bench. Each cell names
  its policy, op, shape, epilogue class and alignment class. Census axes
  include, per op, reductions and widths that are odd, not a multiple of 4,
  and 16 past a multiple of 32, and weight-gradient row counts up to the
  largest training batch any bench uses and at least 8192. A shape a bench
  issues that the census lacks fails the census check. Model benches are
  judged per GEMM through the route trace, never by step totals.
- **What runs.** Every candidate the selector enumerates for the cell on
  this architecture, forced by name; the automatic route; the comparators
  of goal 4 where available.
- **Noise.** A cell is slower than another route or implementation when
  its median over at least three interleaved runs exceeds the other's by
  more than the larger of 2 percent and three times the cell's run-to-run
  spread, and a re-run confirms it. Every variant of a cell runs
  interleaved in one process: one chain cell timed with the same route in
  three separate runs measured 32.3, 32.9 and 36.5 us. The best candidate
  of a cell is re-timed head to head with the chosen route before a ratio
  is reported.
- **Chosen against the best same-bit candidate,** after that re-time: at
  most 1.05 on every model-bench cell and at most 1.15 on every other cell
  (the cell's bound), and a geometric mean of at most 1.02 in every regime.
  A regime is an op, a precision family and a row bucket (1 to 32, 33 to
  255, 256 to 2047, 2048 and up). Statistics cover only cells with at least
  two timed candidates; a cell with one is reported as unmeasured, never as
  1.00. (On the census as first run, the geometric mean was 1.0017 and the
  95th percentile 1.0000 while one cell sat at 1.84: cells with a single
  timed candidate diluted every statistic.)
- **Against the previous release,** on every reachable architecture: a
  slower cell blocks the release unless its plan changed under a contract
  revision that restores goal 1 or 3 and the new plan is the fastest
  shape-only plan measured there.
- **Against the comparators:** every cell where a comparator is faster
  beyond the noise band has a ledger entry with the gap, the cause, and
  either the fix and its release, or a showing that the comparator's route
  on that cell uses a cut goal 1 or 3 forbids (a split of an input
  gradient's reduction by the row count, say) and that the chosen route is
  the fastest shape-only candidate measured there. A release blocks on
  such a cell without an entry, and on an entry whose gap grew. Entries
  for an implementation other than cuBLASLt are kept with the stored
  timings, outside the repository.
- **Cards and partitions.** The speed criteria are judged on physical
  cards. Partitions check rounds and occupancy at other SM counts, on
  compute-bound cells only: a partition keeps the whole card's L2, DRAM
  bandwidth and power budget. A partition never stands in for a card with
  more SMs than its host, nor for the second card of the card rule. The
  release notes list the SM counts each architecture was measured at; no
  speed claim is made outside them, from one card, or from cells outside
  the qualification set.
- **Unreached architectures.** An architecture whose routes or constants
  changed since its last V2 and V3 run ships the SM80 base routes of the
  current plan, and the release notes say so.

**Status.** The census as run has every layer dimension (k and n) a multiple of 4, no input gradient whose reduction is 16 past a multiple of 32, and no weight gradient above 1536 rows.

### 5.5 The even-load cell tool

V3, the fits of 4.3 and V0's data come from one tool. The maintainers keep
it, its cell lists and the stored timings outside the repository, like the
kernel stands of `docs/release-qualification.md`; this section fixes what
the tool does, not where it lives. It runs as one process; uploads
operands once; takes outputs through an asynchronous snapshot ring; ramps a
ballast from idle to full load; runs cells heaviest first, with the
variants of a cell interleaved; drops, after a short probe, variants that
cannot come within the bound; records CUDA-event and host time; and logs
instant power and clocks every 25 ms. A cell whose operands exceed half the
card's L2 is also timed with the L2 flushed between repetitions, and the
cold time decides: on a language-model head the warm chunk route read a
98 MB weight at about 2.5 TB/s, above the RTX 5090's DRAM bandwidth.

On an RTX 5090 it ran the 2384 census cells with 5 variants each in 56 s,
with no load gaps and no hardware slowdown event. Instant power above the
board limit still occurs in heavy steady cells and at transitions from
light cells at high clock, so on a card with a weak power feed its safety
is shown on that card first, with a short run and the power log. The tool
never changes clocks or power limits.

### 5.6 V4: the model against the hardware

V4 is a qualification instrument, not a release gate: performance counters
need administrator rights that rented boxes rarely grant. Under the
`qualification` feature each kernel can be built to record, per CTA, its SM
id and its start and end time; the instrumented build has its own module
identity and never serves a product. From those records the tool derives,
on any board, the waves per SM, the CTAs sharing an SM and the per-SM busy
time. Where counters are open, Nsight Compute (2025.3 or later inside green
contexts) adds achieved occupancy and stall reasons.

A fitted model passes V4 when, on every cell with at least two timed
candidates, it ranks the measured best first or within the cell's bound,
and on at least 90 percent of cells it predicts the chosen route's time
within 15 percent.

## 6. Changing a kernel

### 6.1 What pins a kernel

- `tests/gemm_source_freeze.rs` (host lane): the FNV-1a digest and length
  of every file under the GEMM source trees and their prelude.
- Owner digests the loader checks before it binds: the TF32 RNA module
  refuses to bind when the SHA-256 of `sm80/tf32_rna.cu` differs from
  `tf32_rna_source::SOURCE_SHA256`, the SM89 exact f32, SM89 exact D128 and SM89 TF32 joint owners check theirs, and the SM89 half owners refuse to compose when theirs differ: `sm80/half_tn.cu` with `_typed_prelude.cuh`, `contract.cuh`, `common.cuh`, `epilogue.cuh` and `mma16.cuh`, and `sm80/half_small.cu`, `sm80/half_d128.cu` and `sm80/half_relay.cu`.
- Frozen compile identities (source, compile key, invocation, artifact,
  header manifest, library) per module, target, capacity and toolkit. A
  cohort whose identity stops matching declines silently. The SM89
  finalist kernel is composed from the portable source text
  (`sm80/mma.cu`), so an edit there moves its identity and its
  source-pinned cohort.
- Per-symbol register gates and local-memory pins; the runtime refuses a
  kernel beyond them.
- The acceptance capture (`MAMBA_RS_ACCEPTANCE_TSV`, `acceptance_diff`).
- The fitted constants, recorded with the module identity.

### 6.2 Change classes

Every change names its class, and the class fixes the proof it carries.
1. **Dispatcher only** (no kernel byte and no plan moves): V0 with V3
   timings of the old and new routes on every cell whose route changed;
   V1; V2 on one board of each affected architecture.
2. **Kernel change that keeps every bit:** the source freeze table; the
   owner digests; `identity_mint` on CUDA 12.8, 13.0 and 13.2 for the old
   and the new tree, then `tools/identity_repin.py --apply`; gates and pins
   set to the measured maximum over the toolkits and the oldest supported
   driver, each with its reason; a clean `acceptance_diff`; the
   batch-invariance and every-route tests on a card of every architecture
   whose module changed, then the full V2 for the touched families; V3 for
   them; the affected constants and timings marked stale. The class is
   claimed only after those tests pass on the card (3.1).
3. **Change that moves bits:** class 2 without the repin (the new
   identities are minted and recorded as new cohorts), plus 2.6: the
   revision, the CPU twin, the acceptance capture re-recorded in the same
   commit on a board of every architecture whose bits move, the changelog
   entry and the contract-change qualification.

A release batches its kernel edits so that it pays for one
requalification. A kernel change merged without its class's proof is a
blocking ledger entry until the proof is paid.

## 7. The ledger

`qual/ledger.toml` lists every place where the code does not yet meet this
page. An entry names its kind, the code, the cells it costs or touches, the
cards it was measured on, and whether it blocks the release; V0 reads it.
Like this page, it names only census shapes (a bench's cells by the
bench's name) and only results measured on our own routes, our previous
releases or cuBLASLt. Kinds:
- a cut decided outside `gemm_fold`;
- an input other than the shape that chooses a bit class, the route
  controls' live setters included;
- a wait without the guarantee of 3.6;
- a serialized fold, a register fold whose bits are not yet understood, or
  a tree or strided fold outside 2.2 item 4;
- a measured rule, table, cohort or threshold with its provenance; a rule
  measured on one card, or without recorded provenance, blocks the release;
- a kernel change whose requalification is owed (6.2);
- a missing ladder member, with the cells it costs;
- a cell where a comparator is faster, with gap, cause and fix or showing;
- an unlisted integer atomic.

An entry leaves with the code it names. A rule added to this page that the
code does not meet arrives with its entries.

## 8. Order of work for this release

The order follows requalification cost: a dispatcher-only step costs
nothing in the identity process, a kernel change costs a mint on three
toolkits and a capture per architecture, and GPU time on rented cards is
scarce, so V0 and V1 carry as much of the proof as they can on the CPU.

0. **Measurement base and documents.**
   - Land the ledger with the entries this page names, and V0 where the
     stored timings are kept, outside the repository. The RTX 5090 timings
     already taken are its first data, with every TF32 pieces-tile timing
     from before the register fold marked stale.
   - Make `docs/architecture-dispatch.md` point here for GEMMs and update
     the files that cite it. Correct `docs/gemm-modes.md`: exact f32 bits
     are equal on every architecture; TF32 and half share one plan and
     differ only by the tensor-core generation; no board-dependent choice
     selects between kernels of different bits, except where the ledger
     says the code still does. In `docs/release-qualification.md`: step 7
     becomes the cell-by-cell comparison of 5.4, which is never skipped;
     step 2 takes the rule for unreached architectures (5.4); GEMM speed is
     judged by V3 per GEMM, not by bench totals; the note on routes
     admitted by the first-use proof leaves with that admission (4.2).
1. **Dispatcher only.**
   - Read S from the context and size residency grids from it.
   - Give the plan its type: computed before anything but the compute
     capability is read from the device, constructed only by `gemm_fold`;
     V1 in CI.
   - One enumeration function and one forcing interface for every family
     (4.1).
   - Admission by static proof only: the first-use comparison declines and
     never admits. Routes that rely on it today serve again once V2 covers
     their identity.
   - Route controls set once, when a context is created; tests and
     instruments that compare families or policies create a context per
     setting.
   - Planes stay a candidate on every architecture.
   - Confirm the four rules measured on one RTX 5090 (the chunk partials
     for a thin batch over a wide cut layer, the input gradient's
     transposed 64 x 64 chain tile, the strict TF32 cut-layer rule, the cut
     input gradient's slim chunk partial over the transposed weight) on an SM89 card, the second card the card rule asks of a rule that applies on several architectures, and check their S dependence at a second SM120 SM count on compute-bound cells. A rule the SM89 card does not confirm is withdrawn before the release, and the rule it replaced serves again.
2. **One kernel batch, one requalification.**
   - The plan as launch parameters in every kernel, TN included: piece
     bounds, stream-K deals, close grain. The exact in-CTA pieces kernels
     take the piece length at launch; the TF32 pieces kernels still compile
     a count of four.
   - Running sums on chip in those kernels: in registers where a second set
     fits, in shared memory where residency allows, batched output-tile
     loads only where neither does. For TF32 the 128 x 64 multistage tiles
     already keep it in registers on eight warps, and the 64 x 128 tile is
     measured with the same options; the wide 128 x 96,
     96 x 96 and 128 x 128 tiles stay on batched loads until their
     register-fold difference is explained. For exact f32 the 64 x 64 and
     128 x 128 tiles keep it in registers (3.1), and the SM120 TMA in-CTA fold keeps
     its slab on chip where it fits instead of sending it through L2 at
     every piece end.
   - The input gradient's cut layers onto the NN in-CTA pieces tiles
     through the transposed weight, as its chain layers already reach the
     64 x 64 tile and its cut layers the forward's slim chunk partial: done
     for exact f32 (3.1); the TF32 input gradient keeps its native pieces
     tiles.
   - The layer-plan kernels take the whole epilogue of 2.3, so that a
     scaled or accumulating exact forward or input gradient can leave the
     legacy plan.
   - Forward progress for the SM120 exact dealt fold, the portable tc64
     stream-K and the half relay: tickets, a cooperative launch or a last
     arriver.
   - A TMA pieces tile for TF32 on SM120; an ordered fold inside the exact
     chunk partials, kept beside planes until it is measured at least as
     fast.
   - Explicit rounding intrinsics in every Triad epilogue, with the source
     scan; the atomics scan over every spelling; the +0.0 close, selected
     through the plan.

   Then the mint on the three toolkits, the records of 6.1, and V2 on every
   reachable architecture, which also pays the requalification the TF32
   register fold owes. Every rule whose kernels changed is timed again on
   V3.
3. **Decide and freeze the bits** on that binary, as dispatcher-only
   experiments now that the plan is a launch parameter, every candidate of
   a cell interleaved in one process:
   - the cut of deep, narrow layers (2.2): four pieces against fixed piece
     lengths of 64, 128, 256 and 512 inputs, from 1 to 4096 rows, on an SM89
     and an SM120 card at two SM counts each;
   - the weight-gradient cuts, exact and TF32, moved into `gemm_fold` with
     one rule on every architecture and no workspace condition, the SM120
     and portable stream-K schedules implementing the same plan;
   - the exact weight gradients of `gemm_bi_tc_bench` at 2048 rows, up to
     1.45 times their 0.8.1 time on an RTX 5090, and the exact cut layers of
     `classifier_gemm_tier_bench`, 6 to 21 percent over 0.8.1 on the same
     card: each traced to the cut or to the placement and fixed where the
     cause lies;
   - the scaled and accumulating exact NN and NT products, moved from the
     legacy plan to the layer plan;
   - the two fixed trees of 2.2: moved to the ascending fold, or adopted as
     their plans' folds with twins;
   - the +0.0 close.

   One contract revision under 2.6 carries all of them, with the acceptance
   captures and the changelog entry.
4. **Cost model.** Fit it per compute capability with the V4 instrument. It
   replaces measured rules family by family in this release only where V4
   passes and V3 confirms it on a second physical card of the compute
   capability; elsewhere the measured rules ship with their provenance.
5. **Release qualification.** V0 to V3 on every reachable architecture, the
   comparator ledger complete, the previous release compared cell by cell;
   unreached architectures ship the SM80 base routes, and the release notes
   say so.

## 9. What not to do

Every prohibition of this page lives in its section; this index is for
review. Goal 2 and 4.4: a rule or constant keyed to one card's SM count or
shipped without the card rule, a "grid covers the card" threshold or any
new threshold on the SM count, the grid or the row count without V3
timings under the card rule, an interim guard, a new measured table,
cohort, exclusion list or per-card cell. Section 3 and 3.1: removing the
fastest measured same-bit route before its replacement is measured, a load
of the running sum that waits on a store made at the same piece end, a
fold change treated as bit-preserving before the tests pass on a card.
2.1: a cut decided by a workspace capacity, an alignment, a module's
availability or the architecture. 2.2: instruction classes mixed across
the row counts of one layer. 2.3: bare arithmetic, flushed subnormals, a
NaN fill. 3.4 and 4.2: seams sized from the SM count, cuBLAS as a
candidate, a decline that changes the plan, class or precision family, an
admission by live operands. 3.6: a wait without forward progress. 4.5: one
frozen configuration for every batch size. 5.1: tests that pin a card's
route. Goal 4, 5.4 and 5.5: speed claims from totals, one card, cells
outside the qualification set or warm L2. 5.5 and 7: in tracked files, the
cell tool, its cell lists or its timings, or the shapes or results of a
comparison with any implementation other than cuBLASLt and our releases.
