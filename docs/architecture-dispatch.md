# Kernel selection by GPU architecture

This is how kernel selection in mamba-rs must work. It is a standing rule of
the project, and it applies to every kernel family: the deterministic GEMM
triad, the inference GEMMs, the Mamba scans and every kernel added later. The
same rule binds any project that carries these kernels (SQV-RS / Lilly takes
its GEMMs from this triad).

## The rule

1. **SM80 is the base block.** The portable tier (`mma.sync`, `cp.async`,
   `kernels/*/sm80/` and the family roots) runs on every card from compute
   capability 8.0 up. It is the floor every architecture stands on, and the
   only tier a card needs in order to run at all.

2. **Each architecture adds its own fast paths, in its own module, using its
   own hardware.**
   - SM89 (Ada): the finalist and joint kernels.
   - SM90a (Hopper: H100, H200): WGMMA, TMA and warp specialisation.
   - SM100 (datacenter Blackwell): tcgen05 and tensor memory.
   - SM120 (RTX Blackwell: RTX 5090, RTX PRO 6000): TMA with `mma.sync`.

   A new architecture gets a module the same way. An architecture module is
   bound by the architecture alone; it never needs to know which card of that
   architecture it runs on.

3. **Inside an architecture everything is automatic.** Every card of an
   architecture runs at its best with no manual step:
   - SM120: RTX 5090 (170 SMs) and RTX PRO 6000 (188 SMs);
   - SM89: RTX 6000 Ada and L40S (142 SMs) and RTX 4090 (128 SMs);
   - SM90a: H100 SXM and H200 (132 SMs) and H100 PCIe (114 SMs).

   A different SM count, memory size, bandwidth, clock, driver or box must
   never need a per-card table, a per-card tournament, a frozen per-board
   cohort or a code change to reach its architecture's fast routes.

4. **Selection is deterministic and frozen, and it reads the live device.**
   - The inputs are the architecture, the shape and operands, and the live
     device properties: SM count, opt-in shared memory per block, and L2 size.
   - The architecture's frozen rules turn those inputs into a route. There is
     no online timing-based autotuning.
   - Measurements on a card calibrate the architecture's rules: a measured
     cell is a point the rule must reproduce. A measured cell never becomes a
     key that only one board's identity matches.
   - Wave fill, residency and persistent grid sizes are computed from the
     live SM count.

5. **Bits are identical on every card of an architecture.**
   - Choices that adapt to the board are made only among routes that give the
     same bits: tile shape, stages, persistent grid, rasterisation and
     swizzle, all over a fixed per-output k-order.
   - Choices that change the reduction order (split-K partitions, split-M,
     stream-K) are decided by the shape and the architecture only, never by
     the SM count or any other property of one card.

6. **A rule ships only after it is checked on a second card of its
   architecture**, or is derived from properties that card exposes. Faster on
   the card the work happened to run on is not a speedup.

## Tournaments and qualification are per architecture

A tournament (timing the candidate routes for the shapes we serve) is run once
per architecture, on one representative card of it. Its winners are frozen as
that architecture's rules and cells:
- They are keyed by the architecture: compute capability family, module and
  toolkit.
- Their board-dependent parts are derived from the live device when a route
  is chosen.

A tournament runs again only for a new architecture or for new kernels. A new
card, a new CUDA toolkit or a new driver does not need one: the frozen
architecture rules serve it from the first launch, and the first-use bit proof
confirms that each fast route still gives the reference bits.

A second card of the same architecture never gets a tournament or a cohort of
its own. It runs the architecture's frozen rules from its first launch, and a
check on it only confirms that the rules hold there. A cell that loses there
is a defect of the rule, fixed in the rule for every card of the architecture.

A qualification identity carries two kinds of fields, and only one of them may
gate a card:
- **Integrity fields gate.** The compiled artifact, the source digest, the
  composer and numeric ABI revisions, and the toolkit that produced the
  artifact. They prove that the kernel a card runs is the kernel that was
  qualified.
- **Board fields never gate.** The SM count, the exact opt-in shared memory of
  the measuring card and the driver build. They are recorded as facts about
  where the measurement was taken, and they never decide which card may use
  the result.

## Forbidden

- A tournament, cohort or qualification run whose result only one card can
  use.

- Comparing an SM count, a multiprocessor count or any per-card property to a
  literal (`multiprocessors == 170`, `multiprocessor_count: 142`).
- Constants that stand for one card (`BLACKWELL_SMS = 170`, a dispatch tuning
  constant set to one card's SM count).
- Cohort or qualification identities whose speed tables are keyed to the
  exact SM count of one board.
- "This card only" branches, and exact shapes admitted to a fast route on one
  card only.
- Hand-written per-card route tables beside the dispatcher.

Correctness and integrity checks are a different matter: the compiled
artifact, the source digest and the numeric ABI may still be verified before a
route is trusted. They decide whether a kernel is the one that was qualified,
never which card may use it.

## Enforcement

`make ci-fast` runs the `boards` rule of the xtask gate. It refuses any new
comparison of an SM or multiprocessor count to a literal, any new struct field
set to one, and any new constant named for SM counts. Production code is
scanned; test modules are not.

The stock that predates the rule sits in `tools/board_pin_baseline.txt`, which
can only shrink: `cargo run -p xtask -- boards --update-baseline` refuses to
add a line. Each repair removes its lines from the baseline, until the file is
empty.

## Known gaps being repaired

No selection compares an SM count to a card's number any more (the
`boards` baseline is empty). What remains:
- Several measured cells and identities still pin the toolkit (an NVRTC
  version or the artifact digests); a new toolkit reaches those routes only
  through the first-use proof.
- The SM90a and SM100 TF32 cohorts are empty: Hopper and datacenter
  Blackwell run the portable ladder until a tournament on one of their cards.
- The SM120 table comes from the tournament on an RTX PRO 6000. It has not
  yet been timed against the RTX 5090's own former winners (rule 6); three
  shapes the 5090 had cells for are now served below the table.

Repaired:
- the TF32 evidence cohorts are keyed by architecture (one cohort per
  architecture, any SM count, toolkit or driver);
- the SM120 exact-f32 measured cells and generic boundary serve every SM120
  card;
- the exact-family split decisions read the architecture's partition
  reference (`architecture_partition_reference`), never the live count;
- the Ada half cohort, the SM120 scalar resource check and every inference
  selector serve every card of their architecture;
- stream-K deals its reduction over a fixed worker count (340 on SM120, four
  times the partition reference for the portable tc64 route); the resident
  grid walks the workers in order, so the partition points and the bits no
  longer depend on how many SMs the card has;
- the qualification tools run on any card of their architecture and record
  the measuring card's SM count as provenance only.
