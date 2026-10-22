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

5. **Bits never depend on how many SMs a card has.**
   - Choices that adapt to the board are made only among routes that give the
     same bits: tile shape, stages, persistent grid, rasterisation and
     swizzle, all over a fixed per-output k-order. The live SM count decides
     only these, and how many pieces of a cut reduction run at once.
   - Choices that change the reduction order (split-K partitions, split-M,
     stream-K workers) are decided by the shape alone: a constant counts the
     shape's own tiles, CTAs or slabs, never an SM count, live or per
     architecture. The scalar FMA family therefore writes the same bits on
     every card of every architecture.
   - Tensor-core products are identical within one tensor-core generation:
     A100 sums four products into a 29-bit accumulator, H100 and B200 eight
     into 31 bits (Khattak and Mikaitis, arXiv 2512.07004), so bits differ
     between generations by the hardware, never by the SM count.
   - The constants come from measurements on the card at hand. A count that
     no card's waves divide costs a stream-K launch part of its last wave on
     some card; that is the price of bits that do not move with the card.

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

## Toolkits and drivers

- **The driver assembles the machine code, and the bits do not depend on
  it.** NVRTC compiles the kernels to PTX and the driver assembles the PTX
  as one whole program, with the assembler its build carries. Assembling
  through the toolkit's nvJitLink instead was measured on an RTX 5090 and
  rejected: the linker assembles relocatable code, and 113 of the 128 SM120
  GEMM kernels took more registers (up to 40 percent more). Every route of
  the SM120 table still gave the same bits from both assemblers.
- **GEMM modules round every operation as written.** Before a GEMM module
  (the triad and the inference family) reaches the driver, each `add`,
  `sub` and `mul` on a floating-point type without a rounding mode gets the
  round-to-nearest mode it computes anyway, so no driver can fuse it into an
  FMA (`ptx_rounding`). It moved no bit of the SM120 TF32 table.
- **The Mamba scans and the shared kernels are not pinned yet.** They write
  plain arithmetic, about thirty thousand such instructions in the Mamba-3
  module, and two of their kernel variants (the staged and the
  register-tiled dQKV stage, the released forms of the angle and dQK theta
  kernels) agree only while the assembler fuses that arithmetic the same
  way in both. Pinning them moved those bits by one or two ulps on an RTX
  6000 Ada. Bits that no driver can move there need their arithmetic spelled
  out in the sources, kernel by kernel, with those tests as the guard.
- **Measured toolkits keep their own tables.** Where the route tables were
  timed on CUDA 12.8, 13.0 and 13.2, each of those releases keeps exactly
  the routes it was measured with.
- **Any other toolkit follows the nearest older measured release**
  (`serving_toolkit`: 12.9 follows 12.8, 13.1 follows 13.0, 13.3 and newer
  follow 13.2). A route whose bits must equal a reference route (an
  Ada-found inference route, a scalar plan with frozen evidence, an SM120
  scalar route) is proven against that reference at first use on such a
  toolkit, as it is on another architecture.
- The Mamba-1 and Mamba-3 fold transports have no first-use proof yet, so
  they stay on the measured toolkits; elsewhere the ordinary fold serves.

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
- The Mamba-1 and Mamba-3 fold transports stay on the measured toolkits
  until they have a first-use proof.
- The Mamba scans and the shared kernels still let the driver's assembler
  fuse their plain arithmetic.
- The SM90a and SM100 TF32 cohorts are empty: Hopper and datacenter
  Blackwell run the portable ladder until a tournament on one of their cards.
- The SM120 table comes from the tournament on an RTX PRO 6000. It has not
  yet been timed against the RTX 5090's own former winners (rule 6); three
  shapes the 5090 had cells for are now served below the table.

Still keyed to one card's count, until an SM120 card re-measures them:
- the SM120 neighbour rule bounds its tile swap and stream-K decline at 170
  tiles;
- three SM120 exact-f32 overlays cut or fold a reduction their own way, so
  exact bits on SM120 can differ from the other architectures.

Repaired:
- the TF32 evidence cohorts are keyed by architecture (one cohort per
  architecture, any SM count, toolkit or driver);
- the SM120 exact-f32 measured cells and generic boundary serve every SM120
  card;
- the exact-family split decisions read tile and CTA counts of the shape
  alone (144 and 432 tiles, a 284-CTA split-M target, the RTX 6000 Ada's
  measured cuts); the per-architecture partition reference is gone;
- the Ada half cohort, the SM120 scalar resource check and every inference
  selector serve every card of their architecture;
- the portable tc64 stream-K takes a reduction of 2048 rows or more over at
  most 48 tiles, or 8192 rows or more over at most 192, and deals one worker
  per eight 64-row slabs, at most 512 workers; the resident grid walks the
  workers in order, so the partition points and the bits do not depend on
  how many SMs the card has;
- the SM120 stream-K kernels (bf16, f16 and TF32) deal one worker per 32
  units, at most 1024 workers, by the shape alone; measured on an RTX 5090
  at its 170 SMs and with the grid held to 84 and 48 CTAs, no grid ran more
  than 6% behind its best schedule, and a deep gradient over few tiles no
  longer folds a slab per unit (127 -> 24 us on 128 x 128 over 8192 rows);
- the exact narrow weight gradient (2 to 127 outputs) splits its batch by
  the shape rule of the wider ones on every architecture, where only three
  RTX 5090 cells split before;
- a proof-path half cell that its wave guard declines falls back to the
  tiled reference it proves against, never to an automatic pick of another
  reduction order;
- the dispatch policy digest no longer carries the SM count;
- the qualification tools run on any card of their architecture and record
  the measuring card's SM count as provenance only.
