# Ada Triad half-TN B-ldmatrix x4 resource stop — 2026-09-09

## Decision

STOP before exact or timing. The test-only F16 TN d768-in candidate replaces
the retained regpipe+vec2 body's four B `ldmatrix.x2.trans` loads with two
`ldmatrix.x4.trans` loads. Both the loop form and a refined literal-address form
compile to 137 registers on CUDA13.2, above the predeclared hard cap of128.
Do not wire it into production, run BF16/siblings, or sweep this unchanged
mechanism.

## Mechanism and proof before the resource gate

- Exhaustive all-32-lane mapping proves each x4 result register maps to the
  original adjacent x2 fragment/register.
- XOR-layout addresses are naturally aligned for every K16 step, warp-N,
  pair and lane.
- The source transform is exactly reversible and changes only B fragment loads
  plus the test export. MMA order, S2 ring, fragment regpipe, cp.async barriers
  and vec2 epilogue remain byte-identical.
- Native source/harness tests: 20/20 PASS.
- CUDA13.2 NVRTC candidate and retained compile-only: PASS.

The first form used an unrolled `pair` loop. It compiled to137 regs, local0,
static shared32768 and occupancy3. A single bounded refinement replaced that
loop with two literal x4 loads, literal destinations and short-lived address
scopes. It compiled to the same137 regs/local0/shared32768/occupancy3. This
disproves loop-index lifetime as the source of the nine-register excess; the
x4 fragment allocation itself is the likely cost in this retained body.

Because the declared gate was registers `<=128`, local0, static shared32768 and
occupancy `>=3`, both attempts stopped before target/tail/exception/K0 execution,
paired timing or cuBLAS Fast. Occupancy remaining at3 is recorded but does not
retroactively relax the gate.

## Evidence identity

- attempt1 log SHA256 `d244e0f13f64f5a73fafab6d87218b2b0868a3706e97eaa8a8908debc60be18b`
- attempt2 log SHA256 `524d00bf2e53eeb411e2f8ad4a67ba78f946c57973fc57ab5f92c54a24df7e8b`
- final source adapter SHA256 `418f748f4392ea830f4d200f3f5cf199710a696d2cdda9bf7a753e21012a1ff4`
- standalone harness SHA256 `2268210b8ad9a6170e8750bff7becfbf571432a2c2fd1f472fff9d5ac237db2a`
- refined composed candidate source SHA256 `2c34f61556305a1f8a656654891313f3973437a6c7697c2987eed8f132676d4d`
- refined PTX SHA256 `db2fe69952a077557b8c7d86a955311093b8c5236936390b26423d0948fb4aa8`

Immediate external and in-harness GPU quiet gates passed with ample VRAM; see
`preflight.txt` and both raw logs. Production and dispatcher were unchanged.
