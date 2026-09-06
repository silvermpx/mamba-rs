# Partial paired TF32 checkpoint

The RTX5090 rental was nearly exhausted while the strict paired test was still
running. This checkpoint preserves the first 16 of 24 JSONL records copied
from the live box. It is intentionally labelled partial: the runner had not
exited and the third cell was still in Driver/JIT setup at capture time.

The completed `d768_in_proj` and `d768_out_proj` cells both pass numeric,
guard, AUTO-versus-forced bit, and eager-versus-graph raw-repeat checks. AUTO
and the literal forced route beat installed cuBLAS FAST in every eager/graph
ABBA/BAAB cohort (21 windows):

- `d768_in_proj`: AUTO worst p95 0.902121; forced worst p95 0.902750;
  actual forced symbol uses the SM120 M64N128 TF32 TMA tile.
- `d768_out_proj`: AUTO worst p95 0.928441; forced worst p95 0.927627;
  actual forced symbol uses the SM120 M64N64 TF32 TMA tile.

These are screening wins, not dispatcher admission: the emitted records keep
`dispatch_admission=false`. Long confirmation and the unfinished third cell
remain required.
