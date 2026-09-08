# Ada TF32 NN N96 direct-epilogue screen — 2026-09-09

## Decision

Retain the test-only direct-`float2` epilogue as the new Prism retained-best
candidate. It beats the prior N96 kernel in all four paired strata, but it does
not yet beat cuBLAS Fast in every eager/graph stratum. Do not claim a Fast win
or production admission.

The same candidate is a valid stop for d768-out: its 0.2–0.4% gain does not
clear the strict `<0.99` retained-best gate, so Fast timing was correctly
skipped for that cell.

## Mechanism and correctness

On complete aligned M128xN96 tiles, the candidate stores each lane's two
contiguous accumulators directly as one `float2`. It removes the shared
accumulator round trip and second CTA barrier while keeping the retained MMA
and conversion order. Tails and other unsupported conditions retain the old
path.

- Full-tile finite and exceptional bits pass against retained N96.
- Tail finite and exceptional bits, K0, guards, eager replay and graph replay
  pass.
- Candidate and retained both use 124 registers, 0 local bytes, 86,016 dynamic
  shared bytes and one active block/SM.

## Paired once7 results

Candidate/retained N96:

| Cell | eager p50/p95 | graph p50/p95 | Decision |
| --- | --- | --- | --- |
| d768-out | .99598–.99719 | .99758–.99838 | stop; below 1% admission threshold |
| Prism | .98584–.98949 | .98235–.98434 | retain new best |

For Prism, candidate/cuBLAS Fast is 1.01166–1.01359 eager and
.99614–.99971 graph. The candidate therefore still misses a strict Fast win,
principally by about 1.2–1.3% in eager execution.

## Environment and evidence

- RTX 6000 Ada, SM89, 142 SMs, CUDA 13.2, 1800 MHz.
- User-authorized idle-resident mode required five consecutive samples at no
  more than 1% compute and memory utilization before every timed cohort.
  Pre-context free memory was 2,669 MiB; timed/post samples retained at least
  2,056 MiB. External CUDA contexts were not stopped.
- Raw receipt: `raw.log`.
- Harness SHA-256:
  `b5e5c5c7b20275d0054c705676b9cd8ad23dd7288310bb96bef6570c17782ea1`.
- Source-adapter SHA-256:
  `07afca439cafddea86f94f9e162ed2f1c7295d44ff26d3004c8bbc3620c59f57`.
- Composed CUDA source SHA-256 reported by the harness:
  `f53d17362d3fae07848954544ec85233480336babc0790b2092a2cfd4ddd7069`.
