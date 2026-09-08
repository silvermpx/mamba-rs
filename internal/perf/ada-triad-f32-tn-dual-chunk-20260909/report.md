# Ada exact-F32 TN dual-chunk fused screen — 2026-09-09

## Decision

New retained-best for exact-F32 TN d768-in. A single CTA computes the two
fixed 1024-row chunks and performs the existing fixed-order FP64 finalize,
reducing the retained three-node pipeline to two nodes. It beats the retained
N64 GROUP_M8 fused pipeline by 4.0–4.4% in every eager/graph and ABBA/BAAB
stratum.

This is not a cuBLAS Fast win: the candidate remains 2.30–2.33x slower.
Preserve it for the whole-Triad integration batch; production routes and the
dispatcher remain unchanged during discovery.

## Frozen source

- Harness SHA-256:
  `e6f4e9de23249ce9327b1ba49850c88c1e94e1bc911f2e86ca65151a14a3899e`
- Source adapter SHA-256:
  `eded5589bd7370876a31e8ebeb541278a1336352f6f9e0346f181759dbbf0111`
- Candidate graph: transpose -> dual-chunk fused finalize.
- The change removes the intermediate partial0 global write/read and one
  kernel launch while preserving both chunk accumulation streams and the
  established fixed-order finalizer.

## Correctness, SASS and resources

- Raw chunk planes match the retained SplitM oracle bit-for-bit.
- Target, tail, exceptional payload, non-unit alpha, K0, input/guard checks,
  eager and captured graph: PASS.
- Both raw and fused symbols contain FFMA and LDGSTS and contain no
  LDL/STL/ATOM/RED/REDUX; stack and spill bytes are zero.
- Raw and fused symbols: 163 registers/thread, local0, 32,768 B static shared,
  occupancy3, 128 threads. The hard budgets were <=168 registers and
  occupancy>=3.
- The CUDA 13.2 `nvdisasm` parser now accepts `.text.<symbol>:` function
  labels as well as the older `Function : <symbol>` form; a regression test
  covers this tooling-only compatibility fix.

## Paired once7 timing versus retained-best

Ratios are candidate / retained N64 GROUP_M8 fused pipeline on d768-in
`(2048, 768, 3072)`, 20 logical GEMMs per observation.

| Path / order | p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 0.956167 | 0.959391 |
| eager / BAAB | 0.956802 | 0.957789 |
| graph / ABBA | 0.959370 | 0.961054 |
| graph / BAAB | 0.959431 | 0.959700 |

Candidate observations are roughly 317–319 us versus retained 331–334 us.

## Paired once7 timing versus cuBLAS Fast

| Path / order | p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 2.305506 | 2.317487 |
| eager / BAAB | 2.304083 | 2.312025 |
| graph / ABBA | 2.326599 | 2.332147 |
| graph / BAAB | 2.327133 | 2.332459 |

## GPU isolation

The user-owned models stayed resident. Every phase used the idle-resident
preflight: five consecutive samples at no more than 1% compute and memory
utilization plus an explicit free-VRAM floor. Pre/post utilization was 0%/0%;
free memory was 2669 MiB before setup and 1992 MiB after the test.

Raw output: [raw.log](raw.log).
