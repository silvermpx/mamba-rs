# Ada Fixed RNA-wide production AUTO

This checkpoint promotes the already-qualified RNA-wide kernel from parent
`82fc400f` into actual Fixed AUTO. It is not all-inference release closure.
No CUDA bodies, compiler/composer/numeric/schedule revisions or unrelated
Triad/SM120 selector/qualification hashes changed. Global tuning epoch39->40
requires old graphs to be recaptured; all three Ada compiled cache blobs are
byte-identical to the force checkpoint, independently rehashed by main.

## Admission and numerical contract

Actual loaded optional RNA holder, CC8.9/142SM, known NVRTC13.2, homogeneous
F32 under `AllowDeterministicTf32V1`, nonnull C/A/B16-aligned, optional bias4.
Only the five shapes below, each with/without bias, enter AUTO. C4-only output,
shifted A/B, other shapes/devices/toolkits/policies retain the old picker.
Launch errors propagate; no catch-and-retry. Every old force route is retained.

RNA symbol: `gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3`.
One256-thread launch,98,304dynamic shared, five arguments/final32-byte bundle.
Explicit RNA conversion, bias preseed and the existing ascending k8 order remain
bit-identical to all five ordinary Fixed TF32 rungs in the qualified corpus.

Shapes `(M,K,N)`: A=(4621,384,1928), B=(4621,768,2304),
C=(4621,1928,384), D=(2048,768,2304), E=(2048,2304,768).
Suffix0 means no bias; suffix1 means bias.

## Quiet post-AUTO101 result

RTX6000Ada, driver595.45.04, NVRTC13.2, pinned1800MHz. The final run started
only after architecture compilation exited; preflight records0%GPU,90MiB,41C.
40 unique records = five shapes x two biases x eager/graph x two orders.
Each arm has101 samples. Zero rejections; completion, raw/repeat and all20
applicable graph replay flags pass. Every physical custom graph proves actual
AUTO RNA and old forced M64S2, correct grid/block/shared and one kernel.
Eager records use graph-replay=false as N/A, not as a failed replay.

Main independently checked the matrix, identities, graph geometry and flags,
then recomputed the following matched samplewise ratios from raw arrays.
Quantiles use nearest-index `round((n-1)*q)`; each entry is the worst over
four path/order cohorts. Lower is better; AUTO/old p95 is NOT an inverse of
old/AUTO p95.

| Cell | AUTO/old p50 | AUTO/old p95 | AUTO/FAST p50 | AUTO/FAST p95 |
|---|---:|---:|---:|---:|
| A0 | 0.828687 | 0.848324 | 1.080677 | 1.108626 |
| A1 | 0.837823 | 0.856835 | 0.868782 | 0.904797 |
| B0 | 0.726012 | 0.732621 | 1.098057 | 1.110621 |
| B1 | 0.730103 | 0.738436 | 0.966568 | 0.980568 |
| C0 | 0.621531 | 0.626263 | 1.314221 | 1.334471 |
| C1 | 0.619660 | 0.628503 | 1.207602 | 1.213196 |
| D0 | 0.896758 | 0.911599 | 1.262692 | 1.301159 |
| D1 | 0.899866 | 0.913177 | 1.091251 | 1.126994 |
| E0 | 0.870655 | 0.885280 | 1.307817 | 1.342478 |
| E1 | 0.873246 | 0.887069 | 1.225008 | 1.254475 |

All ten cases beat old AUTO at both quantiles in every cohort. Only A1/B1
also beat FAST. Remaining eight cases still lose FAST; worst AUTO/FAST p95
is1.342478. FAST is explicit `CUBLAS_COMPUTE_32F_FAST_TF32` with the required
bias broadcast timed. PEDANTIC is the independent numerical reference only.

Final log: `confirm101-old-m64-vs-auto-rna-final.log`, SHA256
`3dbb8754b651706d67c83ab1f96fb26ed66a63bc161e8ced015f36ef50831f46`.
Final performance binary SHA256
`2dfbc274dfc1773939491fe2fbc0eb86d6536c7bb6f380720b448898d1cc68e2`.
Fixed source `7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301`;
artifact `c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f`.
Full compiler/cache identities and commands are in `integration-report.md`.

## Verification and truthful treatment of intermediate failures

- Final library633 passed/46 ignored; main independently reran633/633 in1.04s.
- Actual AUTO GPU matrix2/2: A-E/both biases, finite/nonrepresentable/exceptional
  data, all-five-rung bits, M boundaries, row/C4/shifted views, guards/input
  immutability, eager repeats and poisoned graphs with real ABI/geometry.
  Main independently reran hot-A actual AUTO/graph1/1 in2.76s.
- Retained C prefix/special-bias/graph1/1; actual Ada Triad cohorts2/2;
  performance static43/43. Direct rustfmt and diff whitespace checks pass.
- Full architecture suite:56 passed/1 failed in1432.60s. All real target
  compile/assembly/resource gates passed. The failure was an obsolete
  source assertion forbidding the already-public SM120 PairStore tile;
  it also fails on frozen82fc. The corrected focused test passed1/1 after
  an arch-only rebuild. Do not describe this as a fresh full57/57 rerun.
- The other baseline composition fixture and epoch tuple were corrected;
  original RED/intermediate failure logs remain unchanged. In particular
  `green-lib-all.log` is the earlier631-pass/2-failure run, not acceptance.
- `inadmissible-overlap-confirm101-old-m64-vs-auto-rna.log` overlapped host
  NVRTC compilation and is explicitly excluded from all performance claims.
- Independent eight-file spec/quality review and the ninth-file fixture
  amendment review are clean. Main verified all nine local/remote source
  hashes and preserved performance/force binaries; no kernels were deleted.

CUDA12.8/13.0 and other physical GPUs were not newly runtime-qualified here.
The next toolkit census is separate; hardware-specific performance and
cross-toolkit equality must not be inferred from these13.2 Ada results.
