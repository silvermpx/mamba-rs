# TF32 NT shared-stage RNA: first resource stop

Ada/CUDA13.2, NT d768-in(2048,768,3072). First attempt fails the real resource
gate **before bit checks and before timing**:138regs, local/static0,49,152B
dynamic shared,256 threads, occupancy1 versus required2. This is a resource
loss, not a measured runtime loss and not a bit-correctness result.

The test-only helper converts producer-owned shared words after their thread's
async wait and before the CTA publishing barrier. It retains K8 MMA order,
RNA rounding, stage-reuse barriers and the original unconverted scalar fallback.
Native helper5/main76 checks pass; root independently ran8 related native checks
and reviewed the actual producer/ring code. Those do not replace GPU bit checks.

Frozen main SHA256 `03e6aa2506d939fd6d548e50288107485bcacb9a33fa3bacde23a225c5d1459f`;
helper `96ae0a10f57f90632d3d578d1a59de59d1b0c0d563beb3bd6b90e9002855b1af`;
composed CUDA `2e32fa327d62184551d38e0284cffc128918eef2cadbe5ea399216ce6d684b80`.
Exact test: `cuda_suite::ada_tf32_nt_shared_stage_rna_d768_in_discovery_once7`.
[Run1 raw](evidence/cuda132/run1/test.log) SHA256
`c06adf71d82c7fc4d144167d0b8171c6cf86e9a9ff3b0ba86226f8a83bd5b5a3`.
Build/list pass, exact1 fails only at the resource gate. PRE/RELEASE/DRAIN
receipts are quiet/no-apps, private cache hashes unchanged.

Do not rerun this unchanged source or relax occupancy2. One distinct refinement
sets target-only launch bounds to256threads/minBlocks2 to constrain register
scheduling; keep local0 mandatory. NVIDIA documents that this may increase
instructions or spills, so its outcome is unknown. Retained A-ldmatrix stays
the best NTin candidate until a new paired comparison proves otherwise.
No production selector, Fixed inference or SM120 route changed.
