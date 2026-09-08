# TF32 NT shared-stage RNA: resource refinement passes, runtime loses

Latest verdict: the target-only minBlocks2 refinement below restores occupancy2
without spills and passes focused GPU bits, but is35.4–35.5% slower than retained
A-ldmatrix. **Stop this candidate; retain A-ldmatrix.** No unchanged retry or
additional profile of this decisive loss.

## First attempt: resource stop

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
scheduling; keep local0 mandatory. NVIDIA's
[launch-bound guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
explains that this may increase
instructions or spills, so its outcome is unknown. Retained A-ldmatrix stays
the best NTin candidate until a new paired comparison proves otherwise.
No production selector, Fixed inference or SM120 route changed.

## Run2: target-only minBlocks2 refinement

Main unchanged03e6; helper SHA256
`2fbf7d75e102553ebfc669ee3ea477bdaa89884174ebfef9a80dd96fa025f3a9`.
Only target launch bounds change from256/1 to256/2; other exports and arithmetic
stay unchanged. Genuine native RED->GREEN5/5; root reviewed and ran those5.
GPU resources125regs, local/static0, dynamic49,152B, occupancy2. Target actual
AUTO, forced tail/exception, direct K0, eager/graph repeat and guards pass.

| Candidate / comparator | Eager p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| Retained A-ldmatrix | 1.35420–1.35452 | 1.35478–1.35500 | 1.356999 |
| cuBLAS FAST_TF32 | 2.27223–2.29064 | 2.31877–2.31975 | 2.320929 |

Candidate~272us versus retained~201us. All8 once7 strata use20 complete GEMMs
per observation, explicit bracket arms and pre-event C/A/B resets. AUTO is the
identity/bit oracle only, not redundantly timed. Root replayed56brackets/
224observations and all nearest-rank quantiles. Cache unchanged, quiet/no-apps
PRE/RELEASE/DRAIN receipts preserved. No robust Fast win or production promotion.

[Run2 raw](evidence/cuda132/run2-lb2/test.log) SHA256
`450145bf064a6a15e9edaf66662b9d6e94c2a56f8fc337758c8cfdecf099f115`;
composed CUDA `4b081e0c512f3b58399e9f439a6f4f5a7458d0adf807025ad92b7fbd9d7a15ef`;
binary `30da4df53e1addec5fc119eda86df41c6484b0b0a16d814d8c768cd7c2e9d071`.
