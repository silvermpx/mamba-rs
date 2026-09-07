# Independent pre-implementation TN mapping audit

Production SM80 source SHA
74af0ed307f0216e8dd4dd370937242f32e3bcf9d59e60ce64d30f3a96d6330d.
Read-only audit by implement_ada_half_s3_auto,2026-09-07T21:41Z.

Actual padded A/B bank index is `(axis + 8*reduction)&31`, from strides136/72.
Packed XOR bank index is `(axis ^ ((reduction&3)<<3))&31`.
For each actual scalar fragment load, group=lane>>2 and thread=lane&3;
reduction=k8+thread or +4, and axis=aligned_base+group. Low3 bank bits are
group; upper2 are a permutation of thread. Exhaustive all8warps/issues and
4A/2B fragment formulas found32distinctbanks for each warp instruction.

Copy starts have axis%4=0; mask%8=0 preserves each aligned contiguous16B
chunk. Per-reduction XOR is bijective within128 A/64 B axis slots. Existing
TN async staging retains4A+2B copies/thread/stage. Both readers and writers
must apply the identical mapping; this is a source-review condition.

Eight compute warps with MAtoms2/NAtoms4, wm=(warp>>1)*32,wn=(warp&1)*32
cover128x64 once. The old-to-new ownership mapping preserves the lane and
per-output K association: rowbase=64*(old_warp>>1)+16*old_m_atom;
new_warp=2*(rowbase/32)+(old_warp&1),new_m_atom=(rowbase%32)/16.

This does not prove speedup. Grid remains93 for142 SMs; occupancy2 alone
creates no additional CTAs. Eight warps duplicate more B fragment work than
the old four compute warps. Resource/correctness and paired timing decide.

## Frozen combined-source review and root replay

At21:58Z the independent auditor accepted source/spec and code quality with
no blocking findings. The runtime correctly uses TN C=k*n,A=m*k,B=m*n and
strides(k,n,n), asserts actualAUTO TN S3/79872 and93CTAs, and uses forced RNA
for tail/negative-alpha/K0. Each timed arm restores C+A+B before its one-GEMM
event, then downloads/checks output+inputs+guards before the next reset.
Four7-window eager/graph ABBA/BAAB strata keep true execution-order raw arrays.
The only deferred cosmetic issue is inherited NT wording in some error strings.

Frozen main harness SHA
fadf33fdfe384af278abb5ac261a7b089e7b5b8a58799bf2a3c3f4aa40e8c4ab.
Root independently compiled the native test with rustc1.98.1/edition2024 and
executed all28tests:28passed/0failed. Two native-only dead-code warnings cover
the CUDA-only TN variant/methods; there were no native compilation errors.
The separate helper was independently compiled/run too:2passed/0failed.
CUDA13.2 owner build passed in45.32s and four focused host tests passed; this
section alone is not CUDA candidate-load, correctness or timing evidence.
