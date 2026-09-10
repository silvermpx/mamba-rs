# RTX 5090 retained TF32 qualification, CUDA 12.8

The prospective V2 packet `all24-v2/` completed at 2026-09-10 03:40:03 UTC
on the current RTX 5090 / driver 595.84. It contains all 24 intended public
operation/shape keys, with 23 admitted and only `tn_m8192_k128_n128`
(G10, actual TN dimensions M=8192, K=128, N=128) rejected for performance.
The G11 `nt_split_candidate` key is admitted. A passing test process alone
does not mean its candidate was admitted.

Root independently replayed all 21/101 sample statistics, qualification
decisions and completion counts using:

```sh
jq -se --argjson count 24 --argjson cuda '[12,8]' \
  -f internal/perf/sm120-tf32-retained-20260910/verify.jq \
  internal/perf/sm120-tf32-retained-20260910/cuda128/all24-v2/G*.jsonl
```

Result: `true`. All 11 completion SHA-256 values match their original cell
record bytes. The 23 admitted records have one identical specialized
identity and one identical portable identity. These are the live CUDA 12.8
identity inputs for dispatcher integration, not identities borrowed from
another toolkit. The source, runner, executable and device manifests are
preserved beside the original logs. The aborted earlier startup packet is
not an admission receipt.

This packet measures retained candidates against the exact fallback; it is
not the final AUTO-versus-cuBLAS release comparison. At acquisition time,
these fresh lower-toolkit identities had not yet been added to production
AUTO. Actual dispatcher selection is checked separately after integration.
